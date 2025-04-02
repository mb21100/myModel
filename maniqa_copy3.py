import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List
from timm.models.swin_transformer import SwinTransformerBlock
from thop import profile

########################################
# CBAM 관련 모듈
########################################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)

class FusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.cbam = CBAM(channels)
    
    def forward(self, feat_local, feat_global):
        x = torch.cat([feat_local, feat_global], dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.act(x)
        return self.cbam(x)

########################################
# Self-Attention Fusion Module (using Adaptive Pooling)
########################################
class SwinAttentionFusionAvg(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int,
                 window_size: int = 7, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.window_size = window_size
        # 각 stage feature map을 1x1 conv로 out_channels로 맞춤
        self.projs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1)
                                     for in_ch in in_channels_list])
        # Swin Transformer Block: input_resolution 인자를 추가하고, 채널-마지막 형식(NHWC)을 기대함
        self.blocks = nn.Sequential(*[
            timm.models.swin_transformer.SwinTransformerBlock(
                dim=out_channels,
                input_resolution=(window_size, window_size),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=4,
                drop_path=dropout,
                attn_drop=dropout
            )
            for i in range(num_layers)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B = features[0].size(0)
        proj_feats = []
        for proj, feat in zip(self.projs, features):
            # 각 feature map을 window_size×window_size로 adaptive pooling
            pooled = F.adaptive_avg_pool2d(feat, (self.window_size, self.window_size))
            proj_feats.append(proj(pooled))  # [B, out_channels, window_size, window_size]
        fused = torch.stack(proj_feats, dim=0).mean(dim=0)  # [B, out_channels, window_size, window_size]
        # NCHW -> NHWC 변환 (SwinTransformerBlock은 채널-마지막 형식을 기대)
        fused = fused.permute(0, 2, 3, 1).contiguous()  # [B, window_size, window_size, out_channels]
        out = self.blocks(fused)  # Swin 블록 적용, output: [B, window_size, window_size, out_channels]
        # 다시 NHWC -> NCHW 변환
        out = out.permute(0, 3, 1, 2).contiguous()  # [B, out_channels, window_size, window_size]
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(B, -1)  # [B, out_channels]
        return out


########################################
# MANIQA_HF (Wavelet 제거 버전)
########################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
        super().__init__()
        fusion_out_channels = 256  # 공통 fusion 차원
        
        # MaxViT backbone (features_only=True)
        self.backbone = timm.create_model(
            'maxvit_rmlp_small_rw_224.sw_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1,2,3,4)
        )
        self.first_mbconv_features = {}
        self._register_first_mbconv_hooks()
        
        # 각 stage 정보 추출
        feat_infos = self.backbone.feature_info[1:5]
        self.stage_indices = []
        self.feat_channels = []
        for info in feat_infos:
            stage_idx = int(info['module'].split('.')[1])
            self.stage_indices.append(stage_idx)
            self.feat_channels.append(info['num_chs'])
        self.num_stages = len(self.feat_channels)
        
        # CBAM 기반 Fusion
        self.fusion_modules = nn.ModuleDict({
            str(stage_idx): FusionModule(channels)
            for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
        })
        self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
        self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
                        
        self.self_attn_fusion = SwinAttentionFusionAvg(
            in_channels_list=self.feat_channels,
            out_channels=fusion_out_channels,
            window_size=7,
            num_heads=4,
            num_layers=2,
            dropout=drop
        )


        
        self.mlp_basic = nn.Sequential(
            nn.Linear(fusion_out_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        # 최종 head (Wavelet 없이 basic_feature만 사용)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_outputs)
        )
    
    def _register_first_mbconv_hooks(self):
        for name, module in self.backbone.named_modules():
            if name.startswith("stages_"):
                parts = name.split('.')
                # "stages_<stage_idx>.blocks.0.conv.drop_path" 형태 찾기
                if (len(parts) >= 5 and
                    parts[1] == "blocks" and
                    parts[2] == "0" and
                    parts[3] == "conv" and
                    parts[4] == "drop_path"):
                    try:
                        stage_idx = int(parts[0].replace("stages_", ""))
                    except Exception:
                        continue
                    module.register_forward_hook(
                        lambda m, inp, out, stage_idx=stage_idx:
                            self._first_mbconv_hook(stage_idx, m, inp, out)
                    )
    
    def _first_mbconv_hook(self, stage_idx, module, inp, out):
        if stage_idx not in self.first_mbconv_features:
            self.first_mbconv_features[stage_idx] = out
    
    def forward(self, x):
        # Backbone에서 feature 추출
        self.first_mbconv_features = {}
        backbone_feats = self.backbone(x)
        final_stage_features = {}
        for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
            stage_idx = int(info['module'].split('.')[1])
            final_stage_features[stage_idx] = feat
        
        # CBAM Fusion
        fused_stage_features = []
        for i, stage_idx in enumerate(self.stage_indices):
            feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
            feat_global = final_stage_features[stage_idx]
            fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
            fused_feat = self.stage_norms[i](fused_feat)
            fused_feat = self.stage_drops[i](fused_feat)
            fused_stage_features.append(fused_feat)
        
        # Self-Attention Fusion으로 여러 스테이지 특징 융합
        basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
        basic_feature = self.mlp_basic(basic_feature)                # [B, hidden_dim]
        
        # 최종 score 예측
        score = self.fusion_head(basic_feature).squeeze(-1)
        return torch.sigmoid(score)


if __name__ == "__main__":
    model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=768)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print("FLOPs: {:.3f} G".format(flops / 1e9))
    print("Parameters: {:.3f} M".format(params / 1e6))
    with torch.no_grad():
        output = model(dummy_input)
    print("Output score shape:", output.shape)
    if model.first_mbconv_features:
        print("Extracted features from conv.drop_path of each stage:")
        for stage_idx, feat in model.first_mbconv_features.items():
            print(f"  Stage {stage_idx}: feature shape = {feat.shape}")
    else:
        print("No features were extracted by hooks.")
    backbone_feats = model.backbone(dummy_input)
    print("Backbone final feature maps:")
    for idx, feat in enumerate(backbone_feats, 1):
        print(f"  Stage {idx}: feature shape = {feat.shape}")