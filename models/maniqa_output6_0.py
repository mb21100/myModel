# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import timm


import math

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
        #self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        #return x * self.spatial_attention(x)
        return x

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

# ########################################
# # Self-Attention Fusion Module (using Adaptive Pooling)
# ########################################
# class SelfAttentionFusionAvg(nn.Module):
#     def __init__(self, in_channels_list: List[int], out_channels: int,
#                  patch_size: tuple = (7, 7), num_heads: int = 4,
#                  num_layers: int = 2, dropout: float = 0.2):
#         """
#         in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [256, 512, 1024, 2048])
#         out_channels: 모든 feature map을 투영할 공통 채널 수 (예: 256)
#         patch_size: 각 feature map을 Adaptive Pooling할 목표 공간 크기 (예: (7,7))
#         """
#         super().__init__()
#         self.patch_size = patch_size
#         self.projs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=out_channels,
#             nhead=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         """
#         features: list of 4 feature maps, 각각 shape: [B, C_i, H_i, W_i]
#         반환: fused feature vector, shape [B, out_channels]
#         """
#         B = features[0].size(0)
#         tokens = []
#         for proj, feat in zip(self.projs, features):
#             pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
#             proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
#             _, C, H, W = proj_feat.shape
#             token = proj_feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, out_channels]
#             tokens.append(token)

#         token_seq = torch.cat(tokens, dim=1)  # [B, total_tokens, out_channels]
#         token_seq = self.transformer_encoder(token_seq)  # [B, total_tokens, out_channels]
#         fused_feature = token_seq.mean(dim=1)  # [B, out_channels]
#         return fused_feature

# ########################################
# # MANIQA
# ########################################
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
#         super().__init__()
#         fusion_out_channels = 512  # 공통 fusion 차원
        
#         # MaxViT backbone (features_only=True)
#         self.backbone = timm.create_model(
#             'maxvit_rmlp_small_rw_224.sw_in1k',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1,2,3,4)
#         )
#         self.first_mbconv_features = {}
#         self._register_first_mbconv_hooks()
        
#         # 각 stage 정보 추출
#         feat_infos = self.backbone.feature_info[1:5]
#         self.stage_indices = []
#         self.feat_channels = []
#         for info in feat_infos:
#             stage_idx = int(info['module'].split('.')[1])
#             self.stage_indices.append(stage_idx)
#             self.feat_channels.append(info['num_chs'])
#         self.num_stages = len(self.feat_channels)
        
#         # CBAM 기반 Fusion
#         self.fusion_modules = nn.ModuleDict({
#             str(stage_idx): FusionModule(channels)
#             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })
#         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
#         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
#         # Self-Attention Fusion 모듈
#         self.self_attn_fusion = SelfAttentionFusionAvg(
#             in_channels_list=self.feat_channels,
#             out_channels=fusion_out_channels,
#             patch_size=(7, 7),
#             num_heads=4,
#             num_layers=2,
#             dropout=drop
#         )
        
#         self.mlp_basic = nn.Sequential(
#             nn.Linear(fusion_out_channels, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop)
#         )
        
#         # 최종 head (Wavelet 없이 basic_feature만 사용)
#         self.fusion_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, num_outputs)
#         )
    
#     def _register_first_mbconv_hooks(self):
#         for name, module in self.backbone.named_modules():
#             if name.startswith("stages_"):
#                 parts = name.split('.')
#                 # "stages_<stage_idx>.blocks.0.conv.drop_path" 형태 찾기
#                 if (len(parts) >= 5 and
#                     parts[1] == "blocks" and
#                     parts[2] == "0" and
#                     parts[3] == "conv" and
#                     parts[4] == "drop_path"):
#                     try:
#                         stage_idx = int(parts[0].replace("stages_", ""))
#                     except Exception:
#                         continue
#                     module.register_forward_hook(
#                         lambda m, inp, out, stage_idx=stage_idx:
#                             self._first_mbconv_hook(stage_idx, m, inp, out)
#                     )
    
#     def _first_mbconv_hook(self, stage_idx, module, inp, out):
#         if stage_idx not in self.first_mbconv_features:
#             self.first_mbconv_features[stage_idx] = out
    
#     def forward(self, x):
#         # Backbone에서 feature 추출
#         self.first_mbconv_features = {}
#         backbone_feats = self.backbone(x)
#         final_stage_features = {}
#         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
#             stage_idx = int(info['module'].split('.')[1])
#             final_stage_features[stage_idx] = feat
        
#         # CBAM Fusion
#         fused_stage_features = []
#         for i, stage_idx in enumerate(self.stage_indices):
#             feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
#             feat_global = final_stage_features[stage_idx]
#             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
#             fused_feat = self.stage_norms[i](fused_feat)
#             fused_feat = self.stage_drops[i](fused_feat)
#             fused_stage_features.append(fused_feat)
        
#         # Self-Attention Fusion으로 여러 스테이지 특징 융합
#         basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
#         basic_feature = self.mlp_basic(basic_feature)                # [B, hidden_dim]
        
#         # 최종 score 예측
#         score = self.fusion_head(basic_feature).squeeze(-1)
#         return torch.sigmoid(score)



########################################
# Relative Multi-Head Attention with Relative Positional Bias
########################################
class RelativeMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, window_size=(14,14)):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        H, W = window_size
        L = H * W
        # 상대적 위치 bias 테이블: ((2H-1)*(2W-1)) x nhead
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*H-1)*(2*W-1), nhead))
        # 좌표 계산
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, H, W]
        coords_flatten = torch.flatten(coords, 1)  # [2, L]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, L, L]
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # [L, L, 2]
        relative_coords[:,:,0] += H - 1
        relative_coords[:,:,1] += W - 1
        relative_coords[:,:,0] *= 2*W - 1
        relative_position_index = relative_coords.sum(-1)  # [L, L]
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x):
        # x: [B, L, d_model]
        B, L, C = x.shape
        qkv = self.qkv(x)  # [B, L, 3*d_model]
        qkv = qkv.reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, nhead, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, nhead, L, head_dim]
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, nhead, L, L]
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.relative_position_index.shape[0], self.relative_position_index.shape[1], self.nhead)
        # relative_bias: [L, L, nhead] -> [1, nhead, L, L]
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, nhead, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out

########################################
# Relative Transformer Encoder Layer
########################################
class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size=(14,14), mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout, window_size=window_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

########################################
# Relative Self-Attention Fusion Module (with Relative Positional Encoding)
########################################
class RelativeSelfAttentionFusionAvg(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int,
                 patch_size: tuple = (7, 7), total_grid: Optional[tuple] = None,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        """
        in_channels_list: 각 stage feature map의 채널 수 리스트 (예: [256, 512, 1024, 2048])
        out_channels: 모든 feature map을 투영할 공통 채널 수 (예: 256)
        patch_size: 각 stage feature map을 adaptive pooling할 목표 크기 (예: (7,7))
        total_grid: 모든 stage를 concat한 후의 토큰 grid 크기, 예를 들어 4개 stage이면 (14,14)
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_stages = len(in_channels_list)
        if total_grid is None:
            side = int(math.sqrt(self.num_stages))
            total_grid = (patch_size[0] * side, patch_size[1] * side)
        self.total_grid = total_grid  # e.g., (14,14) for 4 stages with patch_size=(7,7)
        self.projs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.layers = nn.ModuleList([
            RelativeTransformerEncoderLayer(d_model=out_channels, nhead=num_heads,
                                              window_size=self.total_grid, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B = features[0].size(0)
        tokens = []
        for proj, feat in zip(self.projs, features):
            pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
            proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
            B, C, H, W = proj_feat.shape
            token = proj_feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, out_channels]
            tokens.append(token)
        # Concatenate tokens from all stages: total_tokens = num_stages * (patch_H * patch_W)
        token_seq = torch.cat(tokens, dim=1)  # [B, total_tokens, out_channels]
        for layer in self.layers:
            token_seq = layer(token_seq)
        fused_feature = token_seq.mean(dim=1)  # [B, out_channels]
        return fused_feature

########################################
# MANIQA 모델 (Wavelet branch 제외, 기본 branch만 사용)
########################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
        super().__init__()
        fusion_out_channels = 512  # 공통 fusion 차원
        
        # MaxViT backbone (features_only=True)
        self.backbone = timm.create_model(
            'maxvit_rmlp_small_rw_224.sw_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1,2,3,4)
        )
        self.first_mbconv_features = {}
        self._register_first_mbconv_hooks()
        
        feat_infos = self.backbone.feature_info[1:5]
        self.stage_indices = []
        self.feat_channels = []
        for info in feat_infos:
            stage_idx = int(info['module'].split('.')[1])
            self.stage_indices.append(stage_idx)
            self.feat_channels.append(info['num_chs'])
        self.num_stages = len(self.feat_channels)
        
        self.fusion_modules = nn.ModuleDict({
            str(stage_idx): FusionModule(channels)
            for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
        })
        self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
        self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
        # Relative Self-Attention Fusion 모듈 (상대적 위치 임베딩 포함)
        # 예를 들어, 각 stage에서 patch_size=(7,7)이고 4개의 stage가 있다면 total_grid=(14,14)
        self.self_attn_fusion = RelativeSelfAttentionFusionAvg(
            in_channels_list=self.feat_channels,
            out_channels=fusion_out_channels,
            patch_size=(7,7),
            total_grid=(14,14),
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
        self.first_mbconv_features = {}
        backbone_feats = self.backbone(x)
        final_stage_features = {}
        for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
            stage_idx = int(info['module'].split('.')[1])
            final_stage_features[stage_idx] = feat
        
        fused_stage_features = []
        for i, stage_idx in enumerate(self.stage_indices):
            feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
            feat_global = final_stage_features[stage_idx]
            fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
            fused_feat = self.stage_norms[i](fused_feat)
            fused_feat = self.stage_drops[i](fused_feat)
            fused_stage_features.append(fused_feat)
        
        basic_feature = self.self_attn_fusion(fused_stage_features)
        basic_feature = self.mlp_basic(basic_feature)
        
        score = self.fusion_head(basic_feature).squeeze(-1)
        return torch.sigmoid(score)

# if __name__ == "__main__":
#     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=768)
#     model.eval()
#     dummy_input = torch.randn(1, 3, 224, 224)
#     from thop import profile
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
#     print("FLOPs: {:.3f} G".format(flops / 1e9))
#     print("Parameters: {:.3f} M".format(params / 1e6))
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("Output score shape:", output.shape)
