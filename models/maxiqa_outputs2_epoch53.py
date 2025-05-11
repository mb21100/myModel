

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# import pywt
# from thop import profile  # pip install thop
# import numpy as np
# from typing import List, Optional

# ########################################
# # (Optional) FPN: multi-scale feature fusion (여기서는 사용하지 않음)
# ########################################
# class FPN(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         self.smooth_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
#         ])

#     def forward(self, features):
#         lateral_features = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
#         fpn_features = [None] * len(lateral_features)
#         fpn_features[-1] = lateral_features[-1]
#         for i in range(len(lateral_features) - 2, -1, -1):
#             upsampled = F.interpolate(fpn_features[i+1],
#                                       size=lateral_features[i].shape[2:],
#                                       mode='bilinear',
#                                       align_corners=False)
#             fpn_features[i] = lateral_features[i] + upsampled
#         fpn_features = [s_conv(f) for f, s_conv in zip(fpn_features, self.smooth_convs)]
#         return fpn_features

# ########################################
# # CBAM 관련 모듈
# ########################################
# # Channel Attention Module
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         return self.sigmoid(avg_out + max_out)

# # Spatial Attention Module
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         return self.sigmoid(self.conv(x_cat))

# # CBAM 모듈: 채널 및 공간 어텐션 결합
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         x = x * self.channel_attention(x)
#         return x * self.spatial_attention(x)

# # FusionModule: 두 feature map을 concat 후 1x1 Conv, BN, ReLU, CBAM 적용
# class FusionModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.act = nn.ReLU(inplace=True)
#         self.cbam = CBAM(channels)
    
#     def forward(self, feat_local, feat_global):
#         x = torch.cat([feat_local, feat_global], dim=1)
#         x = self.conv1x1(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return self.cbam(x)

# ########################################
# # Cross-Attention Fusion Module
# ########################################
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, hidden_dim, num_tokens=4, num_heads=4):
#         """
#         hidden_dim: 기본 branch feature 차원 (예: 512)
#         num_tokens: feature vector를 분할할 토큰 수 (hidden_dim이 num_tokens로 나누어 떨어져야 함)
#         num_heads: MultiheadAttention의 head 수
#         """
#         super().__init__()
#         self.num_tokens = num_tokens
#         assert hidden_dim % num_tokens == 0, "hidden_dim must be divisible by num_tokens"
#         self.token_dim = hidden_dim // num_tokens
#         self.mha = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=num_heads)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
#     def forward(self, basic_feat, wavelet_feat):
#         # basic_feat, wavelet_feat: (B, hidden_dim)
#         B = basic_feat.size(0)
#         basic_tokens = basic_feat.view(B, self.num_tokens, self.token_dim)
#         wavelet_tokens = wavelet_feat.view(B, self.num_tokens, self.token_dim)
#         basic_tokens = basic_tokens.transpose(0, 1)  # (num_tokens, B, token_dim)
#         wavelet_tokens = wavelet_tokens.transpose(0, 1)
#         attn_output, _ = self.mha(basic_tokens, wavelet_tokens, wavelet_tokens)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(B, -1)
#         fused = self.out_proj(attn_output)
#         return fused

# ########################################
# # Self-Attention Fusion Module (using Adaptive Pooling)
# ########################################
# class SelfAttentionFusionAvg(nn.Module):
#     def __init__(self, in_channels_list: List[int], out_channels: int,
#                  patch_size: tuple = (7, 7), num_heads: int = 4,
#                  num_layers: int = 1, dropout: float = 0.1):
#         """
#         in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [256, 512, 1024, 2048])
#         out_channels: 모든 feature map을 투영할 공통 채널 수 (예: 256)
#         patch_size: 각 feature map을 Adaptive Pooling할 목표 공간 크기 (예: (7,7))
#         """
#         super().__init__()
#         self.patch_size = patch_size
#         # 각 feature map을 동일한 차원(out_channels)으로 투영하기 위한 1×1 conv 모듈 리스트
#         self.projs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         # batch_first=True를 추가하여 Transformer Encoder Layer 생성
#         encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads,
#                                                     dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         """
#         features: list of 4 feature maps, 각각 shape: [B, C_i, H_i, W_i]
#         반환: fused feature vector, shape [B, out_channels]
#         """
#         B = features[0].size(0)
#         tokens = []
#         for proj, feat in zip(self.projs, features):
#             # 각 feature map을 목표 크기(patch_size)로 통일
#             pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
#             proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
#             B, C, H, W = proj_feat.shape
#             # Flatten spatial dimensions → [B, H*W, out_channels]
#             token = proj_feat.view(B, C, H * W).transpose(1, 2)
#             tokens.append(token)
#         # Concatenate 토큰 시퀀스: [B, total_tokens, out_channels]
#         token_seq = torch.cat(tokens, dim=1)
#         # Transformer Encoder는 이제 [B, sequence_length, embed_dim] 형태를 기대합니다.
#         token_seq = self.transformer_encoder(token_seq)
#         # 평균 풀링하여 하나의 fused feature vector 생성: [B, out_channels]
#         fused_feature = token_seq.mean(dim=1)
#         return fused_feature


# ########################################
# # MANIQA_HF: MANIQA 모델의 고주파(HF) 버전
# ########################################
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, num_tokens=4, **kwargs):
#         super().__init__()
#         fusion_out_channels = 256  # 공통 fusion 차원
#         # 기본 branch: MaxViT backbone (features_only=True)
#         self.backbone = timm.create_model('maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
#                                           pretrained=True,
#                                           features_only=True,
#                                           out_indices=(1,2,3,4))
#         # hook을 통해 각 stage의 첫 번째 conv.drop_path 출력 저장
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
        
#         # 각 stage에서 CBAM을 적용한 후 Fusion Module
#         self.fusion_modules = nn.ModuleDict({
#             str(stage_idx): FusionModule(channels) for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })
#         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
#         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
#         # 대신 FPN 대신 Self-Attention Fusion 모듈 사용
#         self.self_attn_fusion = SelfAttentionFusionAvg(in_channels_list=self.feat_channels,
#                                                        out_channels=fusion_out_channels,
#                                                        patch_size=(7, 7),
#                                                        num_heads=4,
#                                                        num_layers=1,
#                                                        dropout=drop)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp_basic = nn.Sequential(
#             nn.Linear(fusion_out_channels, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop)
#         )
        
#         # Wavelet branch: 고주파(LH, HL, HH) 성분 추출 (grayscale 이미지 기준)
#         self.wavelet_linear = nn.Sequential(
#             nn.Linear(3 * (img_size // 2) * (img_size // 2), hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop)
#         )
#         self.wavelet_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        
#         # Cross-Attention Fusion: 기본 branch와 wavelet branch의 융합
#         self.cross_attn = CrossAttentionFusion(hidden_dim, num_tokens=num_tokens, num_heads=4)
        
#         # Fusion head: 최종 score 예측
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
#                 if len(parts) >= 5 and parts[1]=="blocks" and parts[2]=="0" and parts[3]=="conv" and parts[4]=="drop_path":
#                     try:
#                         stage_idx = int(parts[0].replace("stages_", ""))
#                     except Exception:
#                         continue
#                     module.register_forward_hook(
#                         lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
#                     )
    
#     def _first_mbconv_hook(self, stage_idx, module, inp, out):
#         if stage_idx not in self.first_mbconv_features:
#             self.first_mbconv_features[stage_idx] = out
    
#     def forward(self, x):
#         # Basic branch: obtain backbone features
#         self.first_mbconv_features = {}
#         backbone_feats = self.backbone(x)
#         final_stage_features = {}
#         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
#             stage_idx = int(info['module'].split('.')[1])
#             final_stage_features[stage_idx] = feat
        
#         fused_stage_features = []
#         for i, stage_idx in enumerate(self.stage_indices):
#             feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
#             feat_global = final_stage_features[stage_idx]
#             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
#             fused_feat = self.stage_norms[i](fused_feat)
#             fused_feat = self.stage_drops[i](fused_feat)
#             fused_stage_features.append(fused_feat)
        
#         # Self-Attention Fusion: 각 stage feature map을 adaptive pooling으로 7×7로 통일한 후 Transformer로 융합
#         basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
#         basic_feature = self.mlp_basic(basic_feature)  # [B, hidden_dim]
        
#         # Wavelet branch: RGB를 grayscale로 변환 후 haar wavelet 변환 적용하여 고주파 성분 추출
#         grayscale = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
#         B, _, H, W = grayscale.shape
#         hf_list = []
#         for i in range(B):
#             img_np = np.array(grayscale[i, 0].detach().cpu())
#             coeffs2 = pywt.dwt2(img_np, 'haar')
#             _, (LH, HL, HH) = coeffs2
#             hf_stack = np.stack([LH, HL, HH], axis=0)  # [3, H//2, W//2]
#             hf_tensor = torch.from_numpy(hf_stack).to(x.device, dtype=x.dtype)
#             hf_list.append(hf_tensor)
#         hf_tensor = torch.stack(hf_list, dim=0)  # [B, 3, H//2, W//2]
#         hf_flat = hf_tensor.view(B, -1)
#         wavelet_feature = self.wavelet_linear(hf_flat)  # [B, hidden_dim//2]
#         wavelet_feature = self.wavelet_proj(wavelet_feature)  # [B, hidden_dim]
        
#         # Cross-Attention Fusion: 기본 branch feature와 wavelet branch feature 융합
#         fused_feature = self.cross_attn(basic_feature, wavelet_feature)  # [B, hidden_dim]
        
#         # Fusion head: 최종 quality score 예측
#         score = self.fusion_head(fused_feature).squeeze(-1)
#         return torch.sigmoid(score)



import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List

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
class SelfAttentionFusionAvg(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int,
                 patch_size: tuple = (7, 7), num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        """
        in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [256, 512, 1024, 2048])
        out_channels: 모든 feature map을 투영할 공통 채널 수 (예: 256)
        patch_size: 각 feature map을 Adaptive Pooling할 목표 공간 크기 (예: (7,7))
        """
        super().__init__()
        self.patch_size = patch_size
        self.projs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: list of 4 feature maps, 각각 shape: [B, C_i, H_i, W_i]
        반환: fused feature vector, shape [B, out_channels]
        """
        B = features[0].size(0)
        tokens = []
        for proj, feat in zip(self.projs, features):
            pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
            proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
            _, C, H, W = proj_feat.shape
            token = proj_feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, out_channels]
            tokens.append(token)

        token_seq = torch.cat(tokens, dim=1)  # [B, total_tokens, out_channels]
        token_seq = self.transformer_encoder(token_seq)  # [B, total_tokens, out_channels]
        fused_feature = token_seq.mean(dim=1)  # [B, out_channels]
        return fused_feature

########################################
# MANIQA_HF (Wavelet 제거 버전)
########################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
        super().__init__()
        fusion_out_channels = 256  # 공통 fusion 차원
        
        # MaxViT backbone (features_only=True)
        self.backbone = timm.create_model(
            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
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
        
        # Self-Attention Fusion 모듈
        self.self_attn_fusion = SelfAttentionFusionAvg(
            in_channels_list=self.feat_channels,
            out_channels=fusion_out_channels,
            patch_size=(7, 7),
            num_heads=4,
            num_layers=1,
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
