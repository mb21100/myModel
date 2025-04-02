# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import timm
# # from typing import List

# # ########################################
# # # CBAM 관련 모듈
# # ########################################
# # class ChannelAttention(nn.Module):
# #     def __init__(self, in_channels, reduction=4):
# #         super().__init__()
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.max_pool = nn.AdaptiveMaxPool2d(1)
# #         self.fc = nn.Sequential(
# #             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
# #         )
# #         self.sigmoid = nn.Sigmoid()
    
# #     def forward(self, x):
# #         avg_out = self.fc(self.avg_pool(x))
# #         max_out = self.fc(self.max_pool(x))
# #         return self.sigmoid(avg_out + max_out)

# # class SpatialAttention(nn.Module):
# #     def __init__(self, kernel_size=7):
# #         super().__init__()
# #         padding = (kernel_size - 1) // 2
# #         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
# #         self.sigmoid = nn.Sigmoid()
    
# #     def forward(self, x):
# #         avg_out = torch.mean(x, dim=1, keepdim=True)
# #         max_out, _ = torch.max(x, dim=1, keepdim=True)
# #         x_cat = torch.cat([avg_out, max_out], dim=1)
# #         return self.sigmoid(self.conv(x_cat))

# # class CBAM(nn.Module):
# #     def __init__(self, in_channels, reduction=4, kernel_size=7):
# #         super().__init__()
# #         self.channel_attention = ChannelAttention(in_channels, reduction)
# #         self.spatial_attention = SpatialAttention(kernel_size)
    
# #     def forward(self, x):
# #         x = x * self.channel_attention(x)
# #         return x * self.spatial_attention(x)

# # class FusionModule(nn.Module):
# #     def __init__(self, channels):
# #         super().__init__()
# #         self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
# #         self.bn = nn.BatchNorm2d(channels)
# #         self.act = nn.ReLU(inplace=True)
# #         self.cbam = CBAM(channels)
    
# #     def forward(self, feat_local, feat_global):
# #         x = torch.cat([feat_local, feat_global], dim=1)
# #         x = self.conv1x1(x)
# #         x = self.bn(x)
# #         x = self.act(x)
# #         return self.cbam(x)

# # ########################################
# # # Self-Attention Fusion Module (using Adaptive Pooling)
# # ########################################
# # class SelfAttentionFusionAvg(nn.Module):
# #     def __init__(self, in_channels_list: List[int], out_channels: int,
# #                  patch_size: tuple = (7, 7), num_heads: int = 4,
# #                  num_layers: int = 2, dropout: float = 0.2):
# #         """
# #         in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [256, 512, 1024, 2048])
# #         out_channels: 모든 feature map을 투영할 공통 채널 수 (예: 256)
# #         patch_size: 각 feature map을 Adaptive Pooling할 목표 공간 크기 (예: (7,7))
# #         """
# #         super().__init__()
# #         self.patch_size = patch_size
# #         self.projs = nn.ModuleList([
# #             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
# #         ])
# #         encoder_layer = nn.TransformerEncoderLayer(
# #             d_model=out_channels,
# #             nhead=num_heads,
# #             dropout=dropout,
# #             batch_first=True
# #         )
# #         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
# #     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
# #         """
# #         features: list of 4 feature maps, 각각 shape: [B, C_i, H_i, W_i]
# #         반환: fused feature vector, shape [B, out_channels]
# #         """
# #         B = features[0].size(0)
# #         tokens = []
# #         for proj, feat in zip(self.projs, features):
# #             pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
# #             proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
# #             _, C, H, W = proj_feat.shape
# #             token = proj_feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, out_channels]
# #             tokens.append(token)

# #         token_seq = torch.cat(tokens, dim=1)  # [B, total_tokens, out_channels]
# #         token_seq = self.transformer_encoder(token_seq)  # [B, total_tokens, out_channels]
# #         fused_feature = token_seq.mean(dim=1)  # [B, out_channels]
# #         return fused_feature

# # ########################################
# # # MANIQA_HF (Wavelet 제거 버전)
# # ########################################
# # class MANIQA(nn.Module):
# #     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
# #         super().__init__()
# #         fusion_out_channels = 256  # 공통 fusion 차원
        
# #         # MaxViT backbone (features_only=True)
# #         self.backbone = timm.create_model(
# #             'maxvit_rmlp_small_rw_224.sw_in1k',
# #             pretrained=True,
# #             features_only=True,
# #             out_indices=(1,2,3,4)
# #         )
# #         self.first_mbconv_features = {}
# #         self._register_first_mbconv_hooks()
        
# #         # 각 stage 정보 추출
# #         feat_infos = self.backbone.feature_info[1:5]
# #         self.stage_indices = []
# #         self.feat_channels = []
# #         for info in feat_infos:
# #             stage_idx = int(info['module'].split('.')[1])
# #             self.stage_indices.append(stage_idx)
# #             self.feat_channels.append(info['num_chs'])
# #         self.num_stages = len(self.feat_channels)
        
# #         # CBAM 기반 Fusion
# #         self.fusion_modules = nn.ModuleDict({
# #             str(stage_idx): FusionModule(channels)
# #             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
# #         })
# #         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
# #         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
# #         # Self-Attention Fusion 모듈
# #         self.self_attn_fusion = SelfAttentionFusionAvg(
# #             in_channels_list=self.feat_channels,
# #             out_channels=fusion_out_channels,
# #             patch_size=(7, 7),
# #             num_heads=4,
# #             num_layers=1,
# #             dropout=drop
# #         )
        
# #         self.mlp_basic = nn.Sequential(
# #             nn.Linear(fusion_out_channels, hidden_dim),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(drop)
# #         )
        
# #         # 최종 head (Wavelet 없이 basic_feature만 사용)
# #         self.fusion_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(drop),
# #             nn.Linear(hidden_dim, num_outputs)
# #         )
    
# #     def _register_first_mbconv_hooks(self):
# #         for name, module in self.backbone.named_modules():
# #             if name.startswith("stages_"):
# #                 parts = name.split('.')
# #                 # "stages_<stage_idx>.blocks.0.conv.drop_path" 형태 찾기
# #                 if (len(parts) >= 5 and
# #                     parts[1] == "blocks" and
# #                     parts[2] == "0" and
# #                     parts[3] == "conv" and
# #                     parts[4] == "drop_path"):
# #                     try:
# #                         stage_idx = int(parts[0].replace("stages_", ""))
# #                     except Exception:
# #                         continue
# #                     module.register_forward_hook(
# #                         lambda m, inp, out, stage_idx=stage_idx:
# #                             self._first_mbconv_hook(stage_idx, m, inp, out)
# #                     )
    
# #     def _first_mbconv_hook(self, stage_idx, module, inp, out):
# #         if stage_idx not in self.first_mbconv_features:
# #             self.first_mbconv_features[stage_idx] = out
    
# #     def forward(self, x):
# #         # Backbone에서 feature 추출
# #         self.first_mbconv_features = {}
# #         backbone_feats = self.backbone(x)
# #         final_stage_features = {}
# #         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
# #             stage_idx = int(info['module'].split('.')[1])
# #             final_stage_features[stage_idx] = feat
        
# #         # CBAM Fusion
# #         fused_stage_features = []
# #         for i, stage_idx in enumerate(self.stage_indices):
# #             feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
# #             feat_global = final_stage_features[stage_idx]
# #             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
# #             fused_feat = self.stage_norms[i](fused_feat)
# #             fused_feat = self.stage_drops[i](fused_feat)
# #             fused_stage_features.append(fused_feat)
        
# #         # Self-Attention Fusion으로 여러 스테이지 특징 융합
# #         basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
# #         basic_feature = self.mlp_basic(basic_feature)                # [B, hidden_dim]
        
# #         # 최종 score 예측
# #         score = self.fusion_head(basic_feature).squeeze(-1)
# #         return torch.sigmoid(score)


# # if __name__ == "__main__":
# #     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
# #     model.eval()
# #     dummy_input = torch.randn(1, 3, 224, 224)
# #     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
# #     print("FLOPs: {:.3f} G".format(flops / 1e9))
# #     print("Parameters: {:.3f} M".format(params / 1e6))
# #     with torch.no_grad():
# #         output = model(dummy_input)
# #     print("Output score shape:", output.shape)
# #     if model.first_mbconv_features:
# #         print("Extracted features from conv.drop_path of each stage:")
# #         for stage_idx, feat in model.first_mbconv_features.items():
# #             print(f"  Stage {stage_idx}: feature shape = {feat.shape}")
# #     else:
# #         print("No features were extracted by hooks.")
# #     backbone_feats = model.backbone(dummy_input)
# #     print("Backbone final feature maps:")
# #     for idx, feat in enumerate(backbone_feats, 1):
# #         print(f"  Stage {idx}: feature shape = {feat.shape}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# import pywt
# from thop import profile  # pip install thop
# import numpy as np
# from typing import List, Optional

# ########################################
# # CBAM 관련 모듈
# ########################################
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=4):
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

# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=4, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         x = x * self.channel_attention(x)
#         return x * self.spatial_attention(x)

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
# # Self-Attention Fusion Module (using Adaptive Pooling)
# ########################################
# class SelfAttentionFusionAvg(nn.Module):
#     def __init__(self, in_channels_list: List[int], out_channels: int,
#                  patch_size: tuple = (7, 7), num_heads: int = 1,
#                  num_layers: int = 1, dropout: float = 0.2):
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
#             pooled = F.adaptive_avg_pool2d(feat, self.patch_size)  # [B, C_i, patch_H, patch_W]
#             proj_feat = proj(pooled)  # [B, out_channels, patch_H, patch_W]
#             B, C, H, W = proj_feat.shape
#             token = proj_feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, out_channels]
#             tokens.append(token)
#         token_seq = torch.cat(tokens, dim=1)  # [B, total_tokens, out_channels]
#         token_seq = self.transformer_encoder(token_seq)  # [B, total_tokens, out_channels]
#         fused_feature = token_seq.mean(dim=1)  # [B, out_channels]
#         return fused_feature

# ########################################
# # MANIQA_HF: MANIQA 모델의 고주파(HF) 버전
# ########################################
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, num_tokens=4, **kwargs):
#         super().__init__()
#         fusion_out_channels = 256  # 공통 fusion 차원
#         # 기본 branch: MaxViT backbone (features_only=True)
#         self.backbone = timm.create_model('maxvit_rmlp_small_rw_224.sw_in1k',
#                                           pretrained=True,
#                                           features_only=True,
#                                           out_indices=(1,2,3,4))
#         self.first_mbconv_features = {}
#         self._register_first_mbconv_hooks()
        
#         feat_infos = self.backbone.feature_info[1:5]
#         self.stage_indices = []
#         self.feat_channels = []
#         for info in feat_infos:
#             stage_idx = int(info['module'].split('.')[1])
#             self.stage_indices.append(stage_idx)
#             self.feat_channels.append(info['num_chs'])
#         self.num_stages = len(self.feat_channels)
        
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
        
#         # 여기서는 Cross-Attention Fusion 대신 두 branch feature를 concat하여 fusion head에 넣습니다.
#         # Fusion head의 입력 차원이 변경되어야 하므로 hidden_dim*2를 입력으로 합니다.
#         self.fusion_head = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
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
        
#         # 대신 Cross-Attention Fusion 대신, concatenate 두 branch의 feature
#         fused_feature = torch.cat([basic_feature, wavelet_feature], dim=1)  # [B, hidden_dim*2]
        
#         # Fusion head: 최종 quality score 예측
#         score = self.fusion_head(fused_feature).squeeze(-1)
#         return torch.sigmoid(score)

# if __name__ == "__main__":
#     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
#     model.eval()
#     dummy_input = torch.randn(1, 3, 224, 224)
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
#     print("FLOPs: {:.3f} G".format(flops / 1e9))
#     print("Parameters: {:.3f} M".format(params / 1e6))
#     with torch.no_grad():
#         output = model(dummy_input)
#         print("Output score shape:", output.shape)
#     if model.first_mbconv_features:
#         print("Extracted features from conv.drop_path of each stage:")
#         for stage_idx, feat in model.first_mbconv_features.items():
#             print(f"  Stage {stage_idx}: feature shape = {feat.shape}")
#     else:
#         print("No features were extracted by hooks.")
#     backbone_feats = model.backbone(dummy_input)
#     print("Backbone final feature maps:")
#     for idx, feat in enumerate(backbone_feats, 1):
#         print(f"  Stage {idx}: feature shape = {feat.shape}")


#  FLOPs: 18.182 G  Parameters: 78.515 M
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import timm
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
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
# 이전의 Relative Self-Attention Fusion 모듈은 제거됨
########################################

########################################
# Improved MLP (새로운 MLP 헤드)
########################################
class ImprovedMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, drop):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        residual = x
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x = x + residual  # residual connection
        return torch.sigmoid(self.fc_out(x))

########################################
# Swin Transformer 관련 모듈 (기존 코드 사용)
########################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim_mlp, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution,
                      num_heads=num_heads, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      dim_mlp=dim_mlp,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, patches_resolution, depths, num_heads, embed_dim, window_size=7,
                 drop=0.0, drop_rate=0.0, drop_path_rate=0.1, dim_mlp=1024, qkv_bias=True,
                 qk_scale=None, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.patches_resolution = patches_resolution  # (H, W)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.dropout(x)
        # x: [B, C, H, W] -> flatten to sequence for each layer block processing
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        for layer in self.layers:
            _x = x
            x = layer(x)
            x = _x + x  # residual connection between layers
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

########################################
# MANIQA 모델 (Wavelet branch 제외, 기본 branch만 사용)
########################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=288, drop=0.1, hidden_dim=768, **kwargs):
        super().__init__()
        fusion_out_channels = 768  # 공통 fusion 차원
        
        # MaxViT backbone (features_only=True)
        self.backbone = timm.create_model(
            'maxvit_rmlp_small_rw_224.sw_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1,2,3,4),
            img_size=img_size 
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
        
        # ----- Multi-Scale Fusion 후 Swin Block 적용 -----
        # 각 stage의 fused feature map을 (7,7)로 adaptive pooling 후 1x1 conv로 투영
        self.swin_proj = nn.ModuleList([
            nn.Conv2d(ch, fusion_out_channels, kernel_size=1)
            for ch in self.feat_channels
        ])
        # 추가: 여러 스테이지의 투영 feature map을 채널-wise concat 후 fusion을 위한 1x1 conv
        self.fusion_conv = nn.Conv2d(fusion_out_channels * self.num_stages, fusion_out_channels, kernel_size=1)
        
        # Swin Transformer 블록: 입력 해상도 (7,7)
        self.swin_block = SwinTransformer(
            patches_resolution=(7,7),
            depths=[2],
            num_heads=[4],
            embed_dim=fusion_out_channels,
            window_size=7,
            drop=drop,
            drop_path_rate=0.1,
            dim_mlp=1024,
            qkv_bias=True,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False
        )
        # -----------------------------------------------------
        
        # 기존 mlp_basic, fusion_head 대신 ImprovedMLP 사용
        self.improved_mlp = ImprovedMLP(in_features=fusion_out_channels, hidden_dim=hidden_dim, drop=drop)
    
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
        
        # 각 스테이지 feature를 (7,7)로 adaptive pooling 후 1x1 conv로 투영
        proj_features = []
        for i, feat in enumerate(fused_stage_features):
            pooled = F.adaptive_avg_pool2d(feat, (7,7))
            proj_feat = self.swin_proj[i](pooled)  # [B, fusion_out_channels, 7,7]
            proj_features.append(proj_feat)
        # 채널-wise concatenate: [B, fusion_out_channels * num_stages, 7,7]
        concat_features = torch.cat(proj_features, dim=1)
        # 1x1 conv를 통해 차원 축소: [B, fusion_out_channels, 7,7]
        fused_map = self.fusion_conv(concat_features)
        
        # Swin Transformer 블록 적용
        swin_out = self.swin_block(fused_map)  # [B, fusion_out_channels, 7,7]
        # Global average pooling 후 벡터화
        basic_feature = F.adaptive_avg_pool2d(swin_out, (1,1)).view(swin_out.size(0), -1)
        score = self.improved_mlp(basic_feature).squeeze(-1)
        return score


if __name__ == "__main__":
    model = MANIQA(num_outputs=1, img_size=288, drop=0.1, hidden_dim=768)
    model.eval()
    dummy_input = torch.randn(1, 3, 288, 288)
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
