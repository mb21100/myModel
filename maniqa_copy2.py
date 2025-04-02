# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from typing import List

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
# # MANIQA_HF (Wavelet 제거 버전)
# ########################################
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=768, **kwargs):
#         super().__init__()
#         fusion_out_channels = 256  # 공통 fusion 차원
        
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
#             num_layers=1,
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


# if __name__ == "__main__":
#     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
#     model.eval()
#     dummy_input = torch.randn(1, 3, 224, 224)
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
#     print("FLOPs: {:.3f} G".format(flops / 1e9))
#     print("Parameters: {:.3f} M".format(params / 1e6))
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("Output score shape:", output.shape)
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



# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=4, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         #self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         x = x * self.channel_attention(x)
#         #return x * self.spatial_attention(x)
#         return x

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
#                                                        num_layers=2,
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from thop import profile  # pip install thop
from typing import List, Optional

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

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # IQA 태스크에서는 spatial attention을 생략
        # self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        # return x * self.spatial_attention(x)
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

########################################
# Self-Attention Fusion Module (using Adaptive Pooling)
########################################
class SelfAttentionFusionAvg(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int,
                 patch_size: tuple = (7, 7), num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_size = patch_size
        self.projs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1)
                                     for in_ch in in_channels_list])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B = features[0].size(0)
        tokens = []
        for proj, feat in zip(self.projs, features):
            pooled = F.adaptive_avg_pool2d(feat, self.patch_size)
            proj_feat = proj(pooled)
            B, C, H, W = proj_feat.shape
            token = proj_feat.view(B, C, H * W).transpose(1, 2)
            tokens.append(token)
        token_seq = torch.cat(tokens, dim=1)
        token_seq = self.transformer_encoder(token_seq)
        fused_feature = token_seq.mean(dim=1)
        return fused_feature

########################################
# Haar Wavelet Transform (differentiable)
########################################
class HaarWaveletTransform(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        import math
        h = 1/math.sqrt(2)
        low_filter = torch.tensor([h, h], dtype=torch.float32)
        high_filter = torch.tensor([h, -h], dtype=torch.float32)
        LL = torch.ger(low_filter, low_filter)  # [2,2]
        LH = torch.ger(low_filter, high_filter)
        HL = torch.ger(high_filter, low_filter)
        HH = torch.ger(high_filter, high_filter)
        kernel = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        # x: [B, 1, H, W]
        return F.conv2d(x, self.kernel, stride=2)

########################################
# 새로운 Wavelet Branch (Differentiable)
########################################
class WaveletBranch(nn.Module):
    def __init__(self, img_size=224, hidden_dim=512, drop=0.1):
        super().__init__()
        # Grayscale conversion (fixed weights)
        self.register_buffer('gray_weight', torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1,3,1,1))
        self.haar = HaarWaveletTransform(in_channels=1)  # transforms grayscale image
        # After Haar, output: [B, 4, H/2, W/2]; we'll use high-frequency components (LH, HL, HH)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(self, x):
        # x: [B, 3, H, W]
        gray = (x * self.gray_weight).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        wavelet = self.haar(gray)  # [B, 4, H/2, W/2]
        # Select high-frequency components: LH, HL, HH (channels 1,2,3)
        high_freq = wavelet[:, 1:, :, :]  # [B, 3, H/2, W/2]
        features = self.cnn(high_freq)  # [B, 16, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 16]
        out = self.fc(features)  # [B, hidden_dim]
        return out

########################################
# MANIQA_HF: MANIQA 모델의 고주파(HF) 버전 (Wavelet Branch 개선)
########################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, num_tokens=4, **kwargs):
        super().__init__()
        fusion_out_channels = 256  # 공통 fusion 차원

        # Basic branch: MaxViT backbone
        self.backbone = timm.create_model('maxvit_rmlp_small_rw_224.sw_in1k',
                                          pretrained=True,
                                          features_only=True,
                                          out_indices=(1,2,3,4))
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
        
        self.self_attn_fusion = SelfAttentionFusionAvg(in_channels_list=self.feat_channels,
                                                       out_channels=fusion_out_channels,
                                                       patch_size=(7,7),
                                                       num_heads=4,
                                                       num_layers=2,
                                                       dropout=drop)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mlp_basic = nn.Sequential(
            nn.Linear(fusion_out_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        # 개선된 Wavelet branch
        self.wavelet_branch = WaveletBranch(img_size=img_size, hidden_dim=hidden_dim, drop=drop)
        
        # Fusion head: 두 branch feature를 concatenate하여 사용
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_outputs)
        )
    
    def _register_first_mbconv_hooks(self):
        for name, module in self.backbone.named_modules():
            if name.startswith("stages_"):
                parts = name.split('.')
                if (len(parts) >= 5 and parts[1]=="blocks" and parts[2]=="0" and 
                    parts[3]=="conv" and parts[4]=="drop_path"):
                    try:
                        stage_idx = int(parts[0].replace("stages_",""))
                    except Exception:
                        continue
                    module.register_forward_hook(
                        lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
                    )
    
    def _first_mbconv_hook(self, stage_idx, module, inp, out):
        if stage_idx not in self.first_mbconv_features:
            self.first_mbconv_features[stage_idx] = out
    
    def forward(self, x, debug=False):
        # Basic branch: backbone feature extraction
        self.first_mbconv_features = {}
        backbone_feats = self.backbone(x)
        final_stage_features = {}
        for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
            stage_idx = int(info['module'].split('.')[1])
            final_stage_features[stage_idx] = feat
            if debug:
                print(f"Backbone Stage {stage_idx} feature shape: {feat.shape}")
        
        fused_stage_features = []
        for i, stage_idx in enumerate(self.stage_indices):
            feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
            feat_global = final_stage_features[stage_idx]
            fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
            fused_feat = self.stage_norms[i](fused_feat)
            fused_feat = self.stage_drops[i](fused_feat)
            fused_stage_features.append(fused_feat)
            if debug:
                print(f"Fused feature from stage {stage_idx} shape: {fused_feat.shape}")
        
        basic_feature = self.self_attn_fusion(fused_stage_features)
        if debug:
            print(f"Basic feature shape after Self-Attention Fusion: {basic_feature.shape}")
        basic_feature = self.mlp_basic(basic_feature)
        if debug:
            print(f"Feature shape after MLP Basic: {basic_feature.shape}")
        
        # Improved Wavelet branch: use original input x
        wavelet_feature = self.wavelet_branch(x)
        if debug:
            print(f"Wavelet branch feature shape: {wavelet_feature.shape}")
        
        fused_feature = torch.cat([basic_feature, wavelet_feature], dim=1)
        score = self.fusion_head(fused_feature).squeeze(-1)
        if debug:
            print(f"Final output score shape: {score.shape}")
        return torch.sigmoid(score)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print("FLOPs: {:.3f} G".format(flops / 1e9))
    print("Parameters: {:.3f} M".format(params / 1e6))
    with torch.no_grad():
        output = model(dummy_input, debug=True)
    print("Output score:", output)
