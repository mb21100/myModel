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
# #         #self.spatial_attention = SpatialAttention(kernel_size)
    
# #     def forward(self, x):
# #         x = x * self.channel_attention(x)
# #         #return x * self.spatial_attention(x)
# #         return x 

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

# # #######################################
# # #MANIQA_HF (Wavelet 제거 버전)
# # #######################################
# # class MANIQA(nn.Module):
# #     def __init__(self, num_outputs=1, img_size=224, drop=0.3, hidden_dim=768, **kwargs):
# #         super().__init__()
# #         fusion_out_channels = 768  # 공통 fusion 차원

# #         # MaxViT backbone (features_only=True)
# #         self.backbone = timm.create_model(
# #             'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
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
# #             num_layers=2,
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
#         # self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         # 채널 어텐션만 사용 (spatial_attention 주석 처리)
#         x = x * self.channel_attention(x)
#         return x

# class FusionModule(nn.Module):
#     """
#     두 feature map(feat_local, feat_global)을 concat 후,
#     1x1 conv + BN + ReLU + CBAM 을 적용하여 융합
#     """
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.act = nn.ReLU(inplace=True)
#         self.cbam = CBAM(channels)
    
#     def forward(self, feat_local, feat_global):
#         x = torch.cat([feat_local, feat_global], dim=1)  # [B, 2C, H, W]
#         x = self.conv1x1(x)                              # [B, C, H, W]
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.cbam(x)
#         return x

# ########################################
# # Self-Attention Fusion Module (using Adaptive Pooling)
# ########################################
# class SelfAttentionFusionAvg(nn.Module):
#     """
#     여러 stage의 feature map을 각각 adaptive avg pooling으로 동일 크기(patch_size)로 만든 뒤,
#     out_channels로 투영(1x1 conv)하고 TransformerEncoder로 융합하는 모듈
#     """
#     def __init__(self, in_channels_list: List[int], out_channels: int,
#                  patch_size: tuple = (7, 7), num_heads: int = 4,
#                  num_layers: int = 2, dropout: float = 0.2):
#         super().__init__()
#         self.patch_size = patch_size
#         self.projs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1)
#             for in_ch in in_channels_list
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
#         features: [stage1, stage2, stage3, stage4]의 feature map 리스트
#         각각 shape: [B, C_i, H_i, W_i]
#         """
#         B = features[0].size(0)
#         tokens = []
#         for proj, feat in zip(self.projs, features):
#             # 각 stage feature map을 adaptive avg pooling
#             pooled = F.adaptive_avg_pool2d(feat, self.patch_size)   # [B, C_i, patch_H, patch_W]
#             proj_feat = proj(pooled)                                # [B, out_channels, patch_H, patch_W]
#             _, C, H, W = proj_feat.shape
#             token = proj_feat.view(B, C, H * W).transpose(1, 2)     # [B, H*W, out_channels]
#             tokens.append(token)

#         # 모든 stage token을 concat => [B, total_tokens, out_channels]
#         token_seq = torch.cat(tokens, dim=1)
#         # TransformerEncoder로 융합
#         token_seq = self.transformer_encoder(token_seq)             # [B, total_tokens, out_channels]
#         # 평균 풀링 => 하나의 vector (fused_feature)
#         fused_feature = token_seq.mean(dim=1)                       # [B, out_channels]
#         return fused_feature

# #######################################
# # MANIQA 모델 (Wavelet 제거 버전 + 마지막 stage feature concat)
# #######################################
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.3, hidden_dim=768, **kwargs):
#         super().__init__()
#         # Self-Attention Fusion 모듈의 출력 차원
#         self.fusion_out_channels = 512  
        
#         # MaxViT backbone (features_only=True)
#         self.backbone = timm.create_model(
#             'maxvit_rmlp_small_rw_224.sw_in1k',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1,2,3,4)
#         )
#         self.first_mbconv_features = {}
#         self._register_first_mbconv_hooks()
        
#         # 각 stage 채널 정보 추출
#         feat_infos = self.backbone.feature_info[1:5]  # (stage1 ~ stage4)
#         self.stage_indices = []
#         self.feat_channels = []
#         for info in feat_infos:
#             stage_idx = int(info['module'].split('.')[1])
#             self.stage_indices.append(stage_idx)
#             self.feat_channels.append(info['num_chs'])
#         self.num_stages = len(self.feat_channels)  # 일반적으로 4
        
#         # CBAM 기반 FusionModule
#         self.fusion_modules = nn.ModuleDict({
#             str(stage_idx): FusionModule(channels)
#             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })
#         self.stage_norms = nn.ModuleList([
#             nn.BatchNorm2d(c) for c in self.feat_channels
#         ])
#         self.stage_drops = nn.ModuleList([
#             nn.Dropout2d(drop) for _ in range(self.num_stages)
#         ])
        
#         # Self-Attention Fusion 모듈
#         self.self_attn_fusion = SelfAttentionFusionAvg(
#             in_channels_list=self.feat_channels,
#             out_channels=self.fusion_out_channels,
#             patch_size=(7, 7),
#             num_heads=4,
#             num_layers=2,
#             dropout=drop
#         )
        
#         # 융합된 feature => MLP (기본 branch)
#         self.mlp_basic = nn.Sequential(
#             nn.Linear(self.fusion_out_channels, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop)
#         )
        
#         # [추가] 백본 마지막 stage feature(글로벌)도 사용 => global pool 후 concat
#         # 마지막 stage의 채널 수
#         self.last_stage_channels = self.feat_channels[-1]  # ex) 768
        
#         # concat 후 => (hidden_dim + last_stage_channels) 차원을 입력으로 받아서 최종 score 예측
#         final_in_dim = hidden_dim + self.last_stage_channels
        
#         self.fusion_head = nn.Sequential(
#             nn.Linear(final_in_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, num_outputs)
#         )
    
#     def _register_first_mbconv_hooks(self):
#         """
#         stages_<stage_idx>.blocks.0.conv.drop_path 모듈에 hook을 걸어 
#         해당 stage의 첫 번째 MBConv 결과를 저장
#         """
#         for name, module in self.backbone.named_modules():
#             if name.startswith("stages_"):
#                 parts = name.split('.')
#                 if (len(parts) >= 5 and
#                     parts[1] == "blocks" and
#                     parts[2] == "0" and
#                     parts[3] == "conv" and
#                     parts[4] == "drop_path"):
#                     try:
#                         stage_idx = int(parts[0].replace("stages_",""))
#                     except Exception:
#                         continue
#                     module.register_forward_hook(
#                         lambda m, inp, out, stage_idx=stage_idx:
#                             self._first_mbconv_hook(stage_idx, m, inp, out)
#                     )
    
#     def _first_mbconv_hook(self, stage_idx, module, inp, out):
#         # 해당 stage의 첫 MBConv 출력(feature)을 저장
#         if stage_idx not in self.first_mbconv_features:
#             self.first_mbconv_features[stage_idx] = out
    
#     def forward(self, x):
#         B = x.size(0)
        
#         # 1) Backbone 추출
#         self.first_mbconv_features = {}
#         backbone_feats = self.backbone(x)  
#         # out_indices=(1,2,3,4)에 해당하는 4개 stage의 최종 feature
#         # final_stage_features: {stage_idx: feature}
#         final_stage_features = {}
#         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
#             stage_idx = int(info['module'].split('.')[1])
#             final_stage_features[stage_idx] = feat
        
#         # 2) CBAM Fusion: (stage별 첫 MBConv 결과 vs 마지막 결과) 융합
#         fused_stage_features = []
#         for i, stage_idx in enumerate(self.stage_indices):
#             feat_local = self.first_mbconv_features.get(
#                 stage_idx, final_stage_features[stage_idx]
#             )
#             feat_global = final_stage_features[stage_idx]
#             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
#             fused_feat = self.stage_norms[i](fused_feat)
#             fused_feat = self.stage_drops[i](fused_feat)
#             fused_stage_features.append(fused_feat)
        
#         # 3) Self-Attention Fusion: 여러 stage의 feature map을 한 덩어리로 융합
#         basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
        
#         # 4) 기본 branch MLP
#         basic_feature = self.mlp_basic(basic_feature)                # [B, hidden_dim]
        
#         # 5) 백본 마지막 stage feature(가장 글로벌 정보) 추가 활용
#         #    ex) stage_idx가 가장 큰 값 (보통 self.stage_indices[-1])의 feature
#         #    global pooling => shape [B, last_stage_channels]
#         last_stage_idx = self.stage_indices[-1]
#         last_stage_feat = final_stage_features[last_stage_idx]       # [B, C, H, W]
#         pooled_global = F.adaptive_avg_pool2d(last_stage_feat, (1, 1))  # [B, C, 1, 1]
#         pooled_global = pooled_global.view(B, -1)                    # [B, C] => ex) [B, 768]
        
#         # 6) concat => shape [B, hidden_dim + last_stage_channels]
#         cat_feature = torch.cat([basic_feature, pooled_global], dim=1)
        
#         # 7) 최종 head
#         score = self.fusion_head(cat_feature).squeeze(-1)  # [B] (or [B, 1] if squeeze(-1) 안 하면)
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
        # self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 채널 어텐션만 사용 (spatial_attention 주석 처리)
        x = x * self.channel_attention(x)
        return x

class FusionModule(nn.Module):
    """
    두 feature map(feat_local, feat_global)을 concat 후,
    1x1 conv + BN + ReLU + CBAM 을 적용하여 융합
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.cbam = CBAM(channels)
    
    def forward(self, feat_local, feat_global):
        x = torch.cat([feat_local, feat_global], dim=1)  # [B, 2C, H, W]
        x = self.conv1x1(x)                              # [B, C, H, W]
        x = self.bn(x)
        x = self.act(x)
        x = self.cbam(x)
        return x

########################################
# Self-Attention Fusion Module (using Adaptive Pooling)
########################################
class SelfAttentionFusionAvg(nn.Module):
    """
    여러 stage의 feature map을 각각 adaptive avg pooling으로 동일 크기(patch_size)로 만든 뒤,
    out_channels로 투영(1x1 conv)하고 TransformerEncoder로 융합하는 모듈
    """
    def __init__(self, in_channels_list: List[int], out_channels: int,
                 patch_size: tuple = (7, 7), num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_size = patch_size
        self.projs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
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
        features: [stage1, stage2, stage3, stage4]의 feature map 리스트
        각각 shape: [B, C_i, H_i, W_i]
        """
        B = features[0].size(0)
        tokens = []
        for proj, feat in zip(self.projs, features):
            # 각 stage feature map을 adaptive avg pooling
            pooled = F.adaptive_avg_pool2d(feat, self.patch_size)   # [B, C_i, patch_H, patch_W]
            proj_feat = proj(pooled)                                # [B, out_channels, patch_H, patch_W]
            _, C, H, W = proj_feat.shape
            token = proj_feat.view(B, C, H * W).transpose(1, 2)     # [B, H*W, out_channels]
            tokens.append(token)

        # 모든 stage token을 concat => [B, total_tokens, out_channels]
        token_seq = torch.cat(tokens, dim=1)
        # TransformerEncoder로 융합
        token_seq = self.transformer_encoder(token_seq)             # [B, total_tokens, out_channels]
        # 평균 풀링 => 하나의 vector (fused_feature)
        fused_feature = token_seq.mean(dim=1)                       # [B, out_channels]
        return fused_feature

#######################################
# MANIQA 모델 (마지막 stage feature concat)
#######################################
class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.3, hidden_dim=768, **kwargs):
        super().__init__()
        # Self-Attention Fusion 모듈의 출력 차원
        self.fusion_out_channels = 384  
        
        # MaxViT backbone (features_only=True)
        self.backbone = timm.create_model(
            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1,2,3,4)
        )
        self.first_mbconv_features = {}
        self._register_first_mbconv_hooks()
        
        # 각 stage 채널 정보 추출
        feat_infos = self.backbone.feature_info[1:5]  # (stage1 ~ stage4)
        self.stage_indices = []
        self.feat_channels = []
        for info in feat_infos:
            stage_idx = int(info['module'].split('.')[1])
            self.stage_indices.append(stage_idx)
            self.feat_channels.append(info['num_chs'])
        self.num_stages = len(self.feat_channels)  # 일반적으로 4
        
        # CBAM 기반 FusionModule
        self.fusion_modules = nn.ModuleDict({
            str(stage_idx): FusionModule(channels)
            for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
        })
        self.stage_norms = nn.ModuleList([
            nn.BatchNorm2d(c) for c in self.feat_channels
        ])
        self.stage_drops = nn.ModuleList([
            nn.Dropout2d(drop) for _ in range(self.num_stages)
        ])
        
        # Self-Attention Fusion 모듈
        self.self_attn_fusion = SelfAttentionFusionAvg(
            in_channels_list=self.feat_channels,
            out_channels=self.fusion_out_channels,
            patch_size=(7, 7),
            num_heads=4,
            num_layers=2,
            dropout=drop
        )
        
        # 융합된 feature => MLP (기본 branch)
        # BatchNorm1d 대신 LayerNorm 사용
        self.mlp_basic = nn.Sequential(
            nn.Linear(self.fusion_out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        # [추가] 백본 마지막 stage feature(글로벌)도 사용 => global pool 후 concat
        # 마지막 stage의 채널 수
        self.last_stage_channels = self.feat_channels[-1]  # ex) 768
        
        # concat 후 => (hidden_dim + last_stage_channels) 차원을 입력으로 받아서 최종 score 예측
        final_in_dim = hidden_dim + self.last_stage_channels
        
        self.fusion_head = nn.Sequential(
            nn.Linear(final_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_outputs)
        )
    
    def _register_first_mbconv_hooks(self):
        """
        stages_<stage_idx>.blocks.0.conv.drop_path 모듈에 hook을 걸어 
        해당 stage의 첫 번째 MBConv 결과를 저장
        """
        for name, module in self.backbone.named_modules():
            if name.startswith("stages_"):
                parts = name.split('.')
                if (len(parts) >= 5 and
                    parts[1] == "blocks" and
                    parts[2] == "0" and
                    parts[3] == "conv" and
                    parts[4] == "drop_path"):
                    try:
                        stage_idx = int(parts[0].replace("stages_",""))
                    except Exception:
                        continue
                    module.register_forward_hook(
                        lambda m, inp, out, stage_idx=stage_idx:
                            self._first_mbconv_hook(stage_idx, m, inp, out)
                    )
    
    def _first_mbconv_hook(self, stage_idx, module, inp, out):
        # 해당 stage의 첫 MBConv 출력(feature)을 저장
        if stage_idx not in self.first_mbconv_features:
            self.first_mbconv_features[stage_idx] = out
    
    def forward(self, x):
        B = x.size(0)
        
        # 1) Backbone 추출
        self.first_mbconv_features = {}
        backbone_feats = self.backbone(x)  
        # out_indices=(1,2,3,4)에 해당하는 4개 stage의 최종 feature
        # final_stage_features: {stage_idx: feature}
        final_stage_features = {}
        for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
            stage_idx = int(info['module'].split('.')[1])
            final_stage_features[stage_idx] = feat
        
        # 2) CBAM Fusion: (stage별 첫 MBConv 결과 vs 마지막 결과) 융합
        fused_stage_features = []
        for i, stage_idx in enumerate(self.stage_indices):
            feat_local = self.first_mbconv_features.get(
                stage_idx, final_stage_features[stage_idx]
            )
            feat_global = final_stage_features[stage_idx]
            fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
            fused_feat = self.stage_norms[i](fused_feat)
            fused_feat = self.stage_drops[i](fused_feat)
            fused_stage_features.append(fused_feat)
        
        # 3) Self-Attention Fusion: 여러 stage의 feature map을 한 덩어리로 융합
        basic_feature = self.self_attn_fusion(fused_stage_features)  # [B, fusion_out_channels]
        
        # 4) 기본 branch MLP
        basic_feature = self.mlp_basic(basic_feature)                # [B, hidden_dim]
        
        # 5) 백본 마지막 stage feature(가장 글로벌 정보) 추가 활용
        #    ex) stage_idx가 가장 큰 값 (보통 self.stage_indices[-1])의 feature
        #    global pooling => shape [B, last_stage_channels]
        last_stage_idx = self.stage_indices[-1]
        last_stage_feat = final_stage_features[last_stage_idx]       # [B, C, H, W]
        pooled_global = F.adaptive_avg_pool2d(last_stage_feat, (1, 1))  # [B, C, 1, 1]
        pooled_global = pooled_global.view(B, -1)                    # [B, C] => ex) [B, 768]
        
        # 6) concat => shape [B, hidden_dim + last_stage_channels]
        cat_feature = torch.cat([basic_feature, pooled_global], dim=1)
        
        # 7) 최종 head
        score = self.fusion_head(cat_feature).squeeze(-1)  # [B] (or [B, 1] if squeeze(-1) 안 하면)
        return torch.sigmoid(score)


# output5 와 다른 점은 batchnorm 과 layernorm, 384 대신 512, 그리고 output 5 에 맨 마지막 feature map 을 추가했는데 원래는 추가 안햇음
