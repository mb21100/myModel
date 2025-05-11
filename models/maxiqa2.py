# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# # cbam 을 사용하여 각 스테이지에서 나온 피쳐맵을 refine 한다
# # 1epoch 에 plcc:0.733, srcc:0.770

################# 여기 부분은 hook 이 안된 코드다.

# class FPN(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
        
#         """
#         in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [C1, C2, C3, C4])
#         out_channels: FPN에서 사용할 통일된 채널 수 (예: 256)
#         """
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         self.smooth_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
#         ])

#     def forward(self, features):
#         # 각 스테이지에 lateral conv 적용
#         lateral_features = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        
#         # Top-down 경로: 가장 깊은 스테이지부터 시작하여 upsample 후 덧셈
#         fpn_features = [None] * len(lateral_features)
#         fpn_features[-1] = lateral_features[-1]
#         for i in range(len(lateral_features) - 2, -1, -1):
#             upsampled = F.interpolate(fpn_features[i+1], size=lateral_features[i].shape[2:], mode='nearest')
#             fpn_features[i] = lateral_features[i] + upsampled
        
#         # smoothing conv 적용
#         fpn_features = [s_conv(f) for f, s_conv in zip(fpn_features, self.smooth_convs)]
#         return fpn_features


# # Channel Attention Module
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# # Spatial Attention Module
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         # 평균, 최대값 pooling을 채널 차원에 대해 수행
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         attn = self.conv(x_cat)
#         return self.sigmoid(attn)

# # CBAM 모듈: 채널 및 공간 어텐션 결합
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         # 채널 어텐션 적용
#         out = x * self.channel_attention(x)
#         # 공간 어텐션 적용
#         out = out * self.spatial_attention(out)
#         return out


# # 기존 FusionModule에 CBAM 통합: 두 feature map을 concat 후 1x1 Conv, BN, ReLU, 그 후 CBAM 적용
# class FusionModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.act = nn.ReLU(inplace=True)
#         self.cbam = CBAM(channels)  # CBAM 모듈 추가

#     def forward(self, feat_local, feat_global):
#         x = torch.cat([feat_local, feat_global], dim=1)
#         x = self.conv1x1(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.cbam(x)  # 어텐션 적용
#         return x


# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, **kwargs):
#         super().__init__()
#         fpn_out_channels = 256
#         # Backbone: MaxViT 모델 생성, features_only 옵션으로 각 stage의 최종 feature map 반환
#         self.backbone = timm.create_model(
#             'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1, 2, 3, 4)
#         )
       
#         # 첫 번째 MbConv (로컬 특징)를 저장할 딕셔너리
#         self.first_mbconv_features = {}
#         self._register_first_mbconv_hooks()

#         # backbone.feature_info[1:5]를 사용하여 각 stage의 채널 수와 stage 인덱스 결정
#         feat_infos = self.backbone.feature_info[1:5]
#         self.stage_indices = []  # 실제 stage index (예: 1, 2, 3, 4)
#         self.feat_channels = []
#         for info in feat_infos:
#             # module 이름은 "stages.<stage_idx>" 형식이라 가정
#             stage_idx = int(info['module'].split('.')[1])
#             self.stage_indices.append(stage_idx)
#             self.feat_channels.append(info['num_chs'])
#         self.num_stages = len(self.feat_channels)

#         # 각 stage별 fusion 모듈 (채널 수는 해당 stage의 feature map 채널 수)
#         self.fusion_modules = nn.ModuleDict({
#             str(stage_idx): FusionModule(channels)
#             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })

#         # 각 stage의 최종 feature map을 정규화, Global Pooling, Dropout 후 flatten
#         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
#         #self.stage_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.num_stages)])
#         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
#         # 모든 stage feature vector 차원의 합
#         self.fpn = FPN(in_channels_list=self.feat_channels, out_channels=fpn_out_channels)

#         # 최종 global pooling 및 MLP: FPN 출력은 num_stages * fpn_out_channels 채널 수를 갖게 됨
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#         # 수정된 MLP: BatchNorm과 추가 hidden layer를 통해 회귀 성능 강화
#         self.mlp = nn.Sequential(
#             nn.Linear(fpn_out_channels * self.num_stages, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim // 2, num_outputs)
#         )

#     def _first_mbconv_hook(self, stage_idx, module, input, output):
#         # 각 stage의 첫 번째 MbConvBlock의 출력을 한 번만 저장합니다.
#         if stage_idx not in self.first_mbconv_features:
#             self.first_mbconv_features[stage_idx] = output

#     def _register_first_mbconv_hooks(self):
#         """
#         backbone 내부의 모든 모듈 중, 이름이 "stages.<stage_idx>.blocks.0"인 MbConvBlock에
#         forward hook을 등록하여 첫 번째 블록의 MBConv 출력을 저장합니다.
#         """
#         for name, module in self.backbone.named_modules():
#             if module.__class__.__name__ == 'MbConvBlock' and name.startswith('stages.'):
#                 parts = name.split('.')
#                 # parts[0] == "stages", parts[1] == stage_idx, parts[2] == "blocks", parts[3] == "0"
#                 if len(parts) >= 4 and parts[2] == "blocks" and parts[3] == "0":
#                     stage_idx = int(parts[1])
#                     module.register_forward_hook(
#                         lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
#                     )

#     def forward(self, x):
#         # 매 forward마다 hook 저장소 초기화
#         self.first_mbconv_features = {}
       
#         # backbone forward: features_only 옵션으로 각 stage의 최종 feature map 리스트 반환
#         backbone_feats = self.backbone(x)
#         # backbone.feature_info[1:5]와 매핑하여, 각 stage의 최종 feature map을 딕셔너리로 저장
#         final_stage_features = {}
#         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
#             stage_idx = int(info['module'].split('.')[1])
#             final_stage_features[stage_idx] = feat

#         # 각 stage별로, 첫 번째 MBConv 출력(로컬 특징)과 최종 feature map(전역 특징)을 fusion
#         fused_stage_features = []
#         for i, stage_idx in enumerate(self.stage_indices):
#             # hook에 저장된 첫 번째 MBConv feature; 없으면 최종 feature map을 사용 (fallback)
#             feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
#             feat_global = final_stage_features[stage_idx]
#             # 두 feature map은 NCHW 형식, 채널 수 및 공간 해상도가 동일해야 함
#             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
#             # 각 stage의 fused feature map에 대해 정규화, Global Average Pooling, Dropout, flatten 수행
#             fused_feat = self.stage_norms[i](fused_feat)
#             #fused_feat = self.stage_pools[i](fused_feat)  # (B, C, 1, 1)
#             fused_feat = self.stage_drops[i](fused_feat)
#             #fused_feat = fused_feat.view(fused_feat.size(0), -1)  # (B, C)
#             fused_stage_features.append(fused_feat)

#         # FPN을 통해 multi-scale feature fusion 진행
#         fpn_features = self.fpn(fused_stage_features)  # 리스트 내 각 feature map: (B, fpn_out_channels, H_i, W_i)
        
#         # 모든 FPN 출력 feature map을 가장 높은 해상도(첫 번째 feature map의 해상도)로 upsample한 후 concat
#         target_size = fpn_features[0].shape[2:]
#         upsampled_features = [F.interpolate(f, size=target_size, mode='nearest') for f in fpn_features]
#         fused_fpn_feature = torch.cat(upsampled_features, dim=1)  # (B, fpn_out_channels * num_stages, H, W)

#         # Global Average Pooling 후 flatten
#         pooled_feature = self.global_pool(fused_fpn_feature)  # (B, fpn_out_channels * num_stages, 1, 1)
#         flattened_feature = pooled_feature.view(pooled_feature.size(0), -1)  # (B, fpn_out_channels * num_stages)
        
#         # MLP를 통해 최종 품질 점수 예측
#         score = self.mlp(flattened_feature).squeeze(-1)
#         score = torch.sigmoid(score)
#         # 비선형 logistic mapping을 적용해 예측 점수 스케일 조정 (PLCC 개선 목적)
#         return score

# # # ------------------------- 테스트 예시 -------------------------
# # if __name__ == "__main__":
# #     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
# #     dummy_input = torch.randn(2, 3, 224, 224)
# #     score = model(dummy_input)
# #     print("Output score shape:", score.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from thop import profile  # pip install thop
# # cbam 을 사용하여 각 스테이지에서 나온 피쳐맵을 refine 한다
# # 1epoch 에 plcc:0.733, srcc:0.770



# class FPN(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
        
#         """
#         in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [C1, C2, C3, C4])
#         out_channels: FPN에서 사용할 통일된 채널 수 (예: 256)
#         """
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
#         ])
#         self.smooth_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
#         ])

#     def forward(self, features):
#         # 각 스테이지에 lateral conv 적용
#         lateral_features = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        
#         # Top-down 경로: 가장 깊은 스테이지부터 시작하여 upsample 후 덧셈
#         fpn_features = [None] * len(lateral_features)
#         fpn_features[-1] = lateral_features[-1]
#         for i in range(len(lateral_features) - 2, -1, -1):
#             upsampled = F.interpolate(fpn_features[i+1], size=lateral_features[i].shape[2:], mode='nearest')
#             fpn_features[i] = lateral_features[i] + upsampled
        
#         # smoothing conv 적용
#         fpn_features = [s_conv(f) for f, s_conv in zip(fpn_features, self.smooth_convs)]
#         return fpn_features


# # Channel Attention Module
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# # Spatial Attention Module
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         # 평균, 최대값 pooling을 채널 차원에 대해 수행
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         attn = self.conv(x_cat)
#         return self.sigmoid(attn)

# # CBAM 모듈: 채널 및 공간 어텐션 결합
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
    
#     def forward(self, x):
#         # 채널 어텐션 적용
#         out = x * self.channel_attention(x)
#         # 공간 어텐션 적용
#         out = out * self.spatial_attention(out)
#         return out


# # 기존 FusionModule에 CBAM 통합: 두 feature map을 concat 후 1x1 Conv, BN, ReLU, 그 후 CBAM 적용
# class FusionModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)
#         self.act = nn.ReLU(inplace=True)
#         self.cbam = CBAM(channels)  # CBAM 모듈 추가

#     def forward(self, feat_local, feat_global):
#         x = torch.cat([feat_local, feat_global], dim=1)
#         x = self.conv1x1(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.cbam(x)  # 어텐션 적용
#         return x


# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, **kwargs):
#         super().__init__()
#         fpn_out_channels = 256
#         # Backbone: MaxViT 모델 생성, features_only 옵션으로 각 stage의 최종 feature map 반환
#         self.backbone = timm.create_model(
#             'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1, 2, 3, 4)
#         )
       
#         # 첫 번째 MbConv (로컬 특징)를 저장할 딕셔너리
#         self.first_mbconv_features = {}
#         self._register_first_mbconv_hooks()

#         # backbone.feature_info[1:5]를 사용하여 각 stage의 채널 수와 stage 인덱스 결정
#         feat_infos = self.backbone.feature_info[1:5]
#         self.stage_indices = []  # 실제 stage index (예: 1, 2, 3, 4)
#         self.feat_channels = []
#         for info in feat_infos:
#             # module 이름은 "stages.<stage_idx>" 형식이라 가정
#             stage_idx = int(info['module'].split('.')[1])
#             self.stage_indices.append(stage_idx)
#             self.feat_channels.append(info['num_chs'])
#         self.num_stages = len(self.feat_channels)

#         # 각 stage별 fusion 모듈 (채널 수는 해당 stage의 feature map 채널 수)
#         self.fusion_modules = nn.ModuleDict({
#             str(stage_idx): FusionModule(channels)
#             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })

#         # 각 stage의 최종 feature map을 정규화, Global Pooling, Dropout 후 flatten
#         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
#         #self.stage_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.num_stages)])
#         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
        
#         # 모든 stage feature vector 차원의 합
#         self.fpn = FPN(in_channels_list=self.feat_channels, out_channels=fpn_out_channels)

#         # 최종 global pooling 및 MLP: FPN 출력은 num_stages * fpn_out_channels 채널 수를 갖게 됨
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#         # 수정된 MLP: BatchNorm과 추가 hidden layer를 통해 회귀 성능 강화
#         self.mlp = nn.Sequential(
#             nn.Linear(fpn_out_channels * self.num_stages, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim // 2, num_outputs)
#         )

#     def _first_mbconv_hook(self, stage_idx, module, input, output):
#         # 각 stage의 첫 번째 MbConvBlock의 출력을 한 번만 저장합니다.
#         if stage_idx not in self.first_mbconv_features:
#             self.first_mbconv_features[stage_idx] = output

#     def _register_first_mbconv_hooks(self):
#         """
#         backbone 내부의 모든 모듈 중, 이름이 "stages.<stage_idx>.blocks.0"인 MbConvBlock에
#         forward hook을 등록하여 첫 번째 블록의 MBConv 출력을 저장합니다.
#         """
#         for name, module in self.backbone.named_modules():
#             if module.__class__.__name__ == 'MbConvBlock' and name.startswith('stages.'):
#                 parts = name.split('.')
#                 # parts[0] == "stages", parts[1] == stage_idx, parts[2] == "blocks", parts[3] == "0"
#                 if len(parts) >= 4 and parts[2] == "blocks" and parts[3] == "0":
#                     stage_idx = int(parts[1])
#                     module.register_forward_hook(
#                         lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
#                     )

#     def forward(self, x):
#         # 매 forward마다 hook 저장소 초기화
#         self.first_mbconv_features = {}
       
#         # backbone forward: features_only 옵션으로 각 stage의 최종 feature map 리스트 반환
#         backbone_feats = self.backbone(x)
#         # backbone.feature_info[1:5]와 매핑하여, 각 stage의 최종 feature map을 딕셔너리로 저장
#         final_stage_features = {}
#         for info, feat in zip(self.backbone.feature_info[1:5], backbone_feats):
#             stage_idx = int(info['module'].split('.')[1])
#             final_stage_features[stage_idx] = feat

#         # 각 stage별로, 첫 번째 MBConv 출력(로컬 특징)과 최종 feature map(전역 특징)을 fusion
#         fused_stage_features = []
#         for i, stage_idx in enumerate(self.stage_indices):
#             # hook에 저장된 첫 번째 MBConv feature; 없으면 최종 feature map을 사용 (fallback)
#             feat_local = self.first_mbconv_features.get(stage_idx, final_stage_features[stage_idx])
#             feat_global = final_stage_features[stage_idx]
#             # 두 feature map은 NCHW 형식, 채널 수 및 공간 해상도가 동일해야 함
#             fused_feat = self.fusion_modules[str(stage_idx)](feat_local, feat_global)
#             # 각 stage의 fused feature map에 대해 정규화, Global Average Pooling, Dropout, flatten 수행
#             fused_feat = self.stage_norms[i](fused_feat)
#             #fused_feat = self.stage_pools[i](fused_feat)  # (B, C, 1, 1)
#             fused_feat = self.stage_drops[i](fused_feat)
#             #fused_feat = fused_feat.view(fused_feat.size(0), -1)  # (B, C)
#             fused_stage_features.append(fused_feat)

#         # FPN을 통해 multi-scale feature fusion 진행
#         fpn_features = self.fpn(fused_stage_features)  # 리스트 내 각 feature map: (B, fpn_out_channels, H_i, W_i)
        
#         # 모든 FPN 출력 feature map을 가장 높은 해상도(첫 번째 feature map의 해상도)로 upsample한 후 concat
#         target_size = fpn_features[0].shape[2:]
#         upsampled_features = [F.interpolate(f, size=target_size, mode='nearest') for f in fpn_features]
#         fused_fpn_feature = torch.cat(upsampled_features, dim=1)  # (B, fpn_out_channels * num_stages, H, W)

#         # Global Average Pooling 후 flatten
#         pooled_feature = self.global_pool(fused_fpn_feature)  # (B, fpn_out_channels * num_stages, 1, 1)
#         flattened_feature = pooled_feature.view(pooled_feature.size(0), -1)  # (B, fpn_out_channels * num_stages)
        
#         # MLP를 통해 최종 품질 점수 예측
#         score = self.mlp(flattened_feature).squeeze(-1)
#         score = torch.sigmoid(score)
#         # 비선형 logistic mapping을 적용해 예측 점수 스케일 조정 (PLCC 개선 목적)
#         return score



# if __name__ == "__main__":
#     # 모델 생성 및 평가 모드 전환
#     model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
#     model.eval()

#     # 더미 입력 생성 (배치 사이즈 1, 3채널, 224x224)
#     dummy_input = torch.randn(1, 3, 224, 224)

#     # FLOPs와 파라미터 수 계산
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
#     print("FLOPs: {:.3f} G".format(flops / 1e9))
#     print("Parameters: {:.3f} M".format(params / 1e6))

#     # Forward pass 실행 (hook이 동작하여 각 stage의 첫번째 MBConv 출력이 저장됨)
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("Output score shape:", output.shape)

#     # hook을 통해 저장된 각 stage의 첫 번째 MBConv feature map 확인
#     if model.first_mbconv_features:
#         print("Extracted first MBConv features from each stage:")
#         for stage_idx, feat in model.first_mbconv_features.items():
#             print(f"  Stage {stage_idx}: feature shape = {feat.shape}")
#     else:
#         print("No MBConv features were extracted by hooks.")

#     # Backbone의 최종 feature map 확인 (features_only 옵션 사용)
#     backbone_feats = model.backbone(dummy_input)
#     print("Backbone final feature maps:")
#     for idx, feat in enumerate(backbone_feats, 1):
#         print(f"  Stage {idx}: feature shape = {feat.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from thop import profile  # pip install thop

# CBAM을 사용하여 각 스테이지에서 나온 피쳐맵을 refine 한다
# 1epoch 에 plcc:0.733, srcc:0.770

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        in_channels_list: 각 스테이지 feature map의 채널 수 리스트 (예: [C1, C2, C3, C4])
        out_channels: FPN에서 사용할 통일된 채널 수 (예: 256)
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        lateral_features = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        fpn_features = [None] * len(lateral_features)
        fpn_features[-1] = lateral_features[-1]
        for i in range(len(lateral_features) - 2, -1, -1):
            upsampled = F.interpolate(fpn_features[i+1], size=lateral_features[i].shape[2:], mode='nearest')
            fpn_features[i] = lateral_features[i] + upsampled
        fpn_features = [s_conv(f) for f, s_conv in zip(fpn_features, self.smooth_convs)]
        return fpn_features

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# Spatial Attention Module
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

# CBAM 모듈
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)

# FusionModule: 두 feature map을 concat 후 1x1 Conv, BN, ReLU, CBAM 적용
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

class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.3, hidden_dim=512, **kwargs):
        super().__init__()
        fpn_out_channels = 256
        self.backbone = timm.create_model(
            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
       
        # hook을 통해 각 stage의 첫 번째 conv.drop_path 출력 저장
        self.first_mbconv_features = {}
        self._register_first_mbconv_hooks()

        # stage 정보 추출 (예: stages_1 → stage 1 등)
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
        self.fpn = FPN(in_channels_list=self.feat_channels, out_channels=fpn_out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(fpn_out_channels * self.num_stages, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, num_outputs)
        )

    def _first_mbconv_hook(self, stage_idx, module, input, output):
        if stage_idx not in self.first_mbconv_features:
            self.first_mbconv_features[stage_idx] = output

    def _register_first_mbconv_hooks(self):
        """
        backbone 내부의 모듈 중 이름이 "stages_<stage_idx>.blocks.0.conv.drop_path"인 모듈에 대해 hook 등록.
        """
        for name, module in self.backbone.named_modules():
            # 이름이 "stages_"로 시작하는지 확인
            if name.startswith("stages_"):
                parts = name.split('.')
                # 예상 구조: parts[0] = "stages_X", parts[1] = "blocks", parts[2] = "0", parts[3] = "conv", parts[4] = "drop_path"
                if len(parts) >= 5 and parts[1] == "blocks" and parts[2] == "0" and parts[3] == "conv" and parts[4] == "drop_path":
                    try:
                        stage_idx = int(parts[0].replace("stages_", ""))
                    except Exception:
                        continue
                    module.register_forward_hook(
                        lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
                    )

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

        fpn_features = self.fpn(fused_stage_features)
        target_size = fpn_features[0].shape[2:]
        upsampled_features = [F.interpolate(f, size=target_size, mode='nearest') for f in fpn_features]
        fused_fpn_feature = torch.cat(upsampled_features, dim=1)
        pooled_feature = self.global_pool(fused_fpn_feature)
        flattened_feature = pooled_feature.view(pooled_feature.size(0), -1)
        score = self.mlp(flattened_feature).squeeze(-1)
        return torch.sigmoid(score)

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
