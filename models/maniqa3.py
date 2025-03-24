# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from einops import rearrange


# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, **kwargs):
#         super().__init__()
#         # CoatNet-2 백본 생성 (features_only=True로 각 stage feature 추출)
#         backbone_model = timm.create_model(
#             'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
#             #'maxvit_small_tf_512',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1, 2, 3, 4)
#         )
#         self.backbone = backbone_model
        
#         # 각 stage의 출력 채널 수
#         feat_channels = [backbone_model.feature_info[i]['num_chs'] for i in (1, 2, 3, 4)]
#         self.sum_ch = sum(feat_channels)
        
#         # 각 stage의 해상도 계산 (예: img_size // reduction)
#         feat_resolutions = [img_size // backbone_model.feature_info[i]['reduction'] for i in (1, 2, 3, 4)]
#         self.max_h = max(feat_resolutions)
#         self.max_w = max(feat_resolutions)

#         # 간단한 MLP를 통한 품질 점수 회귀 (global average pooling 후)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.sum_ch, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, num_outputs)
#         )
    
#     def forward(self, x):
#         """
#         1) 백본에서 4개의 stage feature map 추출  
#         2) 각 feature map을 최대 해상도 (max_h, max_w)로 업샘플링  
#         3) 채널 차원으로 concat 후, global average pooling 수행  
#         4) MLP를 통해 최종 품질 점수 예측  
#         """
#         # 1) 백본으로부터 feature map 추출
#         features = self.backbone(x)

#         # 2) 각 feature map의 해상도를 동일하게 맞춤 (업샘플링)
#         upsampled = []
#         for f in features:
#             b, c, h, w = f.shape
#             if h != self.max_h or w != self.max_w:
#                 f = F.interpolate(f, size=(self.max_h, self.max_w), mode='bilinear', align_corners=False)
#             upsampled.append(f)

#         # 3) 채널 방향으로 concat → (B, sum_ch, max_h, max_w)
#         x_cat = torch.cat(upsampled, dim=1)
#         # Global average pooling: (B, sum_ch, max_h, max_w) → (B, sum_ch)
#         x_pool = F.adaptive_avg_pool2d(x_cat, 1).view(x_cat.size(0), -1)
        
#         # 4) MLP를 통해 품질 점수 예측
#         score = self.mlp(x_pool)
#         return score
import torch

import torch.nn as nn

import torch.nn.functional as F

import timm

 

class MANIQA(nn.Module):

    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, **kwargs):

        super().__init__()

        # CoatNet-2 백본 생성 (features_only=True로 각 stage feature 추출)

        backbone_model = timm.create_model(

            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',

            pretrained=True,

            features_only=True,

            out_indices=(1, 2, 3, 4)

        )

        self.backbone = backbone_model

       

        # 각 stage의 출력 채널 수

        feat_channels = [backbone_model.feature_info[i]['num_chs'] for i in (1, 2, 3, 4)]

        self.sum_ch = sum(feat_channels)

       

        # 각 stage의 해상도 계산 (예: img_size // reduction)

        feat_resolutions = [img_size // backbone_model.feature_info[i]['reduction'] for i in (1, 2, 3, 4)]

        self.max_h = max(feat_resolutions)

        self.max_w = max(feat_resolutions)

 

        # 수정된 MLP: BatchNorm과 추가 hidden layer를 통해 회귀 성능 강화

        self.mlp = nn.Sequential(

            nn.Linear(self.sum_ch, hidden_dim),

            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(),

            nn.Dropout(drop),

            nn.Linear(hidden_dim, hidden_dim // 2),

            nn.BatchNorm1d(hidden_dim // 2),

            nn.ReLU(),

            nn.Dropout(drop),

            nn.Linear(hidden_dim // 2, num_outputs)

        )

   

    def forward(self, x):

        """

        1) 백본에서 4개의 stage feature map 추출 

        2) 각 feature map을 최대 해상도 (max_h, max_w)로 업샘플링 

        3) 채널 차원으로 concat 후, global average pooling 수행 

        4) 수정된 MLP를 통해 최종 품질 점수 예측 

        """

        # 1) 백본으로부터 feature map 추출

        features = self.backbone(x)

 

        # 2) 각 feature map의 해상도를 동일하게 맞춤 (업샘플링)

        upsampled = []

        for f in features:

            b, c, h, w = f.shape

            if h != self.max_h or w != self.max_w:

                f = F.interpolate(f, size=(self.max_h, self.max_w), mode='bilinear', align_corners=False)

            upsampled.append(f)

 

        # 3) 채널 방향으로 concat → (B, sum_ch, max_h, max_w)

        x_cat = torch.cat(upsampled, dim=1)

        # Global average pooling: (B, sum_ch, max_h, max_w) → (B, sum_ch)

        x_pool = F.adaptive_avg_pool2d(x_cat, 1).view(x_cat.size(0), -1)

       

        # 4) 수정된 MLP를 통해 품질 점수 예측

        score = self.mlp(x_pool)

        return score