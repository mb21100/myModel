import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, transformer_layers=1, nhead=8, **kwargs):
        super().__init__()
        # MaxViT 백본 생성 (features_only=True로 각 stage의 feature map 추출)
        backbone_model = timm.create_model(
            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.backbone = backbone_model
        
        # 각 stage의 출력 채널 수 (예: [C1, C2, C3, C4])
        feat_channels = [backbone_model.feature_info[i]['num_chs'] for i in (1, 2, 3, 4)]
        self.sum_ch = sum(feat_channels)
        
        # 각 stage의 해상도 계산 (예: img_size // reduction)
        feat_resolutions = [img_size // backbone_model.feature_info[i]['reduction'] for i in (1, 2, 3, 4)]
        self.max_h = max(feat_resolutions)
        self.max_w = max(feat_resolutions)
        
        # 채널-wise로 융합된 feature map의 채널 수를 hidden_dim (예: 512)으로 축소하기 위한 conv1x1
        self.conv_reduction = nn.Conv2d(self.sum_ch, hidden_dim, kernel_size=1)
        
        # Transformer Encoder를 통한 토큰 재조정
        # 입력 토큰의 수: N = max_h * max_w, 토큰 차원: hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=drop, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Positional embedding (고정 이미지 크기를 가정)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_h * self.max_w, hidden_dim))
        
        # 최종 회귀를 위한 MLP (global pooling 후)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_outputs)
        )
    
    def forward(self, x):
        """
        1) 백본에서 4개 stage의 feature map 추출  
        2) 각 feature map을 최대 해상도 (max_h, max_w)로 업샘플링  
        3) 채널 차원으로 concat하여 통합 feature map 생성  
        4) conv1x1로 임베딩 차원 축소 → (B, hidden_dim, H, W)  
        5) (B, hidden_dim, H, W)를 (B, N, hidden_dim)으로 변환하고, positional embedding 추가  
        6) Transformer Encoder를 통해 토큰 간 상호작용 재조정  
        7) 재조정된 토큰을 다시 feature map으로 복원 후, global average pooling  
        8) MLP 회귀를 통해 최종 품질 점수 예측
        """
        # 1) 백본으로부터 feature map 추출 (list of tensors)
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
        
        # 4) conv1x1을 통해 임베딩 차원 축소 → (B, hidden_dim, max_h, max_w)
        x_embed = self.conv_reduction(x_cat)
        
        # 5) (B, hidden_dim, H, W) → (B, N, hidden_dim) (N = max_h * max_w) 및 positional embedding 추가
        x_tokens = rearrange(x_embed, 'b c h w -> b (h w) c')
        x_tokens = x_tokens + self.pos_embed
        
        # 6) Transformer Encoder 적용 (batch_first=True, so shape: (B, N, hidden_dim))
        x_tokens = self.transformer(x_tokens)
        
        # 7) 토큰을 다시 feature map으로 복원 후, global average pooling → (B, hidden_dim)
        x_out = rearrange(x_tokens, 'b (h w) c -> b c h w', h=self.max_h, w=self.max_w)
        x_pool = F.adaptive_avg_pool2d(x_out, 1).view(x_out.size(0), -1)
        
        # 8) MLP를 통해 최종 품질 점수 예측
        score = self.mlp(x_pool)
        return score
