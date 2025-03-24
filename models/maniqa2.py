import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.swin import SwinTransformer
from einops import rearrange


class TABlock(nn.Module):
    """
    Token Attention Block: 입력 feature에 self-attention을 적용한 후 residual connection을 추가합니다.
    """
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape  # (batch, channel, patch_num)
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x  # Residual connection
        return x


class MANIQA(nn.Module):
    def __init__(
        self,
        embed_dim=72,
        num_outputs=1,
        drop=0.1, 
        depths=[2, 2],
        window_size=4,
        dim_mlp=768,
        img_size=224,
        num_tab=2,
        scale=0.8,
        **kwargs
    ):
        super().__init__()
        backbone_model = timm.create_model(
            'coatnet_2_rw_224.sw_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.backbone = backbone_model
        
        # 각 stage 출력 채널 수 확인
        feat_channels = [backbone_model.feature_info[i]['num_chs'] for i in (1, 2, 3, 4)]
        sum_ch = sum(feat_channels)
        
        # ---------------------------------------------------------
        # 2) 백본 출력 해상도 계산
        #    각 stage의 해상도: img_size // reduction
        #    최대 해상도를 기준으로 TABlock과 SwinTransformer를 구성
        # ---------------------------------------------------------
        feat_resolutions = [img_size // backbone_model.feature_info[i]['reduction'] for i in (1, 2, 3, 4)]
        max_res = max(feat_resolutions)  # 예: 224//4 = 56
        self.patch_resolution = (max_res, max_res)
        self.patch_num = max_res * max_res  # 예: 56*56 = 3136

        # ---------------------------------------------------------
        # 3) 이후 모듈 (TABlock, SwinTransformer 등) 정의
        # ---------------------------------------------------------
        self.tablock1 = nn.ModuleList([TABlock(self.patch_num) for _ in range(num_tab)])

        self.conv1 = nn.Conv2d(sum_ch, embed_dim, kernel_size=1, stride=1, padding=0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patch_resolution,
            depths=depths,
            num_heads=[4, 4],
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList([TABlock(self.patch_num) for _ in range(num_tab)])
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, padding=0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patch_resolution,
            depths=depths,
            num_heads=[4, 4],
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        # ---------------------------------------------------------
        # 4) 최종 품질 점수 회귀
        # ---------------------------------------------------------
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        1) CoAtNet 백본에서 4개 stage feature 추출  
        2) 서로 다른 해상도의 feature를 가장 큰 해상도(max_h, max_w)로 upsample  
        3) 채널 방향으로 concat 후, TABlock, SwinTransformer, FC를 거쳐 최종 품질 점수 계산
        """
        # 1) 백본에서 4개 stage feature 추출
        features = self.backbone(x)

        # 2) 각 feature map의 해상도를 동일하게 맞춤 (업샘플링)
        shapes = [f.shape for f in features]  # ex) (B, C_i, H_i, W_i)
        max_h = max(s[2] for s in shapes)
        max_w = max(s[3] for s in shapes)

        upsampled = []
        for f in features:
            b, c, h, w = f.shape
            if (h != max_h) or (w != max_w):
                f = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            upsampled.append(f)

        # 3) 채널 차원으로 concat → (B, sum_ch, max_h, max_w)
        x = torch.cat(upsampled, dim=1)

        # 4) MANIQA 처리: (B, C, H, W) → (B, C, H*W)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')

        # TABlock 1
        for tab in self.tablock1:
            x = tab(x)

        # (B, C, N) → (B, C, H, W)
        x = rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)

        # conv1 + SwinTransformer1
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # TABlock 2
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)

        # conv2 + SwinTransformer2
        x = self.conv2(x)
        x = self.swintransformer2(x)

        # 5) 최종 FC: (B, C, H, W) → (B, H*W, C) → weighted pooling을 통해 품질 점수 산출
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        device = x.device
        score = torch.empty(0, device=device)
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])   # (N, embed_dim//2)
            w = self.fc_weight(x[i])  # (N, 1)
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), dim=0)

        return score
