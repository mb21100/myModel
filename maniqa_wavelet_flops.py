# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# import pywt
# from thop import profile  # pip install thop

# # FPN: multi-scale feature fusion
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
#             upsampled = F.interpolate(fpn_features[i+1], size=lateral_features[i].shape[2:], mode='nearest')
#             fpn_features[i] = lateral_features[i] + upsampled
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

# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16, kernel_size=7):
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

# # Cross-Attention Fusion Module
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, hidden_dim, num_tokens=4, num_heads=4):

#         super().__init__()
#         self.num_tokens = num_tokens
#         assert hidden_dim % num_tokens == 0, "hidden_dim must be divisible by num_tokens"
#         self.token_dim = hidden_dim // num_tokens
#         self.mha = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=num_heads)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
#     def forward(self, basic_feat, wavelet_feat):
#         # basic_feat: (B, hidden_dim)
#         # wavelet_feat: (B, hidden_dim) (projected to same dimension)
#         B = basic_feat.size(0)
#         # Reshape to (B, num_tokens, token_dim)
#         basic_tokens = basic_feat.view(B, self.num_tokens, self.token_dim)
#         wavelet_tokens = wavelet_feat.view(B, self.num_tokens, self.token_dim)
#         # Transpose to (num_tokens, B, token_dim) as required by nn.MultiheadAttention
#         basic_tokens = basic_tokens.transpose(0, 1)
#         wavelet_tokens = wavelet_tokens.transpose(0, 1)
#         # Cross-attention: basic as query, wavelet as key/value
#         attn_output, _ = self.mha(basic_tokens, wavelet_tokens, wavelet_tokens)
#         # Transpose back and flatten: (B, num_tokens, token_dim) -> (B, hidden_dim)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(B, -1)
#         # Optionally project fused feature
#         fused = self.out_proj(attn_output)
#         return fused

# # MANIQA_Wavelet with Cross-Attention Fusion
# class MANIQA(nn.Module):
#     def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, num_tokens=4, **kwargs):
#         super().__init__()
#         fpn_out_channels = 256
#         self.backbone = timm.create_model(
#             'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
#             pretrained=True,
#             features_only=True,
#             out_indices=(1, 2, 3, 4)
#         )
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
#             str(stage_idx): FusionModule(channels)
#             for stage_idx, channels in zip(self.stage_indices, self.feat_channels)
#         })
#         self.stage_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in self.feat_channels])
#         self.stage_drops = nn.ModuleList([nn.Dropout2d(drop) for _ in range(self.num_stages)])
#         self.fpn = FPN(in_channels_list=self.feat_channels, out_channels=fpn_out_channels)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         # MLP for basic branch feature
#         self.mlp_basic = nn.Sequential(
#             nn.Linear(fpn_out_channels * self.num_stages, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(drop)
#         )
#         # Wavelet branch: LL sub-band feature extraction from grayscale image
#         self.wavelet_linear = nn.Sequential(
#             nn.Linear((img_size//2) * (img_size//2), hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(drop)
#         )
#         self.wavelet_proj = nn.Linear(hidden_dim // 2, hidden_dim)
#         # Cross-Attention Fusion: 기본 feature와 wavelet feature를 융합
#         self.cross_attn = CrossAttentionFusion(hidden_dim, num_tokens=num_tokens, num_heads=4)
#         # Fusion head: cross-attended feature를 통해 최종 score 예측
#         self.fusion_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden_dim, num_outputs)
#         )
    
#     def _register_first_mbconv_hooks(self):

#         for name, module in self.backbone.named_modules():
#             if name.startswith("stages_"):
#                 parts = name.split('.')
#                 if len(parts) >= 5 and parts[1] == "blocks" and parts[2] == "0" and parts[3] == "conv" and parts[4] == "drop_path":
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
#         # Basic branch
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
#         fpn_features = self.fpn(fused_stage_features)
#         target_size = fpn_features[0].shape[2:]
#         upsampled_features = [F.interpolate(f, size=target_size, mode='nearest') for f in fpn_features]
#         fused_fpn_feature = torch.cat(upsampled_features, dim=1)
#         pooled_feature = self.global_pool(fused_fpn_feature)
#         flattened_feature = pooled_feature.view(pooled_feature.size(0), -1)
#         basic_feature = self.mlp_basic(flattened_feature)  # (B, hidden_dim)
        
#         # Wavelet branch
#         # Convert RGB to grayscale using standard luminance weights
#         grayscale = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
#         B, _, H, W = grayscale.shape
#         ll_list = []
#         for i in range(B):
#             img_np = grayscale[i, 0].detach().cpu().numpy()
#             coeffs2 = pywt.dwt2(img_np, 'haar')
#             LL, (LH, HL, HH) = coeffs2
#             ll_list.append(torch.tensor(LL, dtype=x.dtype, device=x.device))
#         ll_tensor = torch.stack(ll_list).unsqueeze(1)  # (B, 1, H/2, W/2)
#         ll_flat = ll_tensor.view(B, -1)
#         wavelet_feature = self.wavelet_linear(ll_flat)  # (B, hidden_dim//2)
#         wavelet_feature = self.wavelet_proj(wavelet_feature)  # (B, hidden_dim)
        
#         score = self.fusion_head(fused_feature).squeeze(-1)
#         return torch.sigmoid(score)


# if __name__ == "__main__":
#     model = MANIQA(num_outputs=1, img_size=224, drop=0.3, hidden_dim=512, num_tokens=4)
#     model.eval()
#     dummy_input = torch.randn(1, 3, 224, 224)
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("Output score shape:", output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pywt
from thop import profile  # pip install thop
import numpy as np

# FPN: multi-scale feature fusion
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):

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
            nn.ReLU(inplace=True),
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

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
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

# Cross-Attention Fusion Module
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_tokens=4, num_heads=4):

        super().__init__()
        self.num_tokens = num_tokens
        assert hidden_dim % num_tokens == 0
        self.token_dim = hidden_dim // num_tokens
        self.mha = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, basic_feat, wavelet_feat):
        # basic_feat: (B, hidden_dim)
        # wavelet_feat: (B, hidden_dim) (projected to same dimension)
        B = basic_feat.size(0)
        basic_tokens = basic_feat.view(B, self.num_tokens, self.token_dim)
        wavelet_tokens = wavelet_feat.view(B, self.num_tokens, self.token_dim)
        basic_tokens = basic_tokens.transpose(0, 1)  # (num_tokens, B, token_dim)
        wavelet_tokens = wavelet_tokens.transpose(0, 1)
        attn_output, _ = self.mha(basic_tokens, wavelet_tokens, wavelet_tokens)
        attn_output = attn_output.transpose(0, 1).contiguous().view(B, -1)
        fused = self.out_proj(attn_output)
        return fused

class MANIQA(nn.Module):
    def __init__(self, num_outputs=1, img_size=224, drop=0.1, hidden_dim=512, num_tokens=4, **kwargs):
        super().__init__()
        fpn_out_channels = 256
        self.backbone = timm.create_model(
            'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
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
        self.fpn = FPN(in_channels_list=self.feat_channels, out_channels=fpn_out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP for basic branch feature
        self.mlp_basic = nn.Sequential(
            nn.Linear(fpn_out_channels * self.num_stages, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        
        self.wavelet_linear = nn.Sequential(
            nn.Linear(3 * (img_size // 2) * (img_size // 2), hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        self.wavelet_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # Cross-Attention Fusion
        self.cross_attn = CrossAttentionFusion(hidden_dim, num_tokens=num_tokens, num_heads=4)
        
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
                if len(parts) >= 5 and parts[1] == "blocks" and parts[2] == "0" and parts[3] == "conv" and parts[4] == "drop_path":
                    try:
                        stage_idx = int(parts[0].replace("stages_", ""))
                    except Exception:
                        continue
                    module.register_forward_hook(
                        lambda m, inp, out, stage_idx=stage_idx: self._first_mbconv_hook(stage_idx, m, inp, out)
                    )
    
    def _first_mbconv_hook(self, stage_idx, module, inp, out):
        if stage_idx not in self.first_mbconv_features:
            self.first_mbconv_features[stage_idx] = out
    
    def forward(self, x):
        # Basic branch
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
        basic_feature = self.mlp_basic(flattened_feature)  # (B, hidden_dim)
        
        # Wavelet branch
        # RGB → Grayscale 
        grayscale = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        B, _, H, W = grayscale.shape
        
        hf_list = []
        for i in range(B):
            img_np = np.array(grayscale[i, 0].cpu())
            coeffs2 = pywt.dwt2(img_np, 'haar')
            # LL, (LH, HL, HH)
            _, (LH, HL, HH) = coeffs2
            hf_stack = np.stack([LH, HL, HH], axis=0)
            hf_tensor = torch.from_numpy(hf_stack).to(x.device, dtype=x.dtype)
            hf_list.append(hf_tensor)
        
        hf_tensor = torch.stack(hf_list, dim=0)  # (B, 3, H//2, W//2)
        hf_flat = hf_tensor.view(B, -1)
        wavelet_feature = self.wavelet_linear(hf_flat)  # (B, hidden_dim//2)
        wavelet_feature = self.wavelet_proj(wavelet_feature)  # (B, hidden_dim)
        
        # Cross-Attention Fusion
        fused_feature = self.cross_attn(basic_feature, wavelet_feature)  # (B, hidden_dim)
        
        score = self.fusion_head(fused_feature).squeeze(-1)
        return torch.sigmoid(score)

if __name__ == "__main__":
    model = MANIQA(num_outputs=1, img_size=224, drop=0.1, hidden_dim=512)
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
