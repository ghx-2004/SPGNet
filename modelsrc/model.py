import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import Tensor, nn
from torchvision.ops import DeformConv2d

from adaWin import SwinTransformerBlock
from physics import ThermalPhysicsPrior
from Swin import SwinTransformer


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            act(),
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim)
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=8)
        self.norm_attn = norm(in_dim)
        self.attn_weight = nn.Parameter(torch.tensor(0.5))

        self.conv1 = nn.Conv2d(in_dim, in_dim // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_dim // 2, in_dim, 3, padding=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge

        B, C, H, W = edge.shape
        edge_flat = edge.view(B, C, H * W).permute(2, 0, 1)
        attn_output, _ = self.attn(edge_flat, edge_flat, edge_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        attn_output = self.norm_attn(attn_output)

        edge1 = self.conv1(edge)
        edge1 = F.leaky_relu(edge1, negative_slope=0.1)
        edge1 = self.upsample(edge1)
        edge1 = self.conv2(edge1)
        edge1 = F.interpolate(edge1, size=(H, W), mode='bilinear', align_corners=False)

        edge = self.out_conv(edge)
        edge = edge + self.attn_weight * attn_output + edge1

        return x + edge

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

# model
class TMSOD(nn.Module):
    def __init__(self):
        super(TMSOD, self).__init__()
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], in_chans=1)
        self.MSA_sem = GMSA_ini(d_model=1024, num_layers=3, resolution=(16, 18))
        self.conv_sem = conv3x3_bn_relu(1024*2, 1024)
        self.MSA4_r = SwinTransformerBlock(dim=1024, input_resolution=(12,12), num_heads=2, up_ratio=1, out_channels=1024)
        self.MSA4_t = SwinTransformerBlock(dim=1024, input_resolution=(12, 12), num_heads=2, up_ratio=1,out_channels=1024)
        self.MSA3_r = SwinTransformerBlock(dim=512, input_resolution=(24,24), num_heads=2, up_ratio=2, out_channels=512)
        self.MSA3_t = SwinTransformerBlock(dim=512, input_resolution=(24, 24), num_heads=2, up_ratio=2, out_channels=512)
        self.MSA2_r = SwinTransformerBlock(dim=256, input_resolution=(48,48), num_heads=2, up_ratio=4, out_channels=256)
        self.MSA2_t = SwinTransformerBlock(dim=256, input_resolution=(48, 48), num_heads=2, up_ratio=4, out_channels=256)

        self.align_att4 = get_aligned_feat(inC=1024, outC=1024)
        self.align_att3 = get_aligned_feat(inC=512, outC=512)
        self.align_att2 = get_aligned_feat(inC=256, outC=256)
        self.convAtt4 = conv3x3_bn_relu(1024*2, 1024)
        self.convAtt3 = conv3x3_bn_relu(512*2, 512)
        self.convAtt2 = conv3x3_bn_relu(256*2, 256)

        # === EdgeEnhancer ===
        self.edge4 = EdgeEnhancer(in_dim=1024, norm=nn.BatchNorm2d, act=nn.ReLU)
        self.edge3 = EdgeEnhancer(in_dim=1024, norm=nn.BatchNorm2d, act=nn.ReLU)
        self.edge2 = EdgeEnhancer(in_dim=512, norm=nn.BatchNorm2d, act=nn.ReLU)
        self.edge1 = EdgeEnhancer(in_dim=256, norm=nn.BatchNorm2d, act=nn.ReLU)

        # Enhance the underlying feature perception module
        self.shallow_fusion = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.conv64 = conv3x3(64, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.thermal_prior = ThermalPhysicsPrior()

        # Channel transformation to adapt it to different layers of the Swin Transformer
        self.thermal_proj_1024 = nn.Conv2d(128, 1024, kernel_size=1)
        self.thermal_gate_proj = nn.Conv2d(128, 1024, kernel_size=1)  # ✅ 新加的
        self.thermal_proj_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.thermal_proj_256 = nn.Conv2d(128, 256, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)
        self.pred_saliency = None
        self.debug_thermal = None
        self.P_thermal = self.thermal_prior
        self.attention4 = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.attention3 = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=8)
    def forward(self, rgb, t):

        fr = self.rgb_swin(rgb)
        ft = self.t_swin(t)

        P_thermal = self.thermal_prior(t)

        # Convenient for subsequent visualization
        self.debug_thermal = P_thermal

        # Transformers adapted to different scales
        P_thermal_1024 = self.thermal_proj_1024(P_thermal)  # (B, 1024, H, W)
        P_thermal_512 = self.thermal_proj_512(P_thermal)  # (B, 512, H, W)
        P_thermal_256 = self.thermal_proj_256(P_thermal)  # (B, 256, H, W)

        # Make sure the size matches the Transformer input
        P_thermal_1024 = F.interpolate(P_thermal_1024, size=(fr[3].shape[2], fr[3].shape[3]), mode='bilinear',
                                       align_corners=False)
        P_thermal_512 = F.interpolate(P_thermal_512, size=(fr[2].shape[2], fr[2].shape[3]), mode='bilinear',
                                      align_corners=False)
        P_thermal_256 = F.interpolate(P_thermal_256, size=(fr[1].shape[2], fr[1].shape[3]), mode='bilinear',
                                      align_corners=False)

        semantic, p_saliency = self.MSA_sem(
            torch.cat((fr[3].flatten(2).transpose(1, 2),
                       ft[3].flatten(2).transpose(1, 2)), dim=1),
            torch.cat((fr[3].flatten(2).transpose(1, 2),
                       ft[3].flatten(2).transpose(1, 2)), dim=1)
        )

        # Save it for the purpose of calculating the loss
        self.pred_saliency = p_saliency

        semantic1, semantic2 = torch.split(semantic, fr[3].shape[2] * fr[3].shape[3], dim=1)
        semantic = self.conv_sem(torch.cat((
            semantic1.view(semantic1.shape[0], int(np.sqrt(semantic1.shape[1])), int(np.sqrt(semantic1.shape[1])), -1)
            .permute(0, 3, 1, 2).contiguous(),
            semantic2.view(semantic2.shape[0], int(np.sqrt(semantic2.shape[1])), int(np.sqrt(semantic2.shape[1])), -1)
            .permute(0, 3, 1, 2).contiguous()
        ), dim=1))

        # Insert the thermal modulation gating mechanism (element-by-element multiplication)
        P_sem = F.interpolate(P_thermal, size=semantic.shape[2:], mode='bilinear', align_corners=False)
        P_sem = self.thermal_gate_proj(P_sem)  # Project to semantic with the same number of channels
        semantic = semantic * P_sem  # Lightweight thermal modulation

        att_4_r = self.MSA4_r(fr[3].flatten(2).transpose(1, 2),
                              ft[3].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_1024,
                              p_saliency = p_saliency, # Pass in the saliency map
                              alpha = 4,
                              threshold_sal = 0.3
                              )

        att_4_t = self.MSA4_t(ft[3].flatten(2).transpose(1, 2),
                              fr[3].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_1024, p_saliency=p_saliency, alpha=4, threshold_sal=0.3)

        att_3_r = self.MSA3_r(fr[2].flatten(2).transpose(1, 2),
                              ft[2].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_512, p_saliency=p_saliency, alpha=4, threshold_sal=0.3)

        att_3_t = self.MSA3_t(ft[2].flatten(2).transpose(1, 2),
                              fr[2].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_512, p_saliency=p_saliency, alpha=4, threshold_sal=0.3)

        att_2_r = self.MSA2_r(fr[1].flatten(2).transpose(1, 2),
                              ft[1].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_256, p_saliency=p_saliency, alpha=4, threshold_sal=0.3)

        att_2_t = self.MSA2_t(ft[1].flatten(2).transpose(1, 2),
                              fr[1].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_256, p_saliency=p_saliency, alpha=4, threshold_sal=0.3)

        # Fuse low-level features of RGB and TIR
        r1 = self.shallow_fusion(torch.cat([fr[0], ft[0]], dim=1))

        # Parse the output of Swin Transformer at different layers
        r4 = att_4_r.view(att_4_r.shape[0], fr[3].shape[2], fr[3].shape[3], -1).permute(0, 3, 1, 2).contiguous()
        t4 = att_4_t.view(att_4_t.shape[0], fr[3].shape[2], fr[3].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        p_saliency_4 = F.interpolate(
            p_saliency, size=(r4.shape[2], r4.shape[3]),
            mode='bilinear', align_corners=False
        )

        F_final4, feat_r2t4, feat_t2r4 = self.align_att4(
            r4, t4, thermal_mask=p_saliency_4
        )

        self.feat_r2t4 = feat_r2t4
        self.feat_t2r4 = feat_t2r4
        self.r4_orig = r4
        self.t4_orig = t4

        r4 = self.convAtt4(torch.cat((r4, F_final4), dim=1))

        r3 = att_3_r.view(att_3_r.shape[0], fr[2].shape[2], fr[2].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        t3 = att_3_t.view(att_3_t.shape[0], fr[2].shape[2], fr[2].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        p_saliency_3 = F.interpolate(
            p_saliency, size=(r3.shape[2], r3.shape[3]),
            mode='bilinear', align_corners=False
        )

        F_final3, feat_r2t3, feat_t2r3 = self.align_att3(
            r3, t3, thermal_mask=p_saliency_3
        )

        self.feat_r2t3 = feat_r2t3
        self.feat_t2r3 = feat_t2r3
        self.r3_orig = r3
        self.t3_orig = t3

        r3 = self.convAtt3(torch.cat((r3, F_final3), dim=1))

        r2 = att_2_r.view(att_2_r.shape[0], fr[1].shape[2], fr[1].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        t2 = att_2_t.view(att_2_t.shape[0], fr[1].shape[2], fr[1].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        p_saliency_2 = F.interpolate(
            p_saliency, size=(r2.shape[2], r2.shape[3]),
            mode='bilinear', align_corners=False
        )

        F_final2, feat_r2t2, feat_t2r2 = self.align_att2(
            r2, t2, thermal_mask=p_saliency_2
        )

        self.feat_r2t2 = feat_r2t2
        self.feat_t2r2 = feat_t2r2
        self.r2_orig = r2
        self.t2_orig = t2

        r2 = self.convAtt2(torch.cat((r2, F_final2), dim=1))

        # Decode
        r4_up = self.up2(r4)
        r4_fused = self.edge4(r4_up)
        r4 = self.decode4(r4_fused)

        r3 = torch.cat([r3, r4], dim=1)
        r3 = self.up2(r3)
        r3 = self.edge3(r3)
        r3 = self.decode3(r3)

        r2 = torch.cat([r2, r3], dim=1)
        r2 = self.up2(r2)
        r2 = self.edge2(r2)
        r2 = self.decode2(r2)

        r1 = torch.cat([r1, r2], dim=1)
        r1 = self.edge1(r1)
        r1 = self.decode1(r1)

        out = self.up4(r1)
        out = self.conv64(out)
        out = self.sigmoid(out)
        return out, p_saliency, P_thermal

    def load_pre(self, pre_model):
        state_dict = torch.load(pre_model)['model']

        # Load the RGB branch weights
        self.rgb_swin.load_state_dict(state_dict, strict=False)
        print(f"RGB SwinTransformer loaded from {pre_model}")

        # The Thermal channel is 1, and the weights need to be manually adapted (by copying the mean values of the first three channels of RGB).
        rgb_weight = state_dict['patch_embed.proj.weight']  # [128, 3, 4, 4]
        thermal_weight = rgb_weight.mean(dim=1, keepdim=True)  # [128, 1, 4, 4]
        state_dict['patch_embed.proj.weight'] = thermal_weight
        self.t_swin.load_state_dict(state_dict, strict=False)
        print(f"Thermal SwinTransformer loaded from {pre_model} (converted to 1-channel)")

    # ======================= Modal consistency loss =======================
    def consistency_loss(self, lamda=0.1):

        import torch.nn.functional as F

        # ------ (1) Define an internal small function to perform L1/L2/KL equidistant measurements on two feature maps ------
        def _dist_loss(featA, featB):
            featA_norm = F.normalize(featA, p=2, dim=1)
            featB_norm = F.normalize(featB, p=2, dim=1)
            return F.l1_loss(featA_norm, featB_norm)

        # ------ (2) For different layers, calculate the distance between the "forward/reverse" and the original feature ------
        loss_layer4 = 0
        if hasattr(self, "feat_r2t4") and hasattr(self, "t4_orig"):
            loss_r2t4 = _dist_loss(self.feat_r2t4, self.t4_orig)
            loss_layer4 = loss_r2t4

        loss_layer3 = 0
        if hasattr(self, "feat_r2t3") and hasattr(self, "t3_orig"):
            loss_r2t3 = _dist_loss(self.feat_r2t3, self.t3_orig)
            loss_layer3 = loss_r2t3

        loss_layer2 = 0
        if hasattr(self, "feat_r2t2") and hasattr(self, "t2_orig"):
            loss_r2t2 = _dist_loss(self.feat_r2t2, self.t2_orig)
            loss_layer2 = loss_r2t2

        # ------ (3) Total consistency loss (optional: weighted average/sum) ------
        loss_consist = loss_layer4 + loss_layer3 + loss_layer2

        return lamda * loss_consist

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=2, resolution=None, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            if resolution is None:
                raise ValueError("The resolution parameter must be provided!")
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8, resolution=resolution)
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, fr, ft):
        output = fr
        p_saliency_final = None
        for layer in self.layers:
            output, p_saliency = layer(output, ft)
            p_saliency_final = p_saliency
        return output, p_saliency_final

class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, resolution,dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.resolution = resolution
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
        self.saliency_conv = nn.Conv2d(d_model, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fr, ft, pos=None, query_pos=None):
        fr2 = self.multihead_attn(
            query=self.with_pos_embed(fr, query_pos).transpose(0, 1),
            key=self.with_pos_embed(ft, pos).transpose(0, 1),
            value=ft.transpose(0, 1)
        )[0].transpose(0, 1)

        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)

        # FFN
        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)

        # ============ Calculate p_saliency ============
        B, L, C = fr.shape  # fr: (B, HW, d_model)
        # H = W = int(L ** 0.5)  # If it is not square, please modify it by yourself
        H, W = self.resolution
        assert H * W == L, f"Resolution mismatch: {H}x{W} should be {L}"

        fr_img = fr.transpose(1, 2).view(B, C, H, W)  # (B, d_model, H, W)
        p_sal_map = self.saliency_conv(fr_img)  # (B, 1, H, W)
        p_saliency = self.sigmoid(p_sal_map)  # (B, 1, H, W)
        # ==================================================

        fr_out = fr_img.view(B, C, H * W).transpose(1, 2)  # (B, HW, d_model)

        return fr_out, p_saliency
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class get_aligned_feat(nn.Module):
    def __init__(self, inC, outC):
        super(get_aligned_feat, self).__init__()

        # --- 1) Retain the original first three levels deformableConv ---
        self.deformConv1 = defomableConv(inC=inC*2, outC=outC)
        self.deformConv2 = defomableConv(inC=inC, outC=outC)
        self.deformConv3 = defomableConv(inC=inC, outC=outC)

        # === 2) Split the fourth-level deformConv into two sets for bidirectional sampling ===
        self.deformConv4_r2t = defomableConv_offset(inC=inC, outC=outC)
        self.deformConv4_t2r = defomableConv_offset(inC=inC, outC=outC)

        # === 3) Add a learnable fusion weight alpha ===
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # === (Optional) If you want to enhance the temperature of the TIR offset, you can add beta, thermal_mask, etc ===
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, fr, ft, thermal_mask=None):
        """
        fr: RGB (B, C, H, W)
        ft: TIR (B, C, H, W)
        thermal_mask: (Optional) Thermal radiation prior/high-temperature saliency map, used to enhance offset
        """

        cat_feat = torch.cat((fr, ft), dim=1)  # (B, 2*C, H, W)
        feat1 = self.deformConv1(cat_feat)
        feat2 = self.deformConv2(feat1)
        feat3 = self.deformConv3(feat2)

        aligned_feat_r2t = self.deformConv4_r2t(feat3, ft)

        aligned_feat_t2r = self.deformConv4_t2r(
            feta3 = feat3,
            x = fr,
            thermal_mask = thermal_mask,  # Pass in the high-temperature saliency map
            beta = self.beta
        )

        # (4) Bidirectional fusion: alpha is a learnable parameter
        F_final = self.alpha * aligned_feat_r2t + (1.0 - self.alpha) * aligned_feat_t2r

        # ========= Output individual features so that consistency loss can be calculated in TMSOD =============
        return F_final, aligned_feat_r2t, aligned_feat_t2r

class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out

class defomableConv_offset(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 2):
        super(defomableConv_offset, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, feta3, x, thermal_mask=None, beta=None):

        offset = self.offset(feta3)

        if (thermal_mask is not None) and (beta is not None):
            # (B, 1, H, W) -> (B, offset.shape[1], H, W)
            thermal_mask_expanded = thermal_mask.expand(-1, offset.shape[1], -1, -1)
            offset = offset * (1.0 + beta * thermal_mask_expanded)

        out = self.deform(x, offset)
        return out