import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.net_utils import FeatureFusionModule as FFM
from models.net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import  Backbone_VSSM_LF
from einops import rearrange
from models.encoders.vmamba import CVSSDecoderBlock
import torch.utils.checkpoint as checkpoint

logger = get_logger()


class LFTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=None,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape

        self.vssm_lf = Backbone_VSSM_LF(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

    def forward_features(self, lf_fuse):

        outs_lf_fuse = self.vssm_lf(lf_fuse)  # B x C x H x W

        return outs_lf_fuse

    def forward(self, lf_fuse):
        out = self.forward_features(lf_fuse)
        return out


class vssm_tiny_lf(LFTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny_lf, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )


class vssm_small_lf(LFTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small_lf, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )


class vssm_base_lf(LFTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base_lf, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6,
            # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )


class MambaDecoder_LF(nn.Module):
    def __init__(self,
                 img_size=[480, 640],
                 in_channels=1024,
                 num_classes=40,
                 dropout_ratio=0.1,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Create only the last layer decoder
        self.layer_up1 = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1)),
                                 input_resolution=(
                                     self.patches_resolution[0] // (2 ** (self.num_layers - 1)),
                                     self.patches_resolution[1] // (2 ** (self.num_layers - 1))),
                                 depth=depths[-1],
                                 mlp_ratio=self.mlp_ratio,
                                 drop=drop_rate,
                                 attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:-1]):],
                                 norm_layer=norm_layer,
                                 upsample=PatchExpand if (self.num_layers > 1) else None,
                                 use_checkpoint=use_checkpoint)

        self.layer_up2 = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1)/2),
                                  input_resolution=(
                                      self.patches_resolution[0] // (2 ** (self.num_layers - 1)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1))),
                                  depth=depths[-1],
                                  mlp_ratio=self.mlp_ratio,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:-1]):],
                                  norm_layer=norm_layer,
                                  upsample=PatchExpand if (self.num_layers > 1) else None,
                                  use_checkpoint=use_checkpoint)

        self.layer_up3 = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1)/4),
                                  input_resolution=(
                                      self.patches_resolution[0] // (2 ** (self.num_layers - 1)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1))),
                                  depth=depths[-1],
                                  mlp_ratio=self.mlp_ratio,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:-1]):],
                                  norm_layer=norm_layer,
                                  upsample=PatchExpand if (self.num_layers > 1) else None,
                                  use_checkpoint=use_checkpoint)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
        # Optionally handle deep supervision layers here

            self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                   patch_size=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, bias=False)

        self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                   patch_size=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward_up_features(self, lf_vss):  # B, C, H, W

        x = lf_vss.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        # y = self.layer_up1(x)  # Perform decoding for the last layer
        # y = self.layer_up2(x)
        y = self.layer_up3(x)
        x = self.norm_up(y)
        return x

    def forward(self, lf_vss):
        x = self.forward_up_features(lf_vss)
        x_last = self.up_x4(x, self.patch_size)
        return x_last

    def up_x4(self, x, pz):
        B, H, W, C = x.shape

        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, 4H, 4W
        x = self.output(x)

        return x

class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 2H, 2W, C
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()  # B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 4H, 4W, C
        x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """

        x = self.expand(x)  # B, H, W, 2C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x

class Mamba_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, dt_rank="auto",
                 d_state=4, ssm_ratio=2.0, attn_drop_rate=0.,
                 drop_rate=0.0, mlp_ratio=4.0,
                 drop_path=0.1, norm_layer=nn.LayerNorm, upsample=None,
                 shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, **kwargs):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.upsample = UpsampleExpand(input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class UpsampleExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear(x).permute(0, 3, 1, 2).contiguous()  # B, C/2, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 2H, 2W, C/2
        x = self.norm(x)
        return x
