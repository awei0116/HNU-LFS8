import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, LFCrossMambaFusionBlock, LFConcatMambaFusionBlock, Backbone_VSSM_LF

logger = get_logger()


class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
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

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

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

        self.lf_cross_mamba = nn.ModuleList(
            LFCrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.lf_concat_mamba = nn.ModuleList(
            LFConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            self.absolute_pos_embed_lf = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                absolute_pos_embed_lf = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_lf, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)
                self.absolute_pos_embed_lf.append(absolute_pos_embed_lf)

    def forward_features(self, x_e, x_lf):
        """
        x_rgb: B x C x H x W
        """
        B = x_e.shape[0]
        outs_fused = []
        outs_x = self.vssm(x_e) # B x C x H x W
        outs_lf = self.vssm_lf(x_lf)  # B x C x H x W

        for i in range(4):
            if self.ape:
                # this has been discarded
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
                out_lf = self.absolute_pos_embed_lf[i].to(outs_lf[i].device) + outs_lf[i]
            else:
                out_x = outs_x[i]
                out_lf = outs_lf[i]
            # cross attention
            cma = True
            cam = True
            if cma and cam:
                cross_x, cross_lf = self.lf_cross_mamba[i](out_x.permute(0, 2, 3, 1).contiguous(),
                                                           out_lf.permute(0, 2, 3, 1).contiguous()) # B x H x W x C
                x_fuse = self.lf_concat_mamba[i](cross_x, cross_lf).permute(0, 3, 1, 2).contiguous()
            elif cam and not cma:
                x_fuse = self.lf_concat_mamba[i](out_x.permute(0, 2, 3, 1).contiguous(),
                                                 out_lf.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            elif not cam and not cma:
                x_fuse = (out_x + out_lf)
            outs_fused.append(x_fuse)        
        return outs_fused

    def forward(self, x_e, x_lfs):

        x_lfs_list = [x_lfs[key] for key in x_lfs]  # 获取所有的张量
        # 将所有张量沿着维度 dim=1 进行拼接
        x_lf = torch.cat(x_lfs_list, dim=1)

        out = self.forward_features(x_e, x_lf)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
