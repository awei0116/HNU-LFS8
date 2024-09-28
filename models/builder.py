import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

        # import backbone and decoder
        if cfg.backbone == 'sigma_tiny':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_vmamba import vssm_tiny as backbone
            from dataloader.LFimg_preprocessing import vssm_tiny_lf as VSS_backbone
            self.backbone = backbone()
            self.VSS_backbone = VSS_backbone()
        elif cfg.backbone == 'sigma_small':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [96, 192, 384, 768]
            from .encoders.dual_vmamba import vssm_small as backbone
            from dataloader.LFimg_preprocessing import vssm_small_lf as VSS_backbone
            self.backbone = backbone()
            self.VSS_backbone = VSS_backbone()
        elif cfg.backbone == 'sigma_base':
            logger.info('Using backbone: V-MAMBA')
            self.channels = [128, 256, 512, 1024]
            from .encoders.dual_vmamba import vssm_base as backbone
            self.backbone = backbone()
        else:
            logger.info('Using backbone: V-MAMBA')
            self.channels = [128, 256, 512, 1024]
            from .encoders.dual_vmamba import vssm_base as backbone
            self.backbone = backbone()

        self.aux_head = None

        if cfg.decoder == 'MambaDecoder':
            logger.info('Using Mamba Decoder')
            from .decoders.MambaDecoder import MambaDecoder
            self.deep_supervision = False
            self.decode_head = MambaDecoder(img_size=[cfg.image_height, cfg.image_width], in_channels=self.channels, num_classes=cfg.num_classes, embed_dim=self.channels[0], deep_supervision=self.deep_supervision)

        else:
            raise ValueError('Not a valid decoder name')

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            if cfg.backbone != 'vmamba':
                logger.info('Loading pretrained model: {}'.format(pretrained))
                self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, modal_x, lf):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        if not self.deep_supervision:
            orisize = next(iter(lf.values())).shape
            # orisize = lf[0].shape
            x = self.backbone(modal_x, lf)
            out = self.decode_head.forward(x)
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, aux_fm
            return out
        else:
            x = self.backbone(modal_x, lf)
            x_last, x_output_0, x_output_1, x_output_2 = self.decode_head.forward(x)
            return x_last, x_output_0, x_output_1, x_output_2

    def forward(self, modal_x, lf, label=None):

        if not self.deep_supervision:
            if self.aux_head:
                out, aux_fm = self.encode_decode(modal_x, lf)
            else:
                out = self.encode_decode(modal_x, lf)
            if label is not None:
                loss = self.criterion(out, label.long())
                if self.aux_head:
                    loss += self.aux_rate * self.criterion(aux_fm, label.long())
                return loss
            return out
        # else:
        #     x_last, x_output_0, x_output_1, x_output_2 = self.encode_decode(modal_x, lf)
        #     if label is not None:
        #         loss = self.criterion(x_last, label.long())
        #         loss += self.criterion(x_output_0, label.long())
        #         loss += self.criterion(x_output_1, label.long())
        #         loss += self.criterion(x_output_2, label.long())
        #         return loss
        #     return x_last
        #

    def flops(self, shape=(3, 480, 640)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        # 定义支持的操作
        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        # 定义输入
        input_tensor = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_dict = {
            'key1': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key2': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key3': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key4': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key5': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key6': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key7': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key8': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key9': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key10': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key11': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key12': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key13': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key14': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key15': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key16': torch.randn((1, *shape), device=next(model.parameters()).device),
            'key17': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key18': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key19': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key20': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key21': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key22': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key23': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key24': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key25': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key26': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key27': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key28': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key29': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key30': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key31': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key32': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 'key33': torch.randn((1, *shape), device=next(model.parameters()).device),
            # 添加其他需要的张量
        }

        # 将输入组合成一个元组
        input = (input_tensor, input_dict)

        # 打印输入信息
        print(len(input))
        for i in input:
            if isinstance(i, dict):
                for k, v in i.items():
                    print(f"{k}: {v.shape}")
            else:
                print(i.shape)

        # 计算参数和 GFLOPs
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops
  
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops