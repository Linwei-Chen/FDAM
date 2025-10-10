# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from mmseg.ops import resize
from mmseg.models.builder import NECKS
from mmcv.ops.carafe import CARAFEPack

@NECKS.register_module()
class SimpleMultiLevelNeck(nn.Module):
    """MultiLevelNeck.
    https://github.com/facebookresearch/xcit/blob/main/semantic_segmentation/backbone/xcit.py
    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 patch_size=16,
                 carafe=False,
                 out_channels=None,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None):
        super(SimpleMultiLevelNeck, self).__init__()
        # assert isinstance(in_channels, list)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.scales = scales
        # self.num_outs = len(scales)
        # self.lateral_convs = nn.ModuleList()
        # self.convs = nn.ModuleList()
        # for in_channel in in_channels:
        #     self.lateral_convs.append(
        #         ConvModule(
        #             in_channel,
        #             out_channels,
        #             kernel_size=1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))
        # for _ in range(self.num_outs):
        #     self.convs.append(
        #         ConvModule(
        #             out_channels,
        #             out_channels,
        #             kernel_size=3,
        #             padding=1,
        #             stride=1,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                nn.SyncBatchNorm(in_channels),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ) if not carafe else CARAFEPack(channels=in_channels, up_group=1, up_kernel=5, encoder_kernel=3, scale_factor=4, compressed_channels= in_channels//8)

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ) if not carafe else CARAFEPack(channels=in_channels, up_group=1, up_kernel=5, encoder_kernel=3, scale_factor=2, compressed_channels= in_channels//8)

            # self.fpn1 = nn.Identity()
            # self.fpn2 = nn.Identity()
            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == 4
        outs = [
            self.fpn1(inputs[0]),
            self.fpn2(inputs[1]),
            self.fpn3(inputs[2]),
            self.fpn4(inputs[3]),
                ]
        # inputs = [
        #     lateral_conv(inputs[i])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        # # for len(inputs) not equal to self.num_outs
        # if len(inputs) == 1:
        #     inputs = [inputs[0] for _ in range(self.num_outs)]
        # outs = []
        # for i in range(self.num_outs):
        #     x_resize = resize(
        #         inputs[i], scale_factor=self.scales[i], mode='bilinear')
        #     outs.append(self.convs[i](x_resize))
        return tuple(outs)
