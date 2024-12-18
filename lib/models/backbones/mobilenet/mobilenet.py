from  torchvision import models
import sys
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy
from torchinfo import summary

model_urls = {
    'mobile_large': "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
    'mobile_small': "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
}

class MobileNet(nn.Module):
    def __init__(self, arch, pretrained_path):
        super(MobileNet, self).__init__()
        # Load MobileNetV2

        if arch == "mobile_large":
          mobilenet = models.mobilenet_v3_large()
          mobilenet.load_state_dict(model_zoo.load_url(model_urls[arch], pretrained_path))
          features = list(mobilenet.features.children())

          summary(mobilenet, input_size = (8, 3, 512, 1024))
          print(summary)

          # Divide MobileNet into stages
          self.stage1 = nn.Sequential(*features[:4])   # 192x192
          self.stage2 = nn.Sequential(*features[4:7]) # 96x96
          self.stage3 = nn.Sequential(*features[7:13])  # 48x48
          self.stage4 = nn.Sequential(*features[13:16])   # 24x24
        
        elif arch == "mobile_small":
          mobilenet = models.mobilenet_v3_small()
          mobilenet.load_state_dict(model_zoo.load_url(model_urls[arch], pretrained_path))
          features = list(mobilenet.features.children())



        # Define channel sizes for each stage
        # in_channels = [96, 320, 1280]

    def forward(self, x):
        f = []
        x = self.stage1(x)
        f.append(x)
        x = self.stage2(x)
        f.append(x)
        x = self.stage3(x)
        f.append(x)
        x = self.stage4(x)
        f.append(x)

        return f


class MobileBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):

        arch = self.configer.sub_arch

        if arch in [
            "mobile_large",
            "mobile_small",
        ]:

          arch_net = MobileNet(arch, self.configer.pretrained_backbone)
        
        else:
          raise Exception("Architecture undefined!")

        return arch_net


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, extra_convs_on_inputs=True, bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d(in_channels[i], out_channels, 1, bn=bn, bias=not bn, same_padding=True)
            fpn_conv = Conv2d(out_channels, out_channels, 3, bn=bn, bias=not bn, same_padding=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        # Build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=False)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x