import torch
from torch import nn
from torch.nn import functional as F

# ... (your existing imports)

class FERAttentionNetWithAttentionModule(nn.Module):
    """FERAttentionNet with attention module
    """

    def __init__(self, dim=32, num_classes=1, num_channels=3, backbone='preactresnet', num_filters=32):
        super().__init__()

        # Attention module
        self.attention_map = AttentionResNet(in_channels=num_channels, out_channels=num_classes, pretrained=True)

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4,
                                   bias=True)
        self.feature = self.make_layer(_Residual_Block_SR, 8, num_classes)
        self.conv_mid = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1,
                                  bias=True)

        # Reconstruction module
        self.reconstruction = nn.Sequential(
            ConvRelu(2 * num_classes + num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)

        # Select backbone classification
        self.backbone = backbone
        if self.backbone == 'preactresnet':
            self.netclass = preactresnet.preactresnet18(num_classes=num_classes, num_channels=num_channels)
        elif self.backbone == 'inception':
            self.netclass = inception.inception_v3(num_classes=num_classes, num_channels=num_channels,
                                                    transform_input=False, pretrained=True)
        elif self.backbone == 'resnet':
            self.netclass = resnet.resnet18(num_classes=num_classes, num_channels=num_channels)
        elif self.backbone == 'cvgg':
            self.netclass = cvgg.cvgg13(num_classes=num_classes, num_channels=num_channels)
        else:
            assert (False)

    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)

    def forward(self, x, x_org=None):

        # Attention map
        g_att = self.attention_map(x)

        # Feature module
        out = self.conv_input(x)
        residual = out
        out = self.feature(out)
        out = self.conv_mid(out)
        g_ft = torch.add(out, residual)

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul(torch.sigmoid(g_att), g_ft)
        att = self.reconstruction(torch.cat((attmap, x, g_att), dim=1))
        att = F.relu(self.conv2_bn(att))
        att_out = normalize_layer(att)

        # Select backbone classification
        if self.backbone == 'preactresnet':
            att_pool = F.avg_pool2d(att_out, 2)
        elif self.backbone == 'inception':
            att_pool = F.interpolate(att_out, size=(299, 299), mode='bilinear', align_corners=False)
        elif self.backbone == 'resnet':
            att_pool = F.interpolate(att_out, size=(224, 224), mode='bilinear', align_corners=False)
        elif self.backbone == 'cvgg':
            att_pool = att_out
        else:
            assert (False)

        y = self.netclass(att_pool)

        return y, att, g_att, g_ft
