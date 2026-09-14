"""Compact CenterNet-style fully-convolutional ship detector.

Operates directly at stride 4 (two stride-2 conv blocks bring the 2-channel
event frame down to stride 4, then a configurable-depth stack of stride-1
conv blocks does the feature extraction) -- there is no encoder/decoder
upsampling path back to stride 32 and down again like a ResNet-backboned
CenterNet, because the inputs here are already small, sparse event frames
with small objects (ships are ~15-55px wide), so we don't need the extra
receptive field a deep backbone would cost us in resolution.

--width/--depth scale the model up for GPU training; defaults are sized for
a quick CPU smoke test.
"""
import torch
import torch.nn as nn


def conv_bn_act(in_ch, out_ch, stride=1, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = conv_bn_act(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return self.act(out + x)


class CenterNetLite(nn.Module):
    def __init__(self, in_ch=2, width=64, depth=6):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, width // 2, stride=2),
            conv_bn_act(width // 2, width, stride=2),
        )
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])

        self.heatmap_head = nn.Sequential(
            conv_bn_act(width, width),
            nn.Conv2d(width, 1, 1),
        )
        self.wh_head = nn.Sequential(
            conv_bn_act(width, width),
            nn.Conv2d(width, 2, 1),
        )
        self.offset_head = nn.Sequential(
            conv_bn_act(width, width),
            nn.Conv2d(width, 2, 1),
        )

        # CenterNet trick: initialize the heatmap head's final bias so the
        # network starts by predicting "background everywhere" (sigmoid ~0.01),
        # which keeps the focal loss well-behaved from epoch 0.
        nn.init.constant_(self.heatmap_head[-1].bias, -4.595)  # sigmoid(-4.595) ~= 0.01

    def forward(self, x):
        feat = self.body(self.stem(x))
        heatmap = torch.sigmoid(self.heatmap_head(feat))
        wh = torch.relu(self.wh_head(feat))  # box sizes are non-negative
        offset = self.offset_head(feat)
        return heatmap, wh, offset
