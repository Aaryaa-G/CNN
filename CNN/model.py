"""
Small CNN for multi-object bounding-box regression from a (2, H, W)
event voxel grid. Kept deliberately compact (no pretrained backbone, no
BatchNorm-heavy design) because we only have 50 training samples -- a big
network would overfit instantly.

IMPORTANT PARAMETERIZATION NOTE:
Ship boxes are tiny relative to the frame (~1.6% of width, ~2.4% of height).
A naive head that sigmoids all 4 corner coordinates independently over the
FULL [0,1] range starts training with wildly oversized boxes (a random
x1,x2 pair drawn near the middle of [0,1] gives a box spanning ~25-50% of
the image -- 10-20x too big) and, empirically, struggles to ever recover
because IoU-based gradients vanish once predicted/target boxes barely
overlap. Instead this head predicts (center_x, center_y, width, height)
where width/height are sigmoid-bounded to a small MAX_SIZE_FRAC of the
frame, and the output bias is initialized from the dataset's actual mean
box per object so training starts near the right answer instead of at a
random, oversized guess.
"""
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


class ShipBoxNet(nn.Module):
    """
    Input:  (B, 2, H, W)  (native VISO resolution: H=451, W=1345)
    Output: (B, num_objects, 4)  -- normalized [x1,y1,x2,y2] in [0,1]

    Internally predicts (cx, cy, w, h) per object, with w/h capped at
    max_size_frac of the frame, then converts to corner format.
    """

    def __init__(self, num_objects: int = 3, base_ch: int = 16,
                 max_size_frac: float = 0.15, init_boxes_cxcywh=None):
        """
        init_boxes_cxcywh: optional array (num_objects, 4) of [cx, cy, w, h]
            in normalized [0,1] coords, e.g. the dataset's mean box per
            object. Used only to set the output layer's bias so training
            starts near the right answer instead of at a random guess.
        """
        super().__init__()
        self.num_objects = num_objects
        self.max_size_frac = max_size_frac

        self.stem = ConvBlock(2, base_ch, stride=2)
        self.block1 = ConvBlock(base_ch, base_ch * 2, 2)
        self.block2 = ConvBlock(base_ch * 2, base_ch * 4, 2)
        self.block3 = ConvBlock(base_ch * 4, base_ch * 8, 2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.out = nn.Linear(128, num_objects * 4)  # raw logits for [cx,cy,w,h] per object

        # Initialize the final layer's weights small and bias near the
        # dataset's true mean box, so sigmoid(bias) already points close to
        # the right answer at epoch 0 (critical with only 50 samples).
        nn.init.normal_(self.out.weight, std=1e-3)
        if init_boxes_cxcywh is not None:
            bias = torch.zeros(num_objects * 4)
            for oid in range(num_objects):
                cx, cy, w, h = init_boxes_cxcywh[oid]
                bias[oid * 4 + 0] = _logit(cx)
                bias[oid * 4 + 1] = _logit(cy)
                bias[oid * 4 + 2] = _logit(min(w / max_size_frac, 0.99))
                bias[oid * 4 + 3] = _logit(min(h / max_size_frac, 0.99))
            self.out.bias.data.copy_(bias)
        else:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.trunk(x)
        raw = self.out(x).view(-1, self.num_objects, 4)

        cx = torch.sigmoid(raw[..., 0])
        cy = torch.sigmoid(raw[..., 1])
        w = torch.sigmoid(raw[..., 2]) * self.max_size_frac
        h = torch.sigmoid(raw[..., 3]) * self.max_size_frac

        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)
