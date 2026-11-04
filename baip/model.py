"""DenseNet-121 classifier and LAMB optimiser, as in Elsherif's notebook 02."""
import torch
import torch.nn as nn
import torchvision.models as models


def build_backbone(name, pretrained):
    name = name.lower()
    if name == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = net.classifier.in_features
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Identity()
        return net, in_f
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = net.classifier[1].in_features
    else:
        raise ValueError(f"unknown backbone {name}")
    net.classifier = nn.Identity()
    return net, in_f


class TextileCNN(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super().__init__()
        self.backbone, in_f = build_backbone(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class Lamb(torch.optim.Optimizer):
    """Elsherif's minimal LAMB (You et al., 2019): Adam moments, decoupled weight decay,
    per-tensor trust ratio. Same maths as his version; the trust ratio stays on the device
    instead of `.item()`, which avoids ~360 GPU syncs per step."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                mhat = st["m"] / (1 - b1 ** st["step"])
                vhat = st["v"] / (1 - b2 ** st["step"])
                upd = mhat / (vhat.sqrt() + grp["eps"])
                if grp["weight_decay"] != 0:
                    upd = upd.add(p, alpha=grp["weight_decay"])
                wn, un = p.norm(), upd.norm()
                trust = torch.where((wn > 0) & (un > 0), wn / un, torch.ones_like(wn))
                p.sub_(upd * (grp["lr"] * trust))
        return loss
