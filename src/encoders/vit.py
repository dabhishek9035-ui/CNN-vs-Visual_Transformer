"""ViT-Base/16 model builder."""

from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b_16(
    num_classes: int,
    pretrained: bool,
    weights_name: str | None = None,
) -> tuple[nn.Module, str | None]:
    weight = None
    if pretrained:
        enum_name = weights_name or "IMAGENET1K_V1"
        try:
            weight = ViT_B_16_Weights[enum_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported ViT-B/16 weights: {enum_name}") from exc
    model = vit_b_16(weights=weight)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model, weight.name if weight is not None else None
