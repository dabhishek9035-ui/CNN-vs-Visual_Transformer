"""ResNet-50 model builder."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(
    num_classes: int,
    pretrained: bool,
    weights_name: str | None = None,
) -> tuple[nn.Module, str | None]:
    weight = None
    if pretrained:
        enum_name = weights_name or "IMAGENET1K_V2"
        try:
            weight = ResNet50_Weights[enum_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported ResNet-50 weights: {enum_name}") from exc
    model = resnet50(weights=weight)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weight.name if weight is not None else None
