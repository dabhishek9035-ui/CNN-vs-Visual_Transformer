"""Torchvision model builders used by the CNN-versus-ViT benchmark."""

from torch import nn

from src.encoders.resnet import build_resnet50
from src.encoders.vit import build_vit_b_16


MODEL_BUILDERS = {
    "resnet50": build_resnet50,
    "vit_b_16": build_vit_b_16,
}


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    weights_name: str | None = None,
) -> tuple[nn.Module, str | None]:
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_BUILDERS)}") from exc
    return builder(num_classes=num_classes, pretrained=pretrained, weights_name=weights_name)


def get_classifier_head(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet50":
        return model.fc
    if model_name == "vit_b_16":
        return model.heads
    raise ValueError(f"Unknown model {model_name!r}")


def configure_trainable_parameters(model: nn.Module, model_name: str, strategy: str) -> None:
    if strategy not in {"scratch", "frozen", "finetune"}:
        raise ValueError("strategy must be one of: scratch, frozen, finetune")
    for parameter in model.parameters():
        parameter.requires_grad = strategy != "frozen"
    if strategy == "frozen":
        for parameter in get_classifier_head(model, model_name).parameters():
            parameter.requires_grad = True
