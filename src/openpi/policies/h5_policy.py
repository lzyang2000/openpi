import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Ensure uint8 HWC RGB."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 2:  # gray -> rgb
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = image[..., :3]
    return image


def _first(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the keys {keys} found in data.")


@dataclasses.dataclass(frozen=True)
class H5Inputs(transforms.DataTransformFn):
    """Map frames from your H5/LeRobot dataset to pi0/pi0-FAST input format.

    Expects:
      - state: 8D (x,y,z, qw,qx,qy,qz, gripper)
      - actions (optional during inference): 8D (x_ee_des + gripper)
      - image: third-person RGB
      - wrist_image: wrist RGB
      - task/prompt (optional): if present, forwarded as 'prompt'
    """

    model_type: _model.ModelType
    use_task_as_prompt: bool = True
    wrist_only: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(_first(data, "image", "observation/image"))
        wrist_image = _parse_image(_first(data, "wrist_image", "observation/wrist_image"))
        state = np.asarray(_first(data, "state", "observation/state"), dtype=np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image if not self.wrist_only else np.zeros_like(base_image),
                "left_wrist_0_rgb": wrist_image,
                # Pad missing right wrist with zeros (same shape as base image).
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_ if not self.wrist_only else np.False_,
                "left_wrist_0_rgb": np.True_,
                # Mask padded views for PI0 only (not PI0-FAST), per OpenPi convention.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are present during training.
        if ("actions" in data) or ("observation/actions" in data):
            inputs["actions"] = np.asarray(
                _first(data, "actions", "observation/actions"), dtype=np.float32
            )

        # Forward a prompt if available.
        prompt = data.get("prompt", None)
        if prompt is None and self.use_task_as_prompt:
            prompt = data.get("task", None)  # set in your converter as episode id
        if prompt is not None:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class H5Outputs(transforms.DataTransformFn):
    """Slice model outputs back to dataset action dimension (8)."""

    action_dim: int = 8  # (x,y,z, qw,qx,qy,qz, gripper)

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
