# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


logger = logging.getLogger(__name__)
TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "templates"
    / "robotwin_eval_conf_template.yaml"
)
RLINF_ROBOTWIN_VIDEO_ORDER = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]
DREAMZERO_ROBOTWIN_VIDEO_ORDER = [
    "video.cam_left_wrist",
    "video.cam_right_wrist",
    "video.cam_high",
]
DREAMZERO_DREAM_TRANSFORM_TARGET = (
    "groot.vla.model.dreamzero.transform.dreamzero_cotrain.DreamTransform"
)


def _to_plain_dict(cfg: Any) -> dict:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("DreamZero actor.model must resolve to a mapping.")
    return value


def _load_template() -> Any:
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"DreamZero eval template not found: {TEMPLATE_PATH}")
    return OmegaConf.load(TEMPLATE_PATH)


def _find_transform(transforms: Any, target: str) -> Any:
    for transform in transforms:
        if transform.get("_target_") == target:
            return transform
    raise KeyError(f"Transform not found in template: {target}")


def _assert_robotwin_template(conf: Any) -> None:
    robotwin = conf.transforms.robotwin
    transforms = list(robotwin.transforms)
    concat = _find_transform(transforms, "groot.vla.data.transform.ConcatTransform")
    dream = _find_transform(transforms, DREAMZERO_DREAM_TRANSFORM_TARGET)
    assert list(concat.video_concat_order) == DREAMZERO_ROBOTWIN_VIDEO_ORDER, (
        "DreamZero eval template must use RoboTwin order "
        "[video.cam_left_wrist, video.cam_right_wrist, video.cam_high]."
    )
    assert list(concat.video_concat_order) != RLINF_ROBOTWIN_VIDEO_ORDER, (
        "DreamZero eval template is using RLinf RoboTwin training video order."
    )
    assert dream.get("_target_") == DREAMZERO_DREAM_TRANSFORM_TARGET


def build_dreamzero_eval_conf(cfg: Any) -> Any:
    model_cfg = cfg.actor.model
    if str(model_cfg.get("model_type", "")) != "dreamzero":
        raise ValueError(
            "DreamZero eval config export requires actor.model.model_type=dreamzero."
        )

    embodiment_tag = str(model_cfg.get("embodiment_tag", ""))
    if embodiment_tag != "robotwin":
        raise NotImplementedError(
            f"DreamZero eval config export only supports robotwin, got {embodiment_tag!r}."
        )

    conf = _load_template()
    _assert_robotwin_template(conf)
    conf.model.config = OmegaConf.merge(
        OmegaConf.create(_to_plain_dict(model_cfg)),
        conf.model.config,
    )
    conf.relative_action = bool(model_cfg.get("relative_action", False))
    conf.relative_action_keys = list(model_cfg.get("relative_action_keys", []) or [])
    conf.relative_action_mode = model_cfg.get("relative_action_mode") or ""
    conf.relative_action_pose_reference = model_cfg.get(
        "relative_action_pose_reference", "per_arm"
    )
    return conf


def save_dreamzero_eval_experiment_cfg(cfg: Any, run_dir: str | Path) -> None:
    model_cfg = cfg.actor.model
    if str(model_cfg.get("model_type", "")) != "dreamzero":
        raise ValueError(
            "DreamZero eval config export requires actor.model.model_type=dreamzero."
        )

    metadata_path = model_cfg.get("metadata_json_path", None)
    if not metadata_path or not Path(str(metadata_path)).is_file():
        raise FileNotFoundError(
            f"DreamZero metadata_json_path not found: {metadata_path}"
        )

    exp_cfg_dir = Path(run_dir) / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(metadata_path), exp_cfg_dir / "metadata.json")

    embodiment_tag = str(model_cfg.get("embodiment_tag", ""))
    if embodiment_tag != "robotwin":
        logger.warning(
            "Skipping DreamZero eval conf export for unsupported tag %r; "
            "copied metadata.json only.",
            embodiment_tag,
        )
        return

    conf = build_dreamzero_eval_conf(cfg)
    OmegaConf.save(conf, exp_cfg_dir / "conf.yaml", resolve=True)
