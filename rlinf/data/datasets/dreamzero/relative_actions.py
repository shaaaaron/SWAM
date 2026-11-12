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

from typing import Any

import numpy as np

RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ = "robotwin_eef_se3_s0_wxyz"
RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ = (
    "robotwin_eef_se3_chunk_wxyz"
)
RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ = (
    "robotwin_eef_se3_s0_action_only_wxyz"
)
RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ = (
    "noe_umi_eef_se3_s0_wxyz"
)
RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ = (
    "noe_umi_eef_se3_episode_wxyz"
)
RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ = (
    "noe_robotics_eef_se3_s0_wxyz"
)
_ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES = (
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ,
)
_ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES = (
    *_ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
)
_DUAL_ARM_EEF_SE3_RELATIVE_ACTION_MODES = (
    *_ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
)
_DUAL_ARM_EEF_SE3_RELATIVE_STATE_MODES = (
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
)
_RELATIVE_ACTION_MODE_TO_EMBODIMENT_TAG = {
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ: "robotwin",
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ: "robotwin",
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ: "robotwin",
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ: "noe_umi_16d",
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ: "noe_umi_16d",
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ: "noe_robotics_16d",
}

ROBOTWIN_EEF_DIM = 16
ROBOTWIN_EEF_ACTION_KEY = "action.eef_position"
ROBOTWIN_EEF_STATE_KEY = "state.eef_position"
RELATIVE_ACTION_POSE_REFERENCE_PER_ARM = "per_arm"
RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND = "left_hand"
_RELATIVE_ACTION_POSE_REFERENCES = (
    RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
)

ACTION_QUATERNION_MODE_PRESERVE = "preserve"
ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS = "temporal_continuous"
_ACTION_QUATERNION_MODES = (
    ACTION_QUATERNION_MODE_PRESERVE,
    ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
)

_EPS = 1e-8
_ARM_STARTS = (0, 8)


def is_robotwin_eef_se3_s0_relative_action(mode: Any) -> bool:
    """判断当前模式是否使用 RoboTwin EEF sample-s0 relative action。"""
    return mode in _ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES


def is_robotwin_eef_se3_relative_action(mode: Any) -> bool:
    """判断当前模式是否使用 RoboTwin EEF SE(3) relative action。"""
    return mode in _ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES


def is_robotwin_eef_se3_chunk_relative(mode: Any) -> bool:
    """判断当前模式是否让每个 action chunk 使用自己的 state anchor。"""
    return mode == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ


def is_robotwin_eef_se3_s0_relative_state(mode: Any) -> bool:
    """判断当前模式是否同时把 RoboTwin EEF state 转成 sample-s0 relative。"""
    return mode == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ


def is_dual_arm_eef_se3_relative_action(mode: Any) -> bool:
    """判断当前模式是否使用受支持的双臂 16D EEF relative action。"""
    return mode in _DUAL_ARM_EEF_SE3_RELATIVE_ACTION_MODES


def is_dual_arm_eef_se3_relative_state(mode: Any) -> bool:
    """判断当前模式是否同时转换双臂 16D EEF state。"""
    return mode in _DUAL_ARM_EEF_SE3_RELATIVE_STATE_MODES


def relative_action_mode_embodiment_tag(mode: Any) -> str | None:
    """返回 relative mode 严格对应的 embodiment tag；未知模式返回 None。"""
    return _RELATIVE_ACTION_MODE_TO_EMBODIMENT_TAG.get(mode)


def resolve_relative_action_pose_reference(
    value: Any,
    *,
    relative_action_mode: Any,
) -> str:
    """Resolve the dual-arm pose reference while preserving legacy behavior."""
    reference = (
        RELATIVE_ACTION_POSE_REFERENCE_PER_ARM
        if value in (None, "")
        else str(value)
    )
    if reference not in _RELATIVE_ACTION_POSE_REFERENCES:
        raise ValueError(
            "relative_action_pose_reference must be one of "
            f"{_RELATIVE_ACTION_POSE_REFERENCES!r}, got {reference!r}."
        )
    if (
        reference == RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND
        and not is_robotwin_eef_se3_relative_action(relative_action_mode)
    ):
        raise ValueError(
            "relative_action_pose_reference='left_hand' currently requires "
            "a RoboTwin EEF SE(3) relative action mode."
        )
    return reference


def ensure_relative_action_cache_compatible(
    *,
    relative_action: bool,
    relative_action_mode: Any,
    use_past_action_context: bool = False,
    use_obs_action_history: bool = False,
) -> None:
    """检查 SE(3) relative action 是否和 action cache 冲突；当前策略是同开即报错。"""
    if relative_action and not is_dual_arm_eef_se3_relative_action(
        relative_action_mode
    ):
        raise ValueError(
            "relative_action=True requires "
            "a supported dual-arm EEF relative action mode: "
            f"{_DUAL_ARM_EEF_SE3_RELATIVE_ACTION_MODES!r}."
        )
    if not relative_action or not is_dual_arm_eef_se3_relative_action(
        relative_action_mode
    ):
        return
    enabled = []
    if use_past_action_context:
        enabled.append("use_past_action_context")
    if use_obs_action_history:
        enabled.append("use_obs_action_history")
    if enabled:
        raise ValueError(
            "RoboTwin s0-relative action is intentionally "
            f"incompatible with action cache for now; disable {', '.join(enabled)}."
        )


def ensure_relative_action_metadata_compatible(
    *,
    embodiment_tag: Any,
    relative_action: bool,
    relative_action_mode: Any,
    action_quaternion_mode: Any = None,
    relative_action_pose_reference: Any = None,
    metadata_relative_action_mode: Any = None,
    metadata_action_quaternion_mode: Any = None,
    metadata_relative_action_pose_reference: Any = None,
    metadata_path: Any = None,
) -> None:
    """Fail fast when RoboTwin config semantics do not match metadata stats semantics."""
    tag = str(embodiment_tag)
    mode = "" if relative_action_mode is None else str(relative_action_mode)
    quaternion_mode = resolve_action_quaternion_mode(action_quaternion_mode)
    metadata_quaternion_mode = resolve_action_quaternion_mode(
        metadata_action_quaternion_mode
    )
    metadata_mode = (
        None if metadata_relative_action_mode is None else str(metadata_relative_action_mode)
    )
    where = f" in {metadata_path}" if metadata_path is not None else ""
    if quaternion_mode != metadata_quaternion_mode:
        raise ValueError(
            "Action quaternion mode mismatch: config uses "
            f"{quaternion_mode!r}, metadata uses {metadata_quaternion_mode!r}{where}."
        )
    supported_modes = _DUAL_ARM_EEF_SE3_RELATIVE_ACTION_MODES
    metadata_is_relative = metadata_mode in supported_modes
    config_is_relative = bool(relative_action)
    pose_reference = resolve_relative_action_pose_reference(
        relative_action_pose_reference,
        relative_action_mode=mode if config_is_relative else None,
    )

    if metadata_mode not in (None, "", *supported_modes):
        raise ValueError(
            f"Unsupported metadata relative_action_mode {metadata_mode!r}{where}; "
            f"supported modes: {supported_modes!r}."
        )

    if not config_is_relative:
        if metadata_is_relative:
            raise ValueError(
                "RoboTwin config has relative_action=False but metadata contains "
                f"relative_action_mode={metadata_mode!r}{where}. Use absolute metadata "
                "or enable matching relative_action config."
            )
        return

    expected_tag = relative_action_mode_embodiment_tag(mode)
    if tag != expected_tag:
        raise ValueError(
            f"relative_action_mode={mode!r} requires "
            f"embodiment_tag={expected_tag!r}, got {tag!r}."
        )
    if mode not in supported_modes:
        raise ValueError(
            "RoboTwin relative_action requires "
            f"relative_action_mode in {supported_modes!r}."
        )
    if metadata_mode != mode:
        raise ValueError(
            "RoboTwin relative_action=True requires metadata relative_action_mode="
            f"{mode!r}{where}, got {metadata_mode!r}."
        )
    metadata_pose_reference = resolve_relative_action_pose_reference(
        metadata_relative_action_pose_reference,
        relative_action_mode=metadata_mode,
    )
    if metadata_pose_reference != pose_reference:
        raise ValueError(
            "RoboTwin relative action pose-reference mismatch: config uses "
            f"{pose_reference!r}, metadata uses {metadata_pose_reference!r}{where}."
        )


def ensure_robotwin_eef_dim(name: str, value: Any) -> tuple[int, ...]:
    """约定 RoboTwin EEF component 必须是 16D；dataset/eval 维度不一致时统一报错。"""
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    if len(shape) == 0 or int(shape[-1]) != ROBOTWIN_EEF_DIM:
        raise ValueError(
            f"RoboTwin EEF {name} must be {ROBOTWIN_EEF_DIM}D, got shape={tuple(shape)}."
        )
    return tuple(int(dim) for dim in shape)


def ensure_robotwin_eef_same_ndim(
    action_name: str,
    action: Any,
    state_name: str,
    state_anchor: Any,
) -> None:
    """约定 action/state_anchor 只能是 2D 或 3D，且 ndim 必须一致。"""
    action_shape = ensure_robotwin_eef_dim(action_name, action)
    state_shape = ensure_robotwin_eef_dim(state_name, state_anchor)
    if len(action_shape) not in (2, 3):
        raise ValueError(
            "RoboTwin EEF action/state only support 2D or 3D tensors: "
            f"{action_name} shape={action_shape}, {state_name} shape={state_shape}."
        )
    if len(action_shape) != len(state_shape):
        raise ValueError(
            "RoboTwin EEF action/state ndim mismatch: "
            f"{action_name} shape={action_shape}, {state_name} shape={state_shape}. "
            "Expand state_anchor to match action ndim, e.g. [1,16] or [B,1,16]."
        )


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数；遇到零范数时回退为单位四元数，避免 NaN。"""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat)
    identity[..., 0] = 1.0
    return np.where(norm > _EPS, quat / np.maximum(norm, _EPS), identity)


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    """计算 wxyz 四元数共轭，用于从当前姿态反解相对旋转。"""
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """计算两个 wxyz 四元数乘积，表示旋转组合 lhs * rhs。"""
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    """把 wxyz 四元数转换成旋转矩阵，用于 xyz 在 state/action 坐标系间变换。"""
    quat = _normalize_quat_wxyz(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = np.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), axis=-1)
    row1 = np.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), axis=-1)
    row2 = np.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def _prepare_robotwin_eef_pair(
    action: np.ndarray,
    state_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """检查 RoboTwin EEF state/action 是否为 16D，并整理成可广播的形状。"""
    action_arr = np.asarray(action, dtype=np.float64)
    state_arr = np.asarray(state_anchor, dtype=np.float64)
    ensure_robotwin_eef_same_ndim("action", action_arr, "state anchor", state_arr)
    try:
        np.broadcast_shapes(action_arr.shape[:-1], state_arr.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "RoboTwin EEF action/state shapes are not broadcastable: "
            f"action={action_arr.shape}, state={state_arr.shape}"
        ) from exc
    return action_arr, state_arr


def _canonicalize_relative_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """归一化相对四元数，并固定到 w >= 0 的等价表示，减少符号跳变。"""
    quat = _normalize_quat_wxyz(quat)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def _align_quat_to_reference_wxyz(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把四元数符号对齐到参考四元数；若 dot < 0，则取等价的负号表示。"""
    dot = np.sum(quat * reference, axis=-1, keepdims=True)
    return np.where(dot < 0.0, -quat, quat)


def resolve_action_quaternion_mode(value: Any) -> str:
    """解析 action 四元数处理模式；缺省值保持旧行为。"""
    mode = ACTION_QUATERNION_MODE_PRESERVE if value in (None, "") else str(value)
    if mode not in _ACTION_QUATERNION_MODES:
        raise ValueError(
            "action_quaternion_mode must be one of "
            f"{_ACTION_QUATERNION_MODES!r}, got {mode!r}."
        )
    return mode


def temporal_continuous_robotwin_eef_action_wxyz(
    action: np.ndarray,
    state_anchor: np.ndarray,
    *,
    action_horizon: int,
) -> np.ndarray:
    """按 action chunk 对齐 RoboTwin absolute EEF 四元数符号。

    action 为 [T, 16]；state anchor 可为一行、每个 chunk 一行或每个 action
    一行。每个 action_horizon 帧重新开始，chunk 首帧与对应 state anchor
    对齐，后续帧与前一个已处理 action 对齐。函数只改变四元数的等价符号，
    位置和夹爪原样保留。
    """
    action_arr = np.asarray(action, dtype=np.float64)
    state_arr = np.asarray(state_anchor, dtype=np.float64)
    ensure_robotwin_eef_same_ndim("action", action_arr, "state anchor", state_arr)
    if action_arr.ndim != 2 or state_arr.ndim != 2:
        raise ValueError(
            "temporal_continuous_robotwin_eef_action_wxyz expects 2D action/state."
        )
    horizon = int(action_horizon)
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon!r}")
    num_chunks = (action_arr.shape[0] + horizon - 1) // horizon
    if state_arr.shape[0] not in (1, num_chunks, action_arr.shape[0]):
        raise ValueError(
            "state anchors must contain one row, one row per chunk, "
            "or one row per action; "
            f"got state={state_arr.shape}, action={action_arr.shape}."
        )

    out = action_arr.copy()
    for start in range(0, action_arr.shape[0], horizon):
        end = min(start + horizon, action_arr.shape[0])
        if state_arr.shape[0] == 1:
            anchor = state_arr[0]
        elif state_arr.shape[0] == num_chunks:
            anchor = state_arr[start // horizon]
        elif state_arr.shape[0] == action_arr.shape[0]:
            anchor = state_arr[start]
        for arm_start in _ARM_STARTS:
            quat_sl = slice(arm_start + 3, arm_start + 7)
            first = out[start, quat_sl]
            reference = anchor[quat_sl]
            if float(np.dot(first, reference)) < 0.0:
                out[start, quat_sl] = -first
            previous = out[start, quat_sl].copy()
            for frame in range(start + 1, end):
                current = out[frame, quat_sl]
                if float(np.dot(current, previous)) < 0.0:
                    out[frame, quat_sl] = -current
                previous = out[frame, quat_sl].copy()
    return out.astype(np.float32, copy=False)


def robotwin_eef_abs_to_rel_wxyz(
    pose_abs: np.ndarray,
    state_anchor: np.ndarray,
    pose_reference: Any = RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
) -> np.ndarray:
    """将 RoboTwin 16D EEF 绝对 pose/action 转成相对 state_anchor 的 SE(3) 表达。"""
    action_arr, state_arr = _prepare_robotwin_eef_pair(pose_abs, state_anchor)
    reference = resolve_relative_action_pose_reference(
        pose_reference,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    )
    out = action_arr.copy()

    for arm_start in _ARM_STARTS:
        pos_sl = slice(arm_start, arm_start + 3)
        quat_sl = slice(arm_start + 3, arm_start + 7)
        anchor_start = (
            0
            if reference == RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND
            else arm_start
        )
        anchor_pos_sl = slice(anchor_start, anchor_start + 3)
        anchor_quat_sl = slice(anchor_start + 3, anchor_start + 7)

        p_action = action_arr[..., pos_sl]
        p_state = state_arr[..., anchor_pos_sl]
        q_state = _normalize_quat_wxyz(state_arr[..., anchor_quat_sl])
        q_action = _normalize_quat_wxyz(action_arr[..., quat_sl])
        q_action = _align_quat_to_reference_wxyz(q_action, q_state)

        state_rot = _quat_to_matrix_wxyz(q_state)
        out[..., pos_sl] = np.einsum("...ji,...j->...i", state_rot, p_action - p_state)
        q_rel = _quat_mul_wxyz(_quat_conj_wxyz(q_state), q_action)
        out[..., quat_sl] = _canonicalize_relative_quat_wxyz(q_rel)

    return out.astype(np.float32, copy=False)


def robotwin_eef_rel_to_abs_wxyz(
    pose_rel: np.ndarray,
    state_anchor: np.ndarray,
    pose_reference: Any = RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
) -> np.ndarray:
    """将 RoboTwin 16D EEF SE(3) 相对 pose/action 还原成绝对表达。"""
    action_arr, state_arr = _prepare_robotwin_eef_pair(pose_rel, state_anchor)
    reference = resolve_relative_action_pose_reference(
        pose_reference,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    )
    out = action_arr.copy()

    for arm_start in _ARM_STARTS:
        pos_sl = slice(arm_start, arm_start + 3)
        quat_sl = slice(arm_start + 3, arm_start + 7)
        anchor_start = (
            0
            if reference == RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND
            else arm_start
        )
        anchor_pos_sl = slice(anchor_start, anchor_start + 3)
        anchor_quat_sl = slice(anchor_start + 3, anchor_start + 7)

        p_rel = action_arr[..., pos_sl]
        p_state = state_arr[..., anchor_pos_sl]
        q_state = _normalize_quat_wxyz(state_arr[..., anchor_quat_sl])
        q_rel = _canonicalize_relative_quat_wxyz(action_arr[..., quat_sl])

        state_rot = _quat_to_matrix_wxyz(q_state)
        out[..., pos_sl] = p_state + np.einsum("...ij,...j->...i", state_rot, p_rel)
        q_abs = _quat_mul_wxyz(q_state, q_rel)
        q_abs = _normalize_quat_wxyz(q_abs)
        out[..., quat_sl] = _align_quat_to_reference_wxyz(q_abs, q_state)

    return out.astype(np.float32, copy=False)
