from typing import Any

import numpy as np


RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ = "robotwin_eef_se3_s0_wxyz"
RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ = (
    "robotwin_eef_se3_chunk_wxyz"
)
RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ = (
    "robotwin_eef_se3_s0_action_only_wxyz"
)
_ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES = (
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ,
)
_ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES = (
    *_ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
)

ROBOTWIN_EEF_DIM = 16
ROBOTWIN_EEF_ACTION_KEY = "action.eef_position"
ROBOTWIN_EEF_STATE_KEY = "state.eef_position"
RELATIVE_ACTION_POSE_REFERENCE_PER_ARM = "per_arm"
RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND = "left_hand"
_RELATIVE_ACTION_POSE_REFERENCES = (
    RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
)

_EPS = 1e-8
_ARM_STARTS = (0, 8)


def is_robotwin_eef_se3_s0_relative_action(mode: Any) -> bool:
    return mode in _ROBOTWIN_EEF_SE3_S0_RELATIVE_ACTION_MODES


def is_robotwin_eef_se3_relative_action(mode: Any) -> bool:
    return mode in _ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES


def is_robotwin_eef_se3_chunk_relative(mode: Any) -> bool:
    return mode == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ


def is_robotwin_eef_se3_s0_relative_state(mode: Any) -> bool:
    return mode == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ


def resolve_relative_action_pose_reference(
    value: Any,
    *,
    relative_action_mode: Any,
) -> str:
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
    if relative_action and not is_robotwin_eef_se3_relative_action(
        relative_action_mode
    ):
        raise ValueError(
            "relative_action=True requires "
            f"relative_action_mode in {_ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES!r}."
        )
    if not relative_action or not is_robotwin_eef_se3_relative_action(
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
    relative_action_pose_reference: Any = None,
    metadata_relative_action_mode: Any = None,
    metadata_relative_action_pose_reference: Any = None,
    metadata_path: Any = None,
) -> None:
    tag = str(embodiment_tag)
    mode = "" if relative_action_mode is None else str(relative_action_mode)
    metadata_mode = (
        None if metadata_relative_action_mode is None else str(metadata_relative_action_mode)
    )
    where = f" in {metadata_path}" if metadata_path is not None else ""
    supported_modes = _ROBOTWIN_EEF_SE3_RELATIVE_ACTION_MODES
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

    if tag != "robotwin":
        raise ValueError(
            "relative_action=True currently only supports embodiment_tag='robotwin'."
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
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat)
    identity[..., 0] = 1.0
    return np.where(norm > _EPS, quat / np.maximum(norm, _EPS), identity)


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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
    quat = _normalize_quat_wxyz(quat)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def _align_quat_to_reference_wxyz(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dot = np.sum(quat * reference, axis=-1, keepdims=True)
    return np.where(dot < 0.0, -quat, quat)


def robotwin_eef_abs_to_rel_wxyz(
    pose_abs: np.ndarray,
    state_anchor: np.ndarray,
    pose_reference: Any = RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
) -> np.ndarray:
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
