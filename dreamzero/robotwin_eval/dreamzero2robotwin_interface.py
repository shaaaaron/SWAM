"""DreamZero ↔ RoboTwin / RoboTwin2 interface.

Provides the three top-level functions required by RoboTwin's eval_policy.py:
    get_model(usr_args)   → DreamZeroClient
    eval(TASK_ENV, model, observation)
    reset_model(model)

The client connects to the DreamZero policy server via websocket (roboarena
protocol) and maintains a per-camera frame buffer so that the server receives
multi-frame video chunks matching DreamZero's training frame schedule:
    - First call:  1 frame   (H, W, 3) — frame 0 as first_image for CLIP + KV init
    - Subsequent:  8 frames  (8, H, W, 3) — block frames replacing predicted video in KV cache

Camera mapping (shared by v1 and v2):
    head_camera  → observation/wrist_image_left     → video.cam_high
    left_camera  → observation/exterior_image_0_left → video.cam_left_wrist
    right_camera → observation/exterior_image_1_left → video.cam_right_wrist

Action/state layout (switched by ``action_type``):
    action_type == "ee"   (RoboTwin v1):
        action_dim = 16 = [L_xyz(3)+L_quat(4)+L_grip(1)+R_xyz(3)+R_quat(4)+R_grip(1)]
        state       from observation.endpose.*  →  server key observation/eef_state
        TASK_ENV.take_action(action, action_type='ee')
    action_type == "qpos" (RoboTwin2):
        action_dim = 14 = [L_arm(6)+L_grip(1)+R_arm(6)+R_grip(1)]
        state       from observation.joint_action.vector  →  server key observation/joint_state
        TASK_ENV.take_action(action, action_type='qpos')
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame schedule constants (matching training: lerobot_sharded.py sampling)
# Training flow per chunk:
#   1) frame 0 sent at init → first_image (CLIP + VAE latent 0 for KV cache)
#   2) Model predicts actions [0..47] + video block frames [6, 12, ..., 48]
#   3) After executing 48 actions, real obs at [6, 12, ..., 48] replace
#      predicted video in KV cache (teacher forcing alignment)
# Server prepends first input frame for VAE 4k+1 requirement, but latent 0
# is DISCARDED — only last num_frame_per_block latents update KV cache.
# ---------------------------------------------------------------------------
FRAMES_PER_CHUNK = 8  # 8 block frames per chunk (excluding first_image sent at init)
MAX_BUFFER_SIZE = 400  # Maximum frames to keep in buffer per camera
# Relative offsets for subsampling 8 block frames from the buffer.
# At query time the latest frame (buf_len-1) is the CURRENT observation.
# We look backward at stride=6 for 8 block frames (matching training block
# frames [6, 12, ..., 48] relative to anchor).  buffer[0] (the episode's
# first observation) is ALWAYS prepended as first_image to form the full
# 8N+1 = 9 frame input that matches training's VAE encoding structure.
RELATIVE_OFFSETS = [-42, -36, -30, -24, -18, -12, -6, 0]
ACTION_HORIZON = 48


class DreamZeroClient:
    """Wrapper policy implementing roboarena BasePolicy interface for RoboTwin/RoboTwin2.

    Maintains a frame buffer per camera and follows the DreamZero frame schedule
    (1 frame initial, then FRAMES_PER_CHUNK=8 block frames per subsequent call).
    Caches action chunks (48 actions) and returns one action per step.

    Supports BOTH:
      - action_type="ee"   → RoboTwin v1 16D EEF: [L_xyz+L_quat+L_grip+R_xyz+R_quat+R_grip]
      - action_type="qpos" → RoboTwin2  14D joint: [L_arm(6)+L_grip(1)+R_arm(6)+R_grip(1)]
    The mode is selected at construction time (from deploy_policy.yml).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        action_dim: int = 14,
        action_chunk_size: int = 48,
        image_height: int = 128,
        image_width: int = 160,
        action_type: str = "qpos",
        use_obs_action_history: bool = False,
    ) -> None:
        # Lazy import to avoid import errors when dreamzero env is not active
        from eval_utils.policy_client import WebsocketClientPolicy

        self.host = host
        self.port = port
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.image_height = image_height
        self.image_width = image_width
        self.action_type = action_type      # "qpos" or "ee"
        self.use_obs_action_history = bool(use_obs_action_history and action_type == "ee")
        if use_obs_action_history and action_type != "ee":
            logger.warning("use_obs_action_history is only implemented for action_type='ee'; disabling.")

        # Connect to server
        logger.info(f"Connecting to DreamZero server at {host}:{port}...")
        self.client = WebsocketClientPolicy(host=host, port=port)

        # Validate server metadata
        metadata = self.client.get_server_metadata()
        logger.info(f"Server metadata: {metadata}")

        # Frame buffers per camera (RoboTwin camera name → frame deque)
        self._frame_buffers: dict[str, deque] = {
            "head_camera": deque(maxlen=MAX_BUFFER_SIZE),
            "left_camera": deque(maxlen=MAX_BUFFER_SIZE),
            "right_camera": deque(maxlen=MAX_BUFFER_SIZE),
        }

        # Action caching
        self.raw_actions: Optional[np.ndarray] = None
        self._action_step_offset: int = 0  # step index at which current chunk starts
        self._obs_action_history: list[np.ndarray] = []
        self._last_obs_action_step_idx: Optional[int] = None

        # Call counter
        self._call_count: int = 0
        self._is_first_call: bool = True

        # Session ID for the current episode
        self._session_id: str = str(uuid.uuid4())

        # Task description
        self._task_description: str = ""

        # Populated by step() for optional per-control-step execution tracing.
        self._last_action_before_shape_fix: Optional[np.ndarray] = None
        self._last_chunk_offset: Optional[int] = None
        self._last_query_id: Optional[int] = None
        self._last_server_queried: bool = False

    def _extract_eef_state(self, observation: dict) -> np.ndarray:
        """Extract RoboTwin v1 16D EEF state/action from obs.endpose."""
        endpose = observation["endpose"]
        left_endpose = endpose["left_endpose"]
        left_gripper = endpose["left_gripper"]
        right_endpose = endpose["right_endpose"]
        right_gripper = endpose["right_gripper"]
        eef_state = np.array(
            list(left_endpose) + [left_gripper]
            + list(right_endpose) + [right_gripper],
            dtype=np.float32,
        )
        if eef_state.shape != (16,):
            raise ValueError(
                "RoboTwin v1 EEF state must be 16D "
                "[L_xyz(3), L_quat(4), L_grip(1), R_xyz(3), R_quat(4), R_grip(1)], "
                f"got shape={eef_state.shape}."
            )
        return eef_state

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to the server's expected resolution (H, W)."""
        h, w = image.shape[:2]
        if h != self.image_height or w != self.image_width:
            image = cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return image

    # ------------------------------------------------------------------
    # Build roboarena observation dict
    # ------------------------------------------------------------------

    def _build_roboarena_obs(self, prompt: str) -> dict:
        """Build observation dict in roboarena format from frame buffers.

        First call: each image key is (H, W, 3) — frame 0 as first_image.
        Subsequent calls: each image key is (8, H, W, 3) — 8 block frames
            subsampled from the buffer using RELATIVE_OFFSETS at stride=6.
            These replace the model's predicted video in the KV cache.

        Frame selection (example at step 48, buf_len=49):
            RELATIVE_OFFSETS = [-42, -36, -30, -24, -18, -12, -6, 0]
            → [obs_6, obs_12, obs_18, obs_24, obs_30, obs_36, obs_42, obs_48]
            Server prepends first frame for VAE 4k+1, but only the last
            num_frame_per_block latents update KV cache (latent 0 discarded).
        """
        obs: dict = {}

        # Camera mapping: RoboTwin name → roboarena key
        camera_mapping = {
            "head_camera": "observation/wrist_image_left",
            "left_camera": "observation/exterior_image_0_left",
            "right_camera": "observation/exterior_image_1_left",
        }

        for robotwin_cam, roboarena_key in camera_mapping.items():
            buffer = self._frame_buffers[robotwin_cam]
            if len(buffer) == 0:
                logger.warning(f"Empty buffer for {robotwin_cam}, using zeros")
                frame = np.zeros(
                    (self.image_height, self.image_width, 3), dtype=np.uint8
                )
                if self._is_first_call:
                    obs[roboarena_key] = frame
                else:
                    obs[roboarena_key] = np.stack(
                        [frame] * FRAMES_PER_CHUNK, axis=0
                    )
                continue

            if self._is_first_call:
                # First call: single latest frame
                obs[roboarena_key] = buffer[-1]  # (H, W, 3)
            else:
                # Subsequent calls: 8 block frames replacing predicted video
                buf_len = len(buffer)
                frames = []
                for offset in RELATIVE_OFFSETS:
                    idx = buf_len - 1 + offset  # offset is negative or 0
                    idx = max(idx, 0)  # clamp to first frame if buffer too short
                    frames.append(buffer[idx])
                obs[roboarena_key] = np.stack(frames, axis=0)  # (8, H, W, 3)

        # State (will be set by caller)
        # Prompt
        obs["prompt"] = prompt
        obs["session_id"] = self._session_id

        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, observation: dict, step_idx: int) -> np.ndarray:
        """Process one RoboTwin step.

        Args:
            observation: RoboTwin observation dict with keys:
                - observation/head_camera/rgb: (H, W, 3)
                - observation/left_camera/rgb: (H, W, 3)
                - observation/right_camera/rgb: (H, W, 3)
                Plus (action_type-specific):
                  - action_type=="qpos": joint_action/vector: (14,)
                      [L_arm(6)+L_grip(1)+R_arm(6)+R_grip(1)]
                  - action_type=="ee":   endpose/{left_endpose, left_gripper,
                                                  right_endpose, right_gripper}
            step_idx: Current step index (TASK_ENV.take_action_cnt).

        Returns:
            action: (action_dim,) for RoboTwin (qpos mode=14, ee mode=16).
        """
        # 1. Extract and buffer images
        head_img = observation["observation"]["head_camera"]["rgb"]
        left_img = observation["observation"]["left_camera"]["rgb"]
        right_img = observation["observation"]["right_camera"]["rgb"]

        # Resize and add to buffers
        self._frame_buffers["head_camera"].append(self._resize_image(head_img))
        self._frame_buffers["left_camera"].append(self._resize_image(left_img))
        self._frame_buffers["right_camera"].append(self._resize_image(right_img))

        # 2. Extract state (dispatched by action_type)
        if self.action_type == "ee":
            # RoboTwin v1: 16D EEF from endpose
            # Layout: [L_xyz(3), L_quat(4), L_grip(1), R_xyz(3), R_quat(4), R_grip(1)]
            eef_state = self._extract_eef_state(observation)  # (16,)
            state_key = "observation/eef_state"
            state_vec = eef_state
        else:
            # RoboTwin2 qpos: 14D joint from joint_action.vector
            # Layout: [L_arm(6), L_grip(1), R_arm(6), R_grip(1)]
            joint_state_raw = observation.get("joint_action", {}).get("vector", None)
            if joint_state_raw is None:
                logger.warning("joint_action.vector missing, falling back to zeros")
                joint_state = np.zeros(14, dtype=np.float32)
            else:
                joint_state = np.asarray(joint_state_raw, dtype=np.float32).reshape(-1)
                if joint_state.shape[0] != 14:
                    logger.warning(
                        f"Expected 14D joint_state, got shape {joint_state.shape}; "
                        f"truncating/padding to 14."
                    )
                    if joint_state.shape[0] > 14:
                        joint_state = joint_state[:14]
                    else:
                        pad = np.zeros(14 - joint_state.shape[0], dtype=np.float32)
                        joint_state = np.concatenate([joint_state, pad], axis=0)
            state_key = "observation/joint_state"
            state_vec = joint_state

        if (
            self.use_obs_action_history
            and self.action_type == "ee"
            and step_idx > 0
            and step_idx != self._last_obs_action_step_idx
        ):
            self._obs_action_history.append(state_vec.astype(np.float32).copy())
            self._last_obs_action_step_idx = step_idx

        # 3. Decide whether to query the server
        # Query on the first call, or when the cached action chunk is exhausted
        need_query = False
        if self.raw_actions is None:
            need_query = True
        else:
            action_idx = step_idx - self._action_step_offset
            if action_idx >= len(self.raw_actions):
                need_query = True

        self._last_server_queried = need_query
        if need_query:
            # Build roboarena observation
            instruction = self._task_description
            obs = self._build_roboarena_obs(prompt=instruction)

            # Add state under the key expected by the server (ee_state or joint_state).
            obs[state_key] = state_vec
            if self.use_obs_action_history:
                if self._obs_action_history:
                    obs["observation/eef_action_history"] = np.stack(
                        self._obs_action_history,
                        axis=0,
                    ).astype(np.float32)
                else:
                    obs["observation/eef_action_history"] = np.zeros(
                        (0, 16),
                        dtype=np.float32,
                    )

            # Query server
            logger.info(
                f"Step {step_idx}: querying server (call_count={self._call_count}, "
                f"first_call={self._is_first_call})"
            )
            action_result = self.client.infer(obs)

            # Parse action result
            if isinstance(action_result, dict):
                # Server returns dict with action keys
                actions = None
                for key, value in action_result.items():
                    if isinstance(value, np.ndarray):
                        if actions is None or "joint" in key:
                            actions = value
                if actions is None:
                    actions = np.zeros((1, self.action_dim), dtype=np.float32)
            elif isinstance(action_result, np.ndarray):
                actions = action_result
            else:
                logger.warning(f"Unexpected action result type: {type(action_result)}")
                actions = np.zeros((1, self.action_dim), dtype=np.float32)

            # Ensure 2D
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            self.raw_actions = actions
            self._action_step_offset = step_idx
            self._call_count += 1

            # Update first call flag
            if self._is_first_call:
                self._is_first_call = False

        # 4. Get current action from cached chunk
        action_idx = step_idx - self._action_step_offset
        if action_idx >= len(self.raw_actions):
            # Repeat last action if we somehow overshoot
            action_idx = len(self.raw_actions) - 1
        current_action = self.raw_actions[action_idx]
        self._last_action_before_shape_fix = np.asarray(current_action).copy()
        self._last_chunk_offset = int(action_idx)
        self._last_query_id = int(self._call_count - 1)

        # 5. Ensure correct dimension (trim padding to native action_dim).
        # For qpos (14) or ee (16), RoboTwin's take_action strictly expects the native
        # dimensionality; padded entries (if model was trained with max_action_dim > native)
        # must be dropped here.
        if len(current_action) < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: len(current_action)] = current_action
            current_action = padded
        elif len(current_action) > self.action_dim:
            current_action = current_action[: self.action_dim]

        return current_action

    @staticmethod
    def _execution_trace_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): DreamZeroClient._execution_trace_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DreamZeroClient._execution_trace_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def write_execution_trace(
        self,
        task_env,
        observation: dict,
        executed_action: np.ndarray,
        step_idx: int,
    ) -> None:
        """Append the control-step action that was accepted by RoboTwin."""
        trace_path = os.environ.get("DREAMZERO_EXECUTION_TRACE_PATH")
        if not trace_path:
            return

        if self.action_type == "ee":
            observed_state = self._extract_eef_state(observation)
        else:
            observed_state = np.asarray(
                observation.get("joint_action", {}).get("vector", []),
                dtype=np.float32,
            ).reshape(-1)

        record = {
            "trace_kind": "control_step",
            "timestamp": time.time(),
            "session_id": self._session_id,
            "episode_index": int(getattr(task_env, "test_num", -1)),
            "step_idx": int(step_idx),
            "post_take_action_cnt": int(getattr(task_env, "take_action_cnt", -1)),
            "query_id": self._last_query_id,
            "chunk_offset": self._last_chunk_offset,
            "server_queried": self._last_server_queried,
            "instruction": str(task_env.get_instruction()),
            "action_type": self.action_type,
            "observed_state_before_action": observed_state,
            "action_before_shape_fix": self._last_action_before_shape_fix,
            "executed_action": np.asarray(executed_action).copy(),
        }

        trace_file = Path(trace_path)
        try:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with trace_file.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        default=self._execution_trace_value,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.warning("Failed to write execution trace %s: %r", trace_file, exc)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, task_description: str = "") -> None:
        """Reset client state for a new episode."""
        # Send reset to server
        try:
            self.client.reset({})
        except Exception as e:
            logger.warning(f"Failed to send reset to server: {e}")

        # Clear frame buffers
        for buf in self._frame_buffers.values():
            buf.clear()

        # Reset action cache
        self.raw_actions = None
        self._action_step_offset = 0
        self._obs_action_history = []
        self._last_obs_action_step_idx = None
        self._call_count = 0
        self._is_first_call = True

        # New session ID
        self._session_id = str(uuid.uuid4())
        self._task_description = task_description


# ======================================================================
# RoboTwin interface functions
# ======================================================================


def get_model(usr_args: dict) -> DreamZeroClient:
    """Create and return a DreamZeroClient.

    Called by RoboTwin's eval_policy.py.

    Args:
        usr_args: Config dict (from deploy_policy.yml + command line overrides).
            Required keys: host, port
            Optional keys: action_dim, action_chunk_size, image_height, image_width,
                           action_type, use_obs_action_history
    """
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 8000)
    action_dim = usr_args.get("action_dim", 14)
    action_chunk_size = usr_args.get("action_chunk_size", 48)
    image_height = usr_args.get("image_height", 128)
    image_width = usr_args.get("image_width", 160)
    action_type = usr_args.get("action_type", "qpos")
    use_obs_action_history = usr_args.get("use_obs_action_history", False)

    logger.info(
        f"Creating DreamZeroClient: host={host}, port={port}, "
        f"action_dim={action_dim}, chunk_size={action_chunk_size}, "
        f"image=({image_height}, {image_width}), "
        f"action_type={action_type}, "
        f"use_obs_action_history={use_obs_action_history}"
    )

    return DreamZeroClient(
        host=host,
        port=port,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        image_height=image_height,
        image_width=image_width,
        action_type=action_type,
        use_obs_action_history=use_obs_action_history,
    )


def reset_model(model: DreamZeroClient) -> None:
    """Reset the model for a new episode.

    Called by RoboTwin's eval_policy.py between episodes.
    """
    model.reset(task_description="")


def eval(TASK_ENV, model: DreamZeroClient, observation: dict) -> None:
    """Evaluate one step.

    Called by RoboTwin's eval_policy.py at each step.

    Args:
        TASK_ENV: RoboTwin task environment.
        model: DreamZeroClient instance.
        observation: RoboTwin observation dict.
    """
    # Get instruction from environment
    instruction = TASK_ENV.get_instruction()
    model._task_description = str(instruction)

    # Run one step
    step_idx = TASK_ENV.take_action_cnt
    action = model.step(observation, step_idx=step_idx)

    # Execute action in environment
    # "qpos": joint position control (14D direct)     — RoboTwin2 native mode
    # "ee":   end-effector pose control (16D: xyz+quat+grip per arm) — RoboTwin v1 mode
    TASK_ENV.take_action(action, action_type=model.action_type)
    model.write_execution_trace(TASK_ENV, observation, action, step_idx)
