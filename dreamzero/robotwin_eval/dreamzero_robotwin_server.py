"""DreamZero policy server for RoboTwin / RoboTwin2 dual-arm evaluation.

Supports BOTH embodiments via ``embodiment_tag`` dispatch:
  - embodiment_tag="robotwin"  (v1, EEF / 16D):
      state key:  observation/eef_state  →  state.eef_position (1, 16)
      action key: action.eef_position    ←  (N, 16) model output
  - embodiment_tag="robotwin2" (v2, qpos / 14D):
      state key:  observation/joint_state →  4 sub-keys
          state.left_arm(6)/left_gripper(1)/right_arm(6)/right_gripper(1)
      action:     4 sub-keys concatenated ← (N, 14)
          action.left_arm/left_gripper/right_arm/right_gripper

Shared by both:
  - Camera mapping: head→cam_high, left→cam_left_wrist, right→cam_right_wrist
  - Image resolution handled by transform pipeline (raw → 128×160 or 180×320)
  - 8 block frames per chunk (first_image at init), action horizon 48
  - roboarena websocket protocol (eval_utils/policy_server.py)

Usage:
    torchrun --nproc_per_node=2 dreamzero_robotwin_server.py --port 8000 \\
        --model_path /path/to/checkpoint --config deploy_policy.yml
"""

import asyncio
import dataclasses
import datetime
import http
import importlib
import json
import logging
import os
import pickle
import socket
import time
import traceback

import imageio
import numpy as np
import torch
import torch.distributed as dist
import tyro
import websockets.asyncio.server as _server
import websockets.frames
from einops import rearrange
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

# Use roboarena policy server interface
from eval_utils.policy_server import PolicyServerConfig
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # ~14 hours default
    model_path: str = "./checkpoints/dreamzero"
    index: int = 0
    # Explicit root for WAN prediction videos. When omitted, keep the legacy
    # model-relative output layout for backward compatibility.
    video_output_root: str | None = None
    max_chunk_size: int | None = None
    # Embodiment tag for the model (must match training metadata.json key).
    embodiment_tag: str = "robotwin2"
    # Action dimension for the environment (RoboTwin2 native joint = 14).
    action_dim: int = 14
    # Input state dimensions expected by the checkpoint's normalizer.
    # RoboTwin2: joint=14 (includes grippers), gripper=0 (no separate concat).
    state_joint_dim: int = 14
    state_gripper_dim: int = 0
    # If true for robotwin/ee, use obs.endpose history as external past action context.
    use_obs_action_history: bool = False
    # Optional eval-time scheduler overrides for checkpoint ablations.
    action_train_sigma_shift: float | None = None
    action_inference_sigma_shift: float | None = None
    # Path to YAML config (e.g., deploy_policy.yml). Values in the YAML
    # override the defaults above; CLI args override everything.
    config: str | None = None


class RoboTwinPolicy:
    """Wrapper policy implementing roboarena BasePolicy for RoboTwin / RoboTwin2 dual-arm.

    Handles:
    - Observation format conversion (roboarena → DreamZero internal format)
    - Frame accumulation across calls (roboarena sends per-step, DreamZero expects multi-frame)
    - State conversion (dispatched by embodiment_tag):
        robotwin  (v1): observation/eef_state → state.eef_position (1, 16)
        robotwin2 (v2): observation/joint_state → 4 sub-keys (left_arm/left_gripper/right_arm/right_gripper)
    - Action conversion: prefers v2 joint sub-keys, falls back to v1 eef_position, then DROID
    - Distributed inference coordination (broadcast obs to worker ranks)
    """

    # Number of frames to accumulate per chunk after the first call
    # 8 block frames replacing model's predicted video in KV cache
    FRAMES_PER_CHUNK = 8

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        action_dim: int = 14,
        state_joint_dim: int = 14,
        state_gripper_dim: int = 0,
        output_dir: str | None = None,
        embodiment_tag: str = "robotwin2",
        use_obs_action_history: bool = False,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._action_dim = action_dim
        self._state_joint_dim = state_joint_dim
        self._state_gripper_dim = state_gripper_dim
        self._output_dir = output_dir
        self._embodiment_tag = embodiment_tag
        self._use_obs_action_history = bool(use_obs_action_history and embodiment_tag == "robotwin")
        if use_obs_action_history and embodiment_tag != "robotwin":
            logger.warning("use_obs_action_history is only implemented for embodiment_tag='robotwin'; disabling.")

        # Frame buffers for accumulation (per camera view)
        # Keys must match modality_config_robotwin*/video.modality_keys in training YAML
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.cam_high": [],
            "video.cam_left_wrist": [],
            "video.cam_right_wrist": [],
        }
        self._call_count = 0
        self._is_first_call = True

        # Session tracking
        self._current_session_id: str | None = None

        # Video predictions for potential saving
        self.video_across_time = []
        self._msg_index = 0
        self._obs_action_history_logged = False

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation to DreamZero internal format.

        Roboarena format (from dreamzero2robotwin_interface.py):
            - observation/exterior_image_0_left: (H, W, 3) or (T, H, W, 3)  ← left_camera
            - observation/exterior_image_1_left: (H, W, 3) or (T, H, W, 3)  ← right_camera
            - observation/wrist_image_left:      (H, W, 3) or (T, H, W, 3)  ← head_camera
            - observation/joint_state: (14,) — already in training layout
              [L_arm(6), L_grip(1), R_arm(6), R_grip(1)]
            - prompt: str

        DreamZero internal format (must match modality_config_robotwin2 in training YAML):
            - video.cam_high:        (T, H, W, 3)
            - video.cam_left_wrist:  (T, H, W, 3)
            - video.cam_right_wrist: (T, H, W, 3)
            - state.left_arm:        (1, 6)
            - state.left_gripper:    (1, 1)
            - state.right_arm:       (1, 6)
            - state.right_gripper:   (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}

        # Map image keys: roboarena → training modality_config_robotwin2 keys
        # RoboTwin cameras: head_camera→cam_high, left→cam_left_wrist, right→cam_right_wrist
        image_key_mapping = {
            "observation/wrist_image_left":      "video.cam_high",
            "observation/exterior_image_0_left": "video.cam_left_wrist",
            "observation/exterior_image_1_left": "video.cam_right_wrist",
        }

        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            num_frames = 1
        else:
            num_frames = self.FRAMES_PER_CHUNK

        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating first frame
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video

        # Convert state (dispatched by embodiment_tag):
        #   robotwin  (v1): observation/eef_state (16,) → state.eef_position (1, 16)
        #   robotwin2 (v2): observation/joint_state (14,) → 4 sub-keys
        if self._embodiment_tag == "robotwin":
            # v1 EEF: [L_xyz(3), L_quat(4), L_grip(1), R_xyz(3), R_quat(4), R_grip(1)] = 16
            eef_state = obs["observation/eef_state"]
            eef_state = np.asarray(eef_state, dtype=np.float32).reshape(-1)
            if eef_state.shape[0] != 16:
                raise ValueError(
                    "RoboTwin v1 observation/eef_state must be 16D "
                    "[L_xyz(3), L_quat(4), L_grip(1), R_xyz(3), R_quat(4), R_grip(1)], "
                    f"got shape={eef_state.shape}."
                )
            converted["state.eef_position"] = eef_state.astype(np.float64).reshape(1, 16)
            if self._use_obs_action_history and "observation/eef_action_history" in obs:
                eef_history = np.asarray(
                    obs["observation/eef_action_history"],
                    dtype=np.float32,
                )
                if eef_history.size == 0:
                    eef_history = np.zeros((0, 16), dtype=np.float32)
                else:
                    if eef_history.ndim == 1:
                        eef_history = eef_history.reshape(1, -1)
                    elif eef_history.ndim > 2:
                        eef_history = eef_history.reshape(-1, eef_history.shape[-1])
                    if eef_history.shape[-1] != 16:
                        # 历史兼容：obs action history 必须是 16D EEF，当前先截断/补零避免旧请求中断。
                        logger.warning(
                            f"eef_action_history has shape {eef_history.shape}, expected (*, 16); "
                            f"trimming/padding."
                        )
                        if eef_history.shape[-1] > 16:
                            eef_history = eef_history[:, :16]
                        else:
                            pad = np.zeros(
                                (eef_history.shape[0], 16 - eef_history.shape[-1]),
                                dtype=np.float32,
                            )
                            eef_history = np.concatenate([eef_history, pad], axis=-1)
                converted["external_past_action.eef_position"] = eef_history.astype(np.float32)
                if not self._obs_action_history_logged:
                    logger.info(
                        "Using obs EEF action history: external_past_action.eef_position=%s",
                        converted["external_past_action.eef_position"].shape,
                    )
                    self._obs_action_history_logged = True
        else:
            # v2 qpos: split 14D joint vector into 4 training sub-keys
            # Layout: [L_arm(0:6) | L_grip(6:7) | R_arm(7:13) | R_grip(13:14)]
            joint_state = obs.get("observation/joint_state", np.zeros(14, dtype=np.float32))
            joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
            if joint_state.shape[0] != 14:
                logger.warning(
                    f"joint_state has shape {joint_state.shape}, expected (14,); "
                    f"trimming/padding."
                )
                if joint_state.shape[0] > 14:
                    joint_state = joint_state[:14]
                else:
                    pad = np.zeros(14 - joint_state.shape[0], dtype=np.float32)
                    joint_state = np.concatenate([joint_state, pad], axis=0)
            # Each sub-key is shape (1, D) to match the (time=1, dim) convention.
            converted["state.left_arm"]      = joint_state[0:6].astype(np.float64).reshape(1, 6)
            converted["state.left_gripper"]  = joint_state[6:7].astype(np.float64).reshape(1, 1)
            converted["state.right_arm"]     = joint_state[7:13].astype(np.float64).reshape(1, 6)
            converted["state.right_gripper"] = joint_state[13:14].astype(np.float64).reshape(1, 1)

        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""
        if "session_id" in obs:
            converted["session_id"] = obs["session_id"]

        return converted

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert DreamZero action dict to roboarena action array.

        DreamZero output for RoboTwin2 (after unapply/denormalization), one tensor per sub-key:
            - action.left_arm:      (N, 6)
            - action.left_gripper:  (N, 1)
            - action.right_arm:     (N, 6)
            - action.right_gripper: (N, 1)
        Concatenation order: [L_arm | L_grip | R_arm | R_grip]  → (N, 14)

        For backward compatibility we also handle:
            - action.eef_position: (N, 16)  — legacy v1 RoboTwin EEF
            - action.joint_position + action.gripper_position  — legacy DROID

        Returns:
            action: (N, action_dim) numpy array (action_dim = 14 for RoboTwin2).
        """

        def _to_np(x) -> np.ndarray:
            """Default helper: 1D → (1, D) row (legacy eef/joint flattened to single step)."""
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return x

        # Strategy 1 (RoboTwin2 native): concatenate the 4 joint sub-keys
        # Model outputs shape (T=action_horizon, D_subkey). Gripper may come back
        # 1D as (T,) instead of (T, 1); disambiguate using each sub-key's expected D.
        subkey_order = [
            "action.left_arm",
            "action.left_gripper",
            "action.right_arm",
            "action.right_gripper",
        ]
        expected_subkey_dim = {
            "action.left_arm": 6,
            "action.left_gripper": 1,
            "action.right_arm": 6,
            "action.right_gripper": 1,
        }
        if all(k in action_dict and action_dict[k] is not None for k in subkey_order):
            parts = []
            for k in subkey_order:
                x = action_dict[k]
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                x = np.asarray(x, dtype=np.float32)
                D = expected_subkey_dim[k]
                if x.ndim == 1:
                    # 1D is ambiguous between (T,) and (D,); disambiguate via D.
                    if D == 1:
                        x = x.reshape(-1, 1)   # (T,) → (T, 1): gripper time series
                    else:
                        x = x.reshape(1, -1)   # (D,) → (1, D): single timestep
                parts.append(x)
            action = np.concatenate(parts, axis=-1)  # (T, 14)
        else:
            # Strategy 2 (legacy): find eef_position / joint_position + gripper
            action_tensor = None
            joint_action = None
            gripper_action = None
            for key, value in action_dict.items():
                if "eef_position" in key:
                    action_tensor = value
                    break
                elif "joint_position" in key:
                    joint_action = value
                elif "gripper_position" in key or "gripper" in key:
                    gripper_action = value

            if action_tensor is not None:
                action = _to_np(action_tensor)
            elif joint_action is not None:
                joint_np = _to_np(joint_action)
                if gripper_action is not None:
                    grip_np = _to_np(gripper_action)
                    if grip_np.ndim == 1:
                        grip_np = grip_np.reshape(-1, 1)
                    action = np.concatenate([joint_np, grip_np], axis=-1)
                else:
                    action = joint_np
            else:
                logger.error(
                    f"No recognized action key found! Available keys: {list(action_dict.keys())}"
                )
                return np.zeros((1, self._action_dim), dtype=np.float32)

        action = action.astype(np.float32)

        # Ensure correct action dimension
        if action.shape[-1] > self._action_dim:
            action = action[:, : self._action_dim]

        return action

    # ------------------------------------------------------------------
    # Distributed broadcast
    # ------------------------------------------------------------------

    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        size_tensor = torch.tensor([data_size], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    # ------------------------------------------------------------------
    # Inference (roboarena interface)
    # ------------------------------------------------------------------

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.

        Args:
            obs: Observation dict in roboarena format.

        Returns:
            action: (N, action_dim) action array.
        """
        # Check for session change
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(
                    f"Session changed: '{self._current_session_id}' → '{session_id}', resetting"
                )
                self._reset_state()
            else:
                logger.info(f"New session: '{session_id}'")
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        # Convert observation format
        converted_obs = self._convert_observation(obs)

        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast obs to workers
        self._broadcast_batch_to_workers(converted_obs)

        # Create batch for policy
        batch = Batch(obs=converted_obs)

        # Distributed forward pass
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # Store video predictions (saved on reset after full trajectory)
        video_pred_to_save = self._video_latents_for_episode(video_pred)
        self.video_across_time.append(video_pred_to_save)

        # Extract and convert action
        action_chunk_dict = result_batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)

        action = self._convert_action(action_dict)

        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False

        return action

    # ------------------------------------------------------------------
    # Video saving helper
    # ------------------------------------------------------------------

    def _video_latents_for_episode(self, video_pred: torch.Tensor) -> torch.Tensor:
        """Remove a cache-reset reference latent from the saved video stream."""
        predicted_latents = (
            self._policy.trained_model.action_head.num_frame_per_block
        )
        num_latents = video_pred.shape[2]

        if self._is_first_call:
            if num_latents != predicted_latents + 1:
                logger.warning(
                    "Expected %d video latents on the first call, got %d; "
                    "keeping all latents",
                    predicted_latents + 1,
                    num_latents,
                )
            return video_pred

        if num_latents == predicted_latents:
            return video_pred

        if num_latents == predicted_latents + 1:
            logger.debug("Dropping mid-episode cache-reset reference latent")
            return video_pred[:, :, 1:].contiguous()

        logger.warning(
            "Expected %d or %d video latents after the first call, got %d; "
            "keeping all latents",
            predicted_latents,
            predicted_latents + 1,
            num_latents,
        )
        return video_pred

    def _decode_and_save_video(self, video_latents: list[torch.Tensor], tag: str = "") -> None:
        """Decode VAE latents and save as MP4.

        Args:
            video_latents: List of video prediction tensors to concatenate and decode.
            tag: Optional tag for the filename.
        """
        if not self._output_dir or len(video_latents) == 0:
            return
        try:
            video_across_time_cat = torch.cat(video_latents, dim=2)
            ah = self._policy.trained_model.action_head
            frames = ah.vae.decode(
                video_across_time_cat,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = frames[0]
            frames = (
                ((frames.float() + 1) * 127.5)
                .clip(0, 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            frame_list = [f for f in frames]

            if len(frame_list) == 0:
                return
            sample_frame = frame_list[0]
            if len(sample_frame.shape) != 3 or sample_frame.shape[2] not in [1, 3, 4]:
                logger.warning(
                    f"Invalid frame shape {sample_frame.shape}, skipping video save"
                )
                return

            save_dir = self._output_dir
            os.makedirs(save_dir, exist_ok=True)
            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            num_frames = len(frame_list)
            n = (num_frames - 1) // 8  # num_frames = 8n+1
            suffix = f"_{tag}" if tag else ""
            output_path = os.path.join(
                save_dir,
                f"{len(all_mp4_files):06}_{timestamp}_n{n}{suffix}.mp4",
            )
            imageio.mimsave(output_path, frame_list, fps=5, codec="libx264")
            logger.info(f"Saved predicted video to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_state(self, save_video: bool = True) -> None:
        """Reset policy state, optionally saving accumulated video."""
        if save_video and len(self.video_across_time) > 0:
            self._decode_and_save_video(self.video_across_time, tag="reset")

        # Clear frame buffers
        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode."""
        self._reset_state(save_video=True)


# ======================================================================
# Worker loop for non-rank-0 processes (from socket_test_optimized_AR.py)
# ======================================================================


class WorkerLoop:
    """Worker loop for non-rank-0 processes to participate in distributed inference."""

    def __init__(
        self,
        policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
    ) -> None:
        self._policy = policy
        self._signal_group = signal_group

    def _receive_batch_from_rank0(self) -> Batch:
        """Receive batch data from rank 0 using torch.distributed primitives."""
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device="cuda")
        dist.broadcast(data_tensor, src=0)

        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    async def run(self):
        """Async worker loop."""
        rank = dist.get_rank()
        logger.info(f"Worker loop started for rank {rank}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                signal = signal_tensor.item()

                if signal == 1:
                    logger.info(f"Rank {rank} received shutdown signal")
                    break
                elif signal == 2:
                    logger.info(f"Rank {rank} received idle signal, waiting...")
                    continue

                # Receive obs from rank 0
                batch = self._receive_batch_from_rank0()

                # Participate in distributed forward pass
                dist.barrier()
                with torch.no_grad():
                    _ = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {rank}: {e}")
                traceback.print_exc()
                break


# ======================================================================
# Initialization helpers (from socket_test_optimized_AR.py)
# ======================================================================


def init_mesh() -> DeviceMesh:
    """Initialize distributed process group and device mesh."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")
    return mesh


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ======================================================================
# Main
# ======================================================================


def main(args: Args) -> None:
    # Environment setup
    os.environ["ATTENTION_BACKEND"] = "TE"
    try:
        torch._dynamo.config.recompile_limit = 800
    except AttributeError:
        torch._dynamo.config.cache_size_limit = 800

    embodiment_tag = args.embodiment_tag
    model_path = args.model_path

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    # Load model
    model_config_overrides = []
    if args.action_train_sigma_shift is not None:
        model_config_overrides.append(
            "action_head_cfg.config.action_train_sigma_shift="
            f"{args.action_train_sigma_shift}"
        )
    if args.action_inference_sigma_shift is not None:
        model_config_overrides.append(
            "action_head_cfg.config.action_inference_sigma_shift="
            f"{args.action_inference_sigma_shift}"
        )
    if rank == 0:
        logger.info("Model config overrides: %s", model_config_overrides)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_config_overrides=model_config_overrides,
        device_mesh=device_mesh,
    )

    # Output directory for videos
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        if args.video_output_root:
            output_dir = os.path.abspath(args.video_output_root)
        else:
            parent_dir = os.path.dirname(model_path)
            date_suffix = datetime.datetime.now().strftime("%Y%m%d")
            checkpoint_name = os.path.basename(model_path)
            output_dir = os.path.join(
                parent_dir,
                f"robotwin_eval_gen_{date_suffix}_{args.index}",
                checkpoint_name,
            )
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    # Create wrapper policy for RoboTwin / RoboTwin2 (dispatched by embodiment_tag)
    wrapper_policy = RoboTwinPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        action_dim=args.action_dim,
        state_joint_dim=args.state_joint_dim,
        state_gripper_dim=args.state_gripper_dim,
        output_dir=output_dir,
        embodiment_tag=args.embodiment_tag,
        use_obs_action_history=args.use_obs_action_history,
    )

    # Server config for RoboTwin (dual-arm, 3 cameras)
    server_config = PolicyServerConfig(
        image_resolution=(480, 640),  # Original RoboTwin camera resolution; model transform handles resize
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank == 0:
        logging.info("Using roboarena policy server interface for RoboTwin")
        logging.info(f"Server config: {server_config}")
        logging.info(f"Action dim: {args.action_dim}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0: run worker loop
        worker = WorkerLoop(
            policy=policy,
            signal_group=signal_group,
        )
        asyncio.run(worker.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)

    # Load YAML config if provided — YAML values override dataclass defaults,
    # but only for fields that were NOT explicitly set on the CLI.
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        _defaults = Args()
        for key, value in cfg.items():
            if not hasattr(args, key) or value is None:
                continue
            # Only apply YAML value when CLI still has the default
            if getattr(args, key) == getattr(_defaults, key):
                setattr(args, key, value)
        logging.info("Applied config from %s", args.config)

    main(args)
