"""Run video-only DreamZero inference on one LeRobot episode.

This script is intentionally small:
  - loads a checkpoint with GrootSimPolicy
  - reads one episode from a LeRobot-style data root
  - feeds video chunks through policy.lazy_joint_forward_causal(..., video_only=True)
  - decodes and saves only the predicted video

Default mapping targets DROID-style data/checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


VAE_TEMPORAL_STRIDE = 4
SINGLE_VIEW_VIDEO_KEYS = ["video.wrist_image_left"]
THREE_VIEW_VIDEO_KEYS = [
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
]

MODALITY_MAPS = {
    "oxe_droid": {
        "video": {
            "exterior_image_1_left": "video.exterior_image_1_left",
            "exterior_image_2_left": "video.exterior_image_2_left",
            "wrist_image_left": "video.wrist_image_left",
        },
        "state": {
            "state.joint_position": (7, 14),
            "state.gripper_position": (6, 7),
        },
        "language": "annotation.language.language_instruction",
    },
    "oxe_droid_wrist": {
        "video": {
            "wrist_image_left": "video.wrist_image_left",
        },
        "state": {
            "state.joint_position": (7, 14),
            "state.gripper_position": (6, 7),
        },
        "language": "annotation.language.language_instruction",
    },
}


import socket
def init_distributed() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((os.environ["MASTER_ADDR"], 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        print(f"Using auto-selected MASTER_PORT={os.environ['MASTER_PORT']}")

    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def load_tasks(data_root: Path) -> dict[int, str]:
    tasks_path = data_root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if not tasks_path.exists():
        return tasks
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def resolve_language(df, tasks: dict[int, str]) -> str:
    for col in df.columns:
        if "language_instruction" not in col:
            continue
        value = df[col].iloc[0]
        if isinstance(value, str):
            return value
        try:
            return tasks.get(int(value), f"task_{int(value)}")
        except Exception:
            continue
    if "task_index" in df.columns:
        task_idx = int(df["task_index"].iloc[0])
        return tasks.get(task_idx, f"task_{task_idx}")
    return "Perform the default behavior."


def read_video(video_file: Path, n_frames: int) -> list[np.ndarray]:
    try:
        from decord import VideoReader

        vr = VideoReader(str(video_file))
        return [vr[i].asnumpy() for i in range(min(len(vr), n_frames))]
    except Exception as exc:
        print(f"  decord failed for {video_file.name} ({exc}); falling back to cv2")

    cap = cv2.VideoCapture(str(video_file))
    frames: list[np.ndarray] = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_episode(data_root: Path, episode_idx: int, modality_map: dict) -> dict:
    import pyarrow.parquet as pq

    parquet_files = sorted(data_root.glob("data/**/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_root / 'data'}")
    if episode_idx >= len(parquet_files):
        episode_idx = 0

    parquet_path = parquet_files[episode_idx]
    df = pq.read_table(parquet_path).to_pandas().sort_values("frame_index")
    episode_id = int(df["episode_index"].iloc[0])
    chunk_name = parquet_path.parent.name
    n_frames = len(df)

    print(f"Episode {episode_id} ({chunk_name}): {n_frames} frames")
    language = resolve_language(df, load_tasks(data_root))
    print(f"  Language: {language}")

    videos: dict[str, list[np.ndarray]] = {}
    for raw_view in modality_map["video"]:
        video_file = (
            data_root
            / "videos"
            / chunk_name
            / f"observation.images.{raw_view}"
            / f"episode_{episode_id:06d}.mp4"
        )
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        videos[raw_view] = read_video(video_file, n_frames)
        print(f"  Loaded {len(videos[raw_view])} frames from {raw_view}")

    usable_frames = min([n_frames] + [len(frames) for frames in videos.values()])
    frames = [
        {raw_view: view_frames[i] for raw_view, view_frames in videos.items()}
        for i in range(usable_frames)
    ]

    states = None
    if modality_map.get("state"):
        if "observation.state" not in df.columns:
            raise KeyError("modality map requires state, but parquet has no observation.state")
        states = [
            np.asarray(row, dtype=np.float64)
            for row in df["observation.state"].iloc[:usable_frames]
        ]

    return {
        "frames": frames,
        "states": states,
        "language": language,
        "episode_id": episode_id,
    }


def reset_action_head_state(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    action_head.current_start_frame = 0
    action_head.language = None
    action_head.clip_feas = None
    action_head.ys = None
    action_head.kv_cache1 = None
    action_head.kv_cache_neg = None
    action_head.crossattn_cache = None
    action_head.crossattn_cache_neg = None


def override_eval_video_views(policy: GrootSimPolicy, eval_views: str) -> None:
    if eval_views == "checkpoint":
        print("Using checkpoint eval video transform")
        return

    video_keys = SINGLE_VIEW_VIDEO_KEYS if eval_views == "single" else THREE_VIEW_VIDEO_KEYS
    for transform in policy.eval_transform.transforms:
        if hasattr(transform, "video_concat_order"):
            transform.video_concat_order = video_keys
        elif getattr(transform, "apply_to", None) and all(
            key.startswith("video.") for key in transform.apply_to
        ):
            transform.apply_to = video_keys

    print(f"Overriding eval video views to {eval_views}: {video_keys}")


def infer_frames_per_chunk(policy: GrootSimPolicy, override: int | None) -> int:
    if override is not None:
        print(f"Using user-provided frames_per_chunk={override}")
        return override
    action_head = policy.trained_model.action_head
    num_frame_per_block = int(getattr(action_head, "num_frame_per_block", 1) or 1)
    frames_per_chunk = max(1, num_frame_per_block * VAE_TEMPORAL_STRIDE)
    print(
        "Inferred frames_per_chunk from checkpoint: "
        f"num_frame_per_block={num_frame_per_block}, "
        f"vae_temporal_stride={VAE_TEMPORAL_STRIDE}, "
        f"frames_per_chunk={frames_per_chunk}"
    )
    return frames_per_chunk


def infer_max_chunks(policy: GrootSimPolicy, frames_per_chunk: int, override: int | None) -> int | None:
    if override is not None:
        print(f"Using user-provided max_chunks={override}")
        return override
    action_head = policy.trained_model.action_head
    num_frames = getattr(action_head, "num_frames", None)
    if num_frames is None:
        print("Could not infer max_chunks: action_head.num_frames is missing")
        return None
    num_frames = int(num_frames)
    max_chunks = max(1, (num_frames - 1) // frames_per_chunk)
    covered_frames = max_chunks * frames_per_chunk + 1
    print(
        "Inferred max_chunks from checkpoint: "
        f"num_frames={num_frames}, frames_per_chunk={frames_per_chunk}, "
        f"max_chunks={max_chunks}, covered_frames={covered_frames}"
    )
    if covered_frames != num_frames:
        print(
            "  WARNING: inferred chunks do not exactly cover checkpoint num_frames "
            f"({covered_frames} != {num_frames})"
        )
    return max_chunks


def maybe_set_num_inference_steps(policy: GrootSimPolicy, override: int | None) -> None:
    action_head = policy.trained_model.action_head
    current_steps = int(getattr(action_head, "num_inference_steps", 0) or 0)
    if override is None:
        print(f"Using checkpoint/code num_inference_steps={current_steps}")
        return
    if override <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {override}")
    action_head.num_inference_steps = override
    print(f"Overriding num_inference_steps: {current_steps} -> {override}")


def decode_and_save_video(
    policy: GrootSimPolicy,
    latents_list: list[torch.Tensor],
    output_path: Path,
    fps: int,
    target_num_frames: int | None = None,
) -> None:
    action_head = policy.trained_model.action_head
    latents = torch.cat(latents_list, dim=2)
    print(f"Predicted video latent shape: {tuple(latents.shape)}")
    with torch.no_grad():
        decoded = action_head.vae.decode(
            latents,
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
    decoded = rearrange(decoded, "B C T H W -> B T H W C")[0]
    frames = ((decoded.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    if target_num_frames is not None and len(frames) > target_num_frames:
        print(f"Cropping decoded video frames: {len(frames)} -> {target_num_frames}")
        frames = frames[:target_num_frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps, codec="libx264")
    print(f"Saved predicted video ({len(frames)} frames @ {fps} fps): {output_path}")


def run_video_only_inference(
    policy: GrootSimPolicy,
    episode_data: dict,
    modality_map: dict,
    output_dir: Path,
    image_height: int,
    image_width: int,
    frames_per_chunk: int,
    max_chunks: int | None,
    fps: int,
    episode_idx,
) -> None:
    frames_list = episode_data["frames"]
    states = episode_data["states"] # GrootPolicy 需要对state输入进行预处理
    language = episode_data["language"]

    if max_chunks is not None:
        last_call_frame = (max_chunks - 1) * frames_per_chunk
        n_frames = min(len(frames_list), last_call_frame + 1)
        target_output_frames = max_chunks * frames_per_chunk + 1
    else:
        n_frames = len(frames_list)
        target_output_frames = None
    print(
        f"Running video-only inference on {n_frames}/{len(frames_list)} input frames "
        f"(frames_per_chunk={frames_per_chunk}, max_chunks={max_chunks}, "
        f"target_output_frames={target_output_frames})"
    )

    reset_action_head_state(policy)
    action_head = policy.trained_model.action_head
    num_frame_per_block = int(getattr(action_head, "num_frame_per_block", 1) or 1)

    model_video_keys = list(modality_map["video"].values())
    frame_buffers = {model_key: [] for model_key in model_video_keys}
    video_pred_latents: list[torch.Tensor] = []
    latent_video: torch.Tensor | None = None
    is_first_call = True
    step = 0

    for frame_idx in range(n_frames):
        frame_data = frames_list[frame_idx]
        for raw_view, model_key in modality_map["video"].items():
            img = cv2.resize(
                frame_data[raw_view],
                (image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            frame_buffers[model_key].append(img)

        buffer_len = len(frame_buffers[model_video_keys[0]])
        if is_first_call and buffer_len >= 1:
            num_frames_to_use = 1
        elif not is_first_call and (buffer_len - 1) % frames_per_chunk == 0 and buffer_len > 1:
            num_frames_to_use = frames_per_chunk
        else:
            continue

        obs: dict = {}
        for model_key, buffer in frame_buffers.items():
            selected = buffer[-num_frames_to_use:]
            while len(selected) < num_frames_to_use:
                selected.insert(0, buffer[0])
            obs[model_key] = np.stack(selected, axis=0)

        if states is not None:
            state_vec = states[frame_idx]
            for state_key, (start, end) in modality_map["state"].items():
                obs[state_key] = state_vec[start:end].reshape(1, -1)

        obs[modality_map["language"]] = language
        # dict_keys(['video.exterior_image_1_left', 'video.exterior_image_2_left', 'video.wrist_image_left', 'state.joint_position', 'state.gripper_position', 'annotation.language.language_instruction'])
        if not is_first_call:
            obs = {k: np.zeros_like(v) if k.startswith("video.") else v for k, v in obs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            _, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=obs),
                latent_video=latent_video,  # 截然不同
                video_only=True,
            )
        elapsed = time.perf_counter() - t0

        if video_pred is not None:
            video_pred_latents.append(video_pred.detach())
            latent_video = video_pred[:, :, -num_frame_per_block:].detach()
            print(
                f"  step={step:03d} frame={frame_idx:04d} "
                f"input_frames={num_frames_to_use} video_pred={tuple(video_pred.shape)} "
                f"latent_video={tuple(latent_video.shape)} "
                f"time={elapsed:.2f}s"
            )

        is_first_call = False
        step += 1

    if not video_pred_latents:
        raise RuntimeError("No video predictions were produced.")

    decode_and_save_video(
        policy,
        video_pred_latents,
        output_dir / f"pred_video_{episode_idx}.mp4",
        fps,
        target_num_frames=target_output_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-only inference on one LeRobot episode")
    parser.add_argument("--model_path", required=True, help="Path to DreamZero checkpoint directory")
    parser.add_argument("--data_root", help="Path to LeRobot dataset root", required=True)
    parser.add_argument("--tokenizer_path", help="Optional tokenizer path override", default=None)
    parser.add_argument("--embodiment_tag", default="oxe_droid")
    parser.add_argument("--modality_map", default="oxe_droid", choices=sorted(MODALITY_MAPS))
    parser.add_argument("--eval_views", default="checkpoint", choices=["checkpoint", "single", "three"],
                        help="Override checkpoint eval video transform view count.")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="./eval_video_only")
    parser.add_argument("--image_height", type=int, default=160)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--frames_per_chunk", type=int, default=None,
                        help="Raw frames per causal chunk. Defaults to action_head.num_frame_per_block * 4.")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Number of causal chunks to run. Defaults to checkpoint num_frames.")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Denoising steps per causal chunk. Defaults to the loaded action head value.")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 800
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1000

    init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    policy_kwargs = dict(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )
    if args.tokenizer_path:
        policy_kwargs["tokenizer_path_override"] = args.tokenizer_path

    print(f"Loading model: {args.model_path}")
    policy = GrootSimPolicy(**policy_kwargs)
    override_eval_video_views(policy, args.eval_views)
    print("Model loaded.")

    modality_map = MODALITY_MAPS[args.modality_map]
    episode_data = load_episode(Path(args.data_root), args.episode_idx, modality_map)
    frames_per_chunk = infer_frames_per_chunk(policy, args.frames_per_chunk)
    max_chunks = infer_max_chunks(policy, frames_per_chunk, args.max_chunks)
    maybe_set_num_inference_steps(policy, args.num_inference_steps)

    run_video_only_inference(
        policy=policy,
        episode_data=episode_data,
        modality_map=modality_map,
        output_dir=Path(args.output_dir),
        image_height=args.image_height,
        image_width=args.image_width,
        frames_per_chunk=frames_per_chunk,
        max_chunks=max_chunks,
        fps=args.fps,
        episode_idx=args.episode_idx,
    )


if __name__ == "__main__":
    main()


"""
THREE_ACT_CKPT=../ckpts/droid_keep/checkpoint-25000
THREE_CKPT=../ckpts/0520_video_keep/checkpoint-25000
ONE_CKPT=../ckpts/0520_wrist_video/checkpoint-10000

DATA_ROOT=/path/to/dreamzero-droid-data-zy
TOKENIZER=/path/to/umt5-xxl

CUDA_VISIBLE_DEVICES=0 \
python scripts/eval/infer_video_only_simple.py \
  --model_path "$SINGLE_CKPT" \
  --output_dir ./ckpt_single-eval_single \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid_wrist \
  --eval_views single \
  --data_root "$DATA_ROOT" \
  --tokenizer_path "$TOKENIZER" \
  --episode_idx 0

CUDA_VISIBLE_DEVICES=1 \
python scripts/eval/infer_video_only_simple.py \
  --model_path "$SINGLE_CKPT" \
  --output_dir ./ckpt_single-eval_three \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --data_root "$DATA_ROOT" \
  --tokenizer_path "$TOKENIZER" \
  --episode_idx 0

CUDA_VISIBLE_DEVICES=2 \
python scripts/eval/infer_video_only_simple.py \
  --model_path "$THREE_CKPT" \
  --output_dir ./ckpt_three-eval_single \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid_wrist \
  --eval_views single \
  --data_root "$DATA_ROOT" \
  --tokenizer_path "$TOKENIZER" \
  --episode_idx 0

CUDA_VISIBLE_DEVICES=3 \
python scripts/eval/infer_video_only_simple.py \
  --model_path "$THREE_CKPT" \
  --output_dir ./ckpt_three-eval_three \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --data_root "$DATA_ROOT" \
  --tokenizer_path "$TOKENIZER" \
  --episode_idx 0


Date: Video Only, Diff LR
python scripts/eval/infer_video_only_simple.py \
  --model_path ../ckpts/0527_droid_video_only_robotwin_resolution_lr5e-5/checkpoint-3000 \
  --output_dir .visual/0527_video_lr_5e-5/checkpoint-3000 \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --episode_idx 0 \
  --image_width 320 \
  --image_height 256
python scripts/eval/infer_video_only_simple.py \
  --model_path ../ckpts/0527_droid_video_only_robotwin_resolution_lr5e-5/checkpoint-2000 \
  --output_dir .visual/0527_video_lr_5e-5/checkpoint-2000 \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --episode_idx 0 \
  --image_width 320 \
  --image_height 256
python scripts/eval/infer_video_only_simple.py \
  --model_path ../ckpts/0527_droid_video_only_robotwin_resolution_lr5e-5/checkpoint-1000 \
  --output_dir .visual/0527_video_lr_5e-5/checkpoint-1000 \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --episode_idx 0 \
  --image_width 320 \
  --image_height 256


Date: Video Only, Diff LR
CUDA_VISIBLE_DEVICES=0 python scripts/eval/infer_video_only_simple.py \
  --model_path /path/to/checkpoint-10000 \
  --output_dir .visual/0527_video_lr_5e-5/checkpoint-3000 \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --episode_idx 0 \
  --image_width 320 \
  --image_height 256
CUDA_VISIBLE_DEVICES=1 python scripts/eval/infer_video_only_simple.py \
  --model_path /path/to/checkpoint-10000 \
  --output_dir .visual/0527_video_lr_5e-5/checkpoint-2000 \
  --embodiment_tag oxe_droid \
  --modality_map oxe_droid \
  --eval_views three \
  --episode_idx 0 \
  --image_width 320 \
  --image_height 256

"""
