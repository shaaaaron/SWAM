"""Streaming UMI episode-relative stats, sharing conversion and quality with training."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from rlinf.data.datasets.dreamzero.noe_umi_episode_reference import (
    NOE_UMI_FRAME_QUALITY_FIELDS,
    noe_umi_episode_quality,
    read_noe_umi_episode_reference,
    read_noe_umi_frame_quality,
)
from rlinf.data.datasets.dreamzero.relative_actions import (
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    robotwin_eef_abs_to_rel_wxyz,
)


def build_noe_umi_episode_metadata(
    roots: list[Path],
    *,
    episode_stride: int = 1,
    max_episodes: int | None = None,
    quantile_sample_limit: int = 100000,
    stats_seed: int = 42,
    apply_frame_quality: bool = False,
    state_key: str = "observation.state",
    action_key: str = "action",
    video_keys: list[str] | None = None,
    video_resolution: list[int] | None = None,
    episode_exclusion_manifest: Path | None = None,
    filter_qualified_episodes: bool = False,
    skip_missing_episodes: bool = False,
    audit_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from toolkits.lerobot.generate_dreamzero_metadata import (
        PRESETS,
        _column_to_2d_array,
        _episode_path,
        _load_episodes,
        _noe_umi_16d_video_modalities,
        _resolve_column,
        _StreamingStatsAccumulator,
    )

    if episode_stride <= 0 or quantile_sample_limit <= 0 or not roots:
        raise ValueError(
            "Episode metadata requires roots and positive stride/sample limit"
        )
    started = time.monotonic()
    roots = [Path(p).resolve() for p in roots]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate UMI roots in metadata input")
    selected: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    sources = []
    for root in roots:
        info_path, episodes_path = root / "meta/info.json", root / "meta/episodes.jsonl"
        info = json.loads(info_path.read_text())
        episodes = [
            json.loads(s) for s in episodes_path.read_text().splitlines() if s.strip()
        ]
        by_id = {int(ep["episode_index"]): ep for ep in episodes}
        if len(by_id) != len(episodes):
            raise ValueError(f"Duplicate episode indices under {root}")
        keep = _load_episodes(
            root / "meta",
            filter_noe_umi_qualified_episodes=filter_qualified_episodes,
            episode_exclusion_manifest=episode_exclusion_manifest,
        )
        keep = keep[::episode_stride]
        if max_episodes is not None:
            keep = keep[:max_episodes]
        chosen = [by_id[i] for i in keep]
        selected.append((root, info, chosen))
        sources.append(
            {
                "root": str(root),
                "selected_episodes": len(chosen),
                "total_episodes": len(episodes),
                "selected_raw_frames": sum(int(ep.get("length", 0)) for ep in chosen),
            }
        )
    total_episodes = sum(len(eps) for _, _, eps in selected)
    total_frames = sum(s["selected_raw_frames"] for s in sources)
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "selection.json").write_text(
            json.dumps(
                [
                    {
                        "root": str(root),
                        "episode_indices": [int(e["episode_index"]) for e in eps],
                    }
                    for root, _, eps in selected
                ],
                indent=2,
            )
        )
    global_acc = {
        k: _StreamingStatsAccumulator(
            quantile_sample_limit=quantile_sample_limit, seed=stats_seed
        )
        for k in ("state", "action")
    }
    counts: Counter = Counter()
    root_audits = []

    def progress(root_name: str, *, complete: bool = False) -> None:
        elapsed = time.monotonic() - started
        done = counts["processed_episodes"]
        frame_rate = counts["raw_frames"] / elapsed if elapsed else 0.0
        remaining = max(0, total_frames - counts["raw_frames"])
        eta = remaining / frame_rate if frame_rate else None
        obj = {
            "root": root_name,
            "processed_episodes": done,
            "selected_episodes": total_episodes,
            "raw_frames": counts["raw_frames"],
            "selected_raw_frames": total_frames,
            "state_rows": global_acc["state"].count,
            "action_rows": global_acc["action"].count,
            "invalid_reference_episodes": counts["invalid_reference_episodes"],
            "missing_episodes": counts["missing_episodes"],
            "elapsed_s": elapsed,
            "raw_frames_per_second": frame_rate,
            "eta_seconds": eta,
            "complete": complete,
        }
        if eta is not None:
            obj["estimated_finish_hong_kong"] = (
                datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=eta)
            ).isoformat()
        print("UMI_EPISODE_PROGRESS " + json.dumps(obj), flush=True)
        if audit_dir is not None:
            path = audit_dir / "progress.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj, indent=2))
            tmp.replace(path)

    for root, info, episodes in selected:
        root_counts: Counter = Counter()
        acc = {
            k: _StreamingStatsAccumulator(
                quantile_sample_limit=min(10000, quantile_sample_limit), seed=stats_seed
            )
            for k in ("state", "action")
        }
        for ep in episodes:
            index = int(ep["episode_index"])
            path = _episode_path(root, info, index)
            if not path.is_file():
                if not skip_missing_episodes:
                    raise FileNotFoundError(path)
                counts["missing_episodes"] += 1
                counts["processed_episodes"] += 1
                root_counts["missing_episodes"] += 1
                print(f"Skipping missing episode: {path}", flush=True)
                continue
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            sk = _resolve_column(names, state_key, ["observation.state", "state"])
            ak = _resolve_column(names, action_key, ["action", "actions"])
            columns = [sk, ak] + [q for q in NOE_UMI_FRAME_QUALITY_FIELDS if q in names]
            table = pf.read(columns=columns, use_threads=False)
            state = _column_to_2d_array(table, sk)[:, :16]
            action = _column_to_2d_array(table, ak)[:, :16]
            if state.shape != action.shape or state.shape[1] != 16:
                raise ValueError(f"Expected matching 16D UMI state/action under {path}")
            eq = noe_umi_episode_quality(ep)
            reference, reference_valid = read_noe_umi_episode_reference(
                table, eq, state_key=sk
            )
            # Always use row zero, including quality-invalid references.
            relative = {
                "state": robotwin_eef_abs_to_rel_wxyz(state, reference, "per_arm"),
                "action": robotwin_eef_abs_to_rel_wxyz(action, reference, "per_arm"),
            }
            n = len(state)
            if apply_frame_quality:
                quality = read_noe_umi_frame_quality(table, np.arange(n), eq)
                masks = {
                    "state": quality & reference_valid,
                    "action": quality[np.minimum(np.arange(n) + 1, n - 1)]
                    & reference_valid,
                }
            else:
                masks = {
                    "state": np.ones(n, dtype=bool),
                    "action": np.ones(n, dtype=bool),
                }
            for kind in ("state", "action"):
                values = relative[kind][masks[kind]]
                if len(values):
                    global_acc[kind].update(values)
                    acc[kind].update(values)
                root_counts[kind + "_rows"] += len(values)
            counts["processed_episodes"] += 1
            counts["raw_frames"] += n
            counts["invalid_reference_episodes"] += not reference_valid
            root_counts["loaded_episodes"] += 1
            root_counts["raw_frames"] += n
            root_counts["invalid_reference_episodes"] += not reference_valid
            root_counts["quality_fallback_episodes"] += not set(
                NOE_UMI_FRAME_QUALITY_FIELDS
            ).issubset(names)
            if counts["processed_episodes"] % 100 == 0:
                progress(root.name)
        root_audits.append(
            {
                "root": str(root),
                "counts": dict(root_counts),
                "statistics": {
                    k: a.finalize() if a.count else None for k, a in acc.items()
                },
            }
        )
        progress(root.name)
    statistics = {k: {"eef_position": a.finalize()} for k, a in global_acc.items()}
    modality = {
        "eef_position": {
            "absolute": True,
            "rotation_type": None,
            "shape": [16],
            "continuous": True,
        }
    }
    entry = {
        "statistics": statistics,
        "embodiment_tag": "noe_umi_16d",
        "modalities": {
            "video": _noe_umi_16d_video_modalities(
                selected[0][1],
                video_keys or PRESETS["noe_umi_16d"]["video_keys"],
                video_resolution,
            ),
            "state": modality,
            "action": modality,
        },
        "action_quaternion_mode": "preserve",
        "relative_action_mode": RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
        "relative_action_pose_reference": "per_arm",
        "relative_action_reference": "episode_frame0",
        "apply_noe_umi_frame_quality": apply_frame_quality,
        "relative_action_stats": {
            "episode_stride": episode_stride,
            "max_episodes": max_episodes,
            "streaming_stats": True,
            "quantile_sample_limit": quantile_sample_limit,
            "stats_seed": stats_seed,
            "statistics_unit": "source_frame",
            "reference_quality_gate": apply_frame_quality,
            "state_rows": global_acc["state"].count,
            "action_rows": global_acc["action"].count,
            "sources": sources,
        },
    }
    audit = {
        "state.eef_position": statistics["state"]["eef_position"],
        "action.eef_position": statistics["action"]["eef_position"],
        "audit": {
            "sources": sources,
            "roots": root_audits,
            "counts": dict(counts),
            "selected_episodes": total_episodes,
            "elapsed_s": time.monotonic() - started,
            "apply_noe_umi_frame_quality": apply_frame_quality,
        },
    }
    progress("all", complete=True)
    return {"noe_umi_16d": entry}, audit
