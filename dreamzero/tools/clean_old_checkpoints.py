import argparse
import re
import shutil
from pathlib import Path


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
GLOBAL_STEP_RE = re.compile(r"^global_step(\d+)$")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def global_step(path: Path) -> int | None:
    match = GLOBAL_STEP_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def find_experiment_dirs(root: Path) -> list[Path]:
    if any(child.is_dir() and checkpoint_step(child) is not None for child in root.iterdir()):
        return [root]

    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and any(grandchild.is_dir() and checkpoint_step(grandchild) is not None for grandchild in child.iterdir())
    ]


def old_global_steps(exp_dir: Path, keep: int) -> tuple[list[Path], list[Path]]:
    checkpoints = sorted(
        [child for child in exp_dir.iterdir() if child.is_dir() and checkpoint_step(child) is not None],
        key=lambda path: checkpoint_step(path),
    )
    if keep <= 0:
        raise ValueError("--keep must be positive")

    global_step_dirs = sorted(
        [
            grandchild
            for checkpoint in checkpoints
            for grandchild in checkpoint.iterdir()
            if grandchild.is_dir() and global_step(grandchild) is not None
        ],
        key=lambda path: global_step(path),
    )
    return global_step_dirs[:-keep], global_step_dirs[-keep:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete older global_stepN directories under checkpoint-N dirs, keeping the latest per experiment."
    )
    parser.add_argument("root", type=Path, help="Checkpoint root, or one experiment directory containing checkpoint-N dirs.")
    parser.add_argument("--keep", type=int, default=1, help="How many latest global_stepN directories to keep per experiment.")
    parser.add_argument("--delete", action="store_true", help="Actually delete files. Without this, only print a dry run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    exp_dirs = find_experiment_dirs(root)
    total_delete = 0

    print(f"root: {root}")
    print(f"mode: {'DELETE' if args.delete else 'DRY-RUN'}")
    print(f"experiments found: {len(exp_dirs)}")

    for exp_dir in exp_dirs:
        delete_dirs, keep_dirs = old_global_steps(exp_dir, args.keep)
        if not delete_dirs and not keep_dirs:
            continue

        keep_names = ", ".join(str(path.relative_to(exp_dir)) for path in keep_dirs)
        print(f"\n[EXP] {exp_dir}")
        print(f"keep: {keep_names}")

        for ckpt_dir in delete_dirs:
            total_delete += 1
            print(f"{'delete' if args.delete else 'would delete'}: {ckpt_dir}")
            if args.delete:
                shutil.rmtree(ckpt_dir)

    print(f"\nold global_step dirs {'deleted' if args.delete else 'to delete'}: {total_delete}")
    if not args.delete:
        print("dry run only; rerun with --delete to remove them")


if __name__ == "__main__":
    main()

"""
# 先 dry-run，看会删哪些
python /path/to/clean_old_checkpoints.py /path/to/ckpts

# 确认没问题后真删
python /path/to/clean_old_checkpoints.py /path/to/ckpts --delete
"""