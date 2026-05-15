#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


DEFAULT_CATEGORIES = Path(__file__).resolve().parents[1] / "configs" / "places365_building_categories.txt"


def read_categories(path: Path):
    categories = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            categories.append(line.strip("/"))
    return categories


def category_matches(rel_path: str, categories):
    rel_path = rel_path.replace("\\", "/")
    for category in categories:
        if rel_path.startswith(f"{category}/") or f"/{category}/" in f"/{rel_path}":
            return True
    return False


def collect_images(source_root: Path, categories):
    images = []
    for suffix in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for image_path in source_root.rglob(suffix):
            rel_path = image_path.relative_to(source_root).as_posix()
            if category_matches(rel_path, categories):
                images.append(image_path)
    return sorted(set(images))


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_split(items, output_root: Path, split: str, mode: str):
    split_root = output_root / split
    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(items):
        safe_name = f"{index:06d}_{image_path.stem}.jpg"
        link_or_copy(image_path, split_root / safe_name, mode)


def main():
    parser = argparse.ArgumentParser(description="Prepare a Places365 building subset for LaMa fine-tuning.")
    parser.add_argument("--source-root", required=True, type=Path, help="Root of extracted Places365 images.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root with train/val/test folders.")
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES, type=Path, help="Category list file.")
    parser.add_argument("--train-count", default=3000, type=int)
    parser.add_argument("--val-count", default=300, type=int)
    parser.add_argument("--test-count", default=300, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    categories = read_categories(args.categories)
    images = collect_images(args.source_root, categories)
    required = args.train_count + args.val_count + args.test_count
    if len(images) < required:
        raise SystemExit(f"Need {required} images but found {len(images)} under {args.source_root}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    train = images[:args.train_count]
    val = images[args.train_count:args.train_count + args.val_count]
    test = images[args.train_count + args.val_count:required]

    write_split(train, args.output_root, "train", args.mode)
    write_split(val, args.output_root, "val", args.mode)
    write_split(test, args.output_root, "test", args.mode)

    print(f"categories: {', '.join(categories)}")
    print(f"found: {len(images)}")
    print(f"train: {len(train)}")
    print(f"val: {len(val)}")
    print(f"test: {len(test)}")
    print(f"output: {args.output_root}")
    print(f"mode: {args.mode}")


if __name__ == "__main__":
    main()
