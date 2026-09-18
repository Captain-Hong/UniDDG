import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config.defaults import get_cfg_defaults
from oiod_core import (
    build_models,
    build_train_val_loaders,
    load_checkpoint,
    validate,
    write_json,
)


def resolve_config_path(config_path: str):
    candidate = Path(str(config_path))
    if candidate.exists():
        return str(candidate.resolve())

    local_candidate = (CURRENT_DIR / candidate).resolve()
    if local_candidate.exists():
        return str(local_candidate)

    return str(config_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained UniDDG model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, choices=["optic", "prostate"], help="Dataset type")
    parser.add_argument("--root", type=str, default=None, help="Override DATASET.ROOT")
    parser.add_argument("--batch_size", type=int, default=None, help="Override TRAIN.BATCH_SIZE")
    parser.add_argument("--workers", type=int, default=None, help="Override TRAIN.NUM_WORKERS")
    parser.add_argument("--kernel_size", type=int, default=None, help="Override OIOD.KERNEL_SIZE")
    parser.add_argument("--eval_mode", type=str, default=None, choices=["argmax", "threshold"], help="Override OIOD.EVAL_MODE")
    parser.add_argument("--save_csv", type=str, default="UniDDG_Dice_results.csv", help="CSV output file")
    parser.add_argument("--save_json", type=str, default=None, help="Optional JSON output file")
    return parser.parse_args()


def prepare_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(resolve_config_path(args.config))

    cfg.defrost()
    if args.root is not None:
        cfg.DATASET.ROOT = args.root
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = int(args.batch_size)
    if args.workers is not None:
        cfg.TRAIN.NUM_WORKERS = int(args.workers)
    if args.kernel_size is not None:
        cfg.OIOD.KERNEL_SIZE = int(args.kernel_size)
    if args.eval_mode is not None:
        cfg.OIOD.EVAL_MODE = str(args.eval_mode).lower()

    if args.data == "optic" and int(cfg.MODEL.NUM_CLASSES) < 3:
        raise ValueError("Optic task requires MODEL.NUM_CLASSES >= 3 for OD/OC channels.")
    if args.data == "prostate" and int(cfg.MODEL.NUM_CLASSES) < 2:
        raise ValueError("Prostate task requires MODEL.NUM_CLASSES >= 2.")

    cfg.freeze()
    return cfg


def write_csv(save_path, metrics, data_type):
    if not save_path:
        return
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if data_type == "optic":
        header = ["cup_dice", "disc_dice"]
        rows = [[cup, disc] for cup, disc in zip(metrics.get("cup_list", []), metrics.get("disc_list", []))]
    else:
        header = ["seg_dice"]
        rows = [[value] for value in metrics.get("disc_list", [])]

    file_exists = os.path.exists(save_path)
    with open(save_path, "a+", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj, dialect="excel")
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    cfg = prepare_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = build_train_val_loaders(cfg)
    models = build_models(cfg, device)

    checkpoint = load_checkpoint(args.weight_path, models=models, optimizers=None, map_location=device)
    metrics = validate(test_loader, models, cfg, device, data_type=args.data)

    result = {
        "data": args.data,
        "kernel_size": int(cfg.OIOD.KERNEL_SIZE),
        "weight_path": args.weight_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "metrics": metrics,
    }

    if args.data == "optic":
        print(
            f"Cup Dice: {metrics['val_cup_dice']:.4f} | Disc Dice: {metrics['val_disc_dice']:.4f} | "
            f"ASD OC: {metrics['total_asd_OC']:.4f} | ASD OD: {metrics['total_asd_OD']:.4f}"
        )
    else:
        print(f"Seg Dice: {metrics['val_seg_dice']:.4f} | ASD Seg: {metrics['total_asd_seg']:.4f}")

    write_csv(args.save_csv, metrics, args.data)
    if args.save_json:
        write_json(args.save_json, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
