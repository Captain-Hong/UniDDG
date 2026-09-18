import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config.defaults import get_cfg_defaults
from oiod_core import (
    build_criterions,
    build_models,
    build_optimizers,
    build_train_val_loaders,
    build_writer,
    make_checkpoint_state,
    save_checkpoint,
    setup_logging,
    setup_seed,
    train_one_epoch,
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
    parser = argparse.ArgumentParser(description="Train UniDDG")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data", type=str, required=True, choices=["optic", "prostate"], help="Dataset type")
    parser.add_argument("--root", type=str, default=None, help="Override DATASET.ROOT")
    parser.add_argument("--batch_size", type=int, default=None, help="Override TRAIN.BATCH_SIZE")
    parser.add_argument("--workers", type=int, default=None, help="Override TRAIN.NUM_WORKERS")
    parser.add_argument("--epochs", type=int, default=None, help="Override TRAIN.EPOCHS")
    parser.add_argument("--seed", type=int, default=None, help="Override TRAIN.SEED")
    parser.add_argument("--kernel_size", type=int, default=None, help="Override OIOD.KERNEL_SIZE")
    parser.add_argument("--eval_mode", type=str, default=None, choices=["argmax", "threshold"], help="Override OIOD.EVAL_MODE")
    parser.add_argument("--output_tag", type=str, default="", help="Extra tag for log directory")
    parser.add_argument("--result_json", type=str, default=None, help="Optional path to write result json")
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
    if args.epochs is not None:
        cfg.TRAIN.EPOCHS = int(args.epochs)
    if args.seed is not None:
        cfg.TRAIN.SEED = int(args.seed)
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


def run_training(cfg, args):
    setup_seed(int(cfg.TRAIN.SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_train_val_loaders(cfg)
    models = build_models(cfg, device)
    optimizers = build_optimizers(models, cfg)
    criterions = build_criterions()

    run_name = f"{cfg.DATASET.NAME}_uniddg_{time.strftime('%Y%m%d-%H%M%S')}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"

    log_dir = os.path.join(str(cfg.LOG.LOG_DIR), run_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, str(cfg.LOG.LOG_FILE)))
    writer = build_writer(log_dir)

    logger.info("==== UniDDG Training ====")
    logger.info(f"Device: {device}")
    logger.info(f"Data type: {args.data}")
    logger.info(f"Kernel size: {cfg.OIOD.KERNEL_SIZE}")
    logger.info(f"Config:\n{cfg.dump()}")

    best_f1 = -1.0
    best_epoch = 0
    best_metrics = {}

    for epoch in range(1, int(cfg.TRAIN.EPOCHS) + 1):
        train_terms = train_one_epoch(
            train_loader=train_loader,
            models=models,
            optimizers=optimizers,
            criterions=criterions,
            cfg=cfg,
            device=device,
            logger=logger,
            writer=writer,
            epoch=epoch,
        )

        writer.add_scalar("train/loss_epoch", float(train_terms.get("total_loss", 0.0)), epoch)

        if epoch % max(int(cfg.TRAIN.VAL_FREQ), 1) == 0:
            val_metrics = validate(test_loader, models, cfg, device, data_type=args.data)
            writer.add_scalar("val/avg_f1", float(val_metrics["avg_f1"]), epoch)

            if args.data == "optic":
                writer.add_scalar("val/cup_dice", float(val_metrics["val_cup_dice"]), epoch)
                writer.add_scalar("val/disc_dice", float(val_metrics["val_disc_dice"]), epoch)
                writer.add_scalar("val/asd_oc", float(val_metrics["total_asd_OC"]), epoch)
                writer.add_scalar("val/asd_od", float(val_metrics["total_asd_OD"]), epoch)
                logger.info(
                    f"Epoch {epoch}: avg_f1={val_metrics['avg_f1']:.4f}, "
                    f"cup={val_metrics['val_cup_dice']:.4f}, disc={val_metrics['val_disc_dice']:.4f}, "
                    f"asd_oc={val_metrics['total_asd_OC']:.4f}, asd_od={val_metrics['total_asd_OD']:.4f}"
                )
            else:
                writer.add_scalar("val/seg_dice", float(val_metrics["val_seg_dice"]), epoch)
                writer.add_scalar("val/asd_seg", float(val_metrics["total_asd_seg"]), epoch)
                logger.info(
                    f"Epoch {epoch}: avg_f1={val_metrics['avg_f1']:.4f}, "
                    f"seg={val_metrics['val_seg_dice']:.4f}, asd_seg={val_metrics['total_asd_seg']:.4f}"
                )

            if float(val_metrics["avg_f1"]) > best_f1:
                best_f1 = float(val_metrics["avg_f1"])
                best_epoch = int(epoch)
                best_metrics = dict(val_metrics)

                best_state = make_checkpoint_state(
                    models=models,
                    optimizers=optimizers,
                    cfg=cfg,
                    epoch=epoch,
                    best_f1=best_f1,
                    best_epoch=best_epoch,
                )
                save_checkpoint(best_state, os.path.join(log_dir, "best_model.pth"))
                logger.info(f"New best checkpoint @ epoch {epoch}, avg_f1={best_f1:.4f}")

    last_state = make_checkpoint_state(
        models=models,
        optimizers=optimizers,
        cfg=cfg,
        epoch=int(cfg.TRAIN.EPOCHS),
        best_f1=best_f1,
        best_epoch=best_epoch,
    )
    save_checkpoint(last_state, os.path.join(log_dir, "last_model.pth"))

    result = {
        "data": args.data,
        "kernel_size": int(cfg.OIOD.KERNEL_SIZE),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "log_dir": log_dir,
    }

    write_json(os.path.join(log_dir, "result.json"), result)
    if args.result_json:
        write_json(args.result_json, result)

    logger.info(f"Training finished. best_epoch={best_epoch}, best_f1={best_f1:.4f}")
    writer.close()
    return result


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    result = run_training(cfg, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
