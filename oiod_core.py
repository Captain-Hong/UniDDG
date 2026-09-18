import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
try:
    import medpy.metric.binary as binary
except Exception:
    binary = None
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import NpySegmentationDataset

try:
    from models.networks.decoder import Ada_Decoder
    from models.networks.sdnet import AEncoder, MEncoder
    from models.networks.segmentor import OIODSegmentor
    from models.weight_init import initialize_weights
    from utils.losses import KL_divergence, SoftDiceLoss
    from utils.metrics import dice_coeff_2label
    from utils.tools import postprocessing
except ModuleNotFoundError as import_exc:
    raise ModuleNotFoundError(
        "Local dependencies missing. Please ensure `models` and `utils` are present."
    ) from import_exc


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def build_writer(log_dir: str):
    if SummaryWriter is None:
        return DummyWriter()
    return SummaryWriter(log_dir=log_dir)


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: str):
    logger = logging.getLogger("oiod_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_paths(root_dir: str, dataset_paths: Sequence[str]) -> List[str]:
    resolved = []
    for path in dataset_paths:
        path = str(path)
        if os.path.isabs(path):
            resolved.append(os.path.normpath(path))
        elif root_dir:
            resolved.append(os.path.normpath(os.path.join(root_dir, path)))
        else:
            resolved.append(os.path.normpath(path))
    return resolved


def expand_file_list(file_list: Sequence[str], root_count: int) -> List[str]:
    file_list = [str(item) for item in file_list]
    if len(file_list) == 1 and root_count > 1:
        return file_list * root_count
    if len(file_list) != root_count:
        raise ValueError(f"roots and file_list length mismatch: roots={root_count}, file_list={len(file_list)}")
    return file_list


def build_train_val_loaders(cfg):
    root_dir = str(cfg.DATASET.ROOT)

    train_roots = resolve_paths(root_dir, list(cfg.DATASET.TRAIN))
    test_roots = resolve_paths(root_dir, list(cfg.DATASET.TEST))
    train_files = expand_file_list(list(cfg.DATASET.TRAIN_LIST), len(train_roots))
    test_files = expand_file_list(list(cfg.DATASET.TEST_LIST), len(test_roots))

    train_dataset = NpySegmentationDataset(
        train_roots,
        train_files,
        aug=True,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
    )
    test_dataset = NpySegmentationDataset(
        test_roots,
        test_files,
        aug=False,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE),
        shuffle=True,
        num_workers=int(cfg.TRAIN.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE),
        shuffle=False,
        num_workers=int(cfg.TRAIN.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def build_models(cfg, device):
    model_dict = {
        "m_encoder": MEncoder(
            z_length=int(cfg.MODEL.Z_LENGTH),
            in_channel=int(cfg.MODEL.IMG_CH),
            img_size=int(cfg.MODEL.IMG_SIZE),
        ).to(device),
        "a_encoder": AEncoder(
            in_channel=int(cfg.MODEL.IMG_CH),
            width=int(cfg.MODEL.IMG_SIZE),
            height=int(cfg.MODEL.IMG_SIZE),
            ndf=int(cfg.MODEL.NDF),
            num_output_channels=int(cfg.MODEL.ANATOMY_CH),
            norm=str(cfg.MODEL.NORM),
            upsample=str(cfg.MODEL.UPSAMPLE),
        ).to(device),
        "segmentor": OIODSegmentor(
            num_output_channels=int(cfg.MODEL.ANATOMY_CH),
            num_class=int(cfg.MODEL.NUM_CLASSES),
        ).to(device),
        "decoder": Ada_Decoder(
            anatomy_out_channel=int(cfg.MODEL.ANATOMY_CH),
            z_length=int(cfg.MODEL.Z_LENGTH),
            out_channel=int(cfg.MODEL.IMG_CH),
        ).to(device),
    }

    for module in model_dict.values():
        initialize_weights(module, "xavier")

    return model_dict


def build_optimizers(models, cfg):
    lr = float(cfg.TRAIN.LR)
    weight_decay = float(cfg.TRAIN.WD)
    return {
        "m_optimizer": torch.optim.RMSprop(models["m_encoder"].parameters(), lr=lr, weight_decay=weight_decay),
        "a_optimizer": torch.optim.RMSprop(models["a_encoder"].parameters(), lr=lr, weight_decay=weight_decay),
        "s_optimizer": torch.optim.RMSprop(models["segmentor"].parameters(), lr=lr, weight_decay=weight_decay),
        "d_optimizer": torch.optim.RMSprop(models["decoder"].parameters(), lr=lr, weight_decay=weight_decay),
    }


def build_criterions():
    return {
        "seg_criterion": SoftDiceLoss(),
        "l1": torch.nn.L1Loss(),
    }


def get_loss_weights(cfg):
    return {
        "lambda_1": float(cfg.LOSS.LAMBDA_1),
        "lambda_2": float(cfg.LOSS.LAMBDA_2),
        "lambda_3": float(cfg.LOSS.LAMBDA_3),
        "lambda_4": float(cfg.LOSS.LAMBDA_4),
        "lambda_5": float(cfg.LOSS.LAMBDA_5),
        "lambda_6": float(cfg.LOSS.LAMBDA_6),
        "lambda_7": float(cfg.LOSS.LAMBDA_7),
        "lambda_8": float(cfg.LOSS.LAMBDA_8),
        "lambda_9": float(cfg.LOSS.LAMBDA_9),
    }


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1], None
    raise ValueError("Unexpected dataloader batch format")


def forward_a_encoder(a_encoder, images: torch.Tensor) -> torch.Tensor:
    """
    Compatibility wrapper for different AEncoder implementations.
    Accept either a tensor or a sequence whose final item is the anatomy tensor.
    """
    outputs = a_encoder(images)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise RuntimeError("AEncoder returned empty tuple/list")
        outputs = outputs[-1]
    if not torch.is_tensor(outputs):
        raise RuntimeError(f"AEncoder output type not supported: {type(outputs)}")
    return outputs


def forward_segmentor(segmentor, anatomy_feature: torch.Tensor) -> torch.Tensor:
    """
    Compatibility wrapper for segmentor outputs.
    """
    outputs = segmentor(anatomy_feature)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise RuntimeError("Segmentor returned empty tuple/list")
        outputs = outputs[-1]
    if not torch.is_tensor(outputs):
        raise RuntimeError(f"Segmentor output type not supported: {type(outputs)}")
    return outputs


def build_eval_prediction(prediction: torch.Tensor, cfg, num_class: int) -> np.ndarray:
    """
    Build post-processed prediction in [C,H,W] for metric computation.
    - threshold: apply sigmoid and channel-wise thresholding
    - argmax: use class-wise one-hot from softmax logits (stable for multi-class training)
    """
    eval_mode = str(getattr(cfg.OIOD, "EVAL_MODE", "threshold")).lower()
    if eval_mode == "argmax":
        pred_label = torch.argmax(prediction, dim=0).detach().cpu().numpy()
        pred_onehot = np.zeros((int(num_class), pred_label.shape[0], pred_label.shape[1]), dtype=np.uint8)
        for class_index in range(int(num_class)):
            pred_onehot[class_index] = (pred_label == class_index).astype(np.uint8)
        return pred_onehot

    return postprocessing(prediction, dataset=str(cfg.DATASET.NAME))


def compute_ema_mask(mask_tensor: torch.Tensor, kernel_size: int):
    mask_argmax = torch.argmax(mask_tensor, dim=1, keepdim=True)
    foreground = (mask_argmax != 0).float()
    kernel = torch.ones((1, 1, int(kernel_size), int(kernel_size)), dtype=torch.float32, device=mask_tensor.device)
    dilated = F.conv2d(foreground, kernel, stride=1, padding=int(kernel_size) // 2)
    return (dilated > 0).float()


def compute_loss_terms(images, masks, models, criterions, cfg):
    m_encoder = models["m_encoder"]
    a_encoder = models["a_encoder"]
    segmentor = models["segmentor"]
    decoder = models["decoder"]

    seg_criterion = criterions["seg_criterion"]
    l1_distance = criterions["l1"]

    batch_size = int(images.size(0))

    a_out = forward_a_encoder(a_encoder, images)
    seg_logits = forward_segmentor(segmentor, a_out)
    seg_prob = torch.softmax(seg_logits, dim=1)
    seg_loss = seg_criterion(seg_prob, masks, num_class=int(cfg.MODEL.NUM_CLASSES))

    z_out, mu_out, logvar_out = m_encoder(images)
    reco = decoder(a_out, z_out)
    z_out_tiled, _, _ = m_encoder(reco)

    if batch_size > 1:
        random_idx = torch.roll(torch.arange(batch_size, device=images.device), shifts=-1)
    else:
        random_idx = torch.arange(batch_size, device=images.device)

    z_out_random = z_out[random_idx, :]
    reco_random = decoder(a_out, z_out_random)
    z_out_tiled_random, _, _ = m_encoder(reco_random)

    a_out_random = forward_a_encoder(a_encoder, reco_random)
    seg_pred_random = torch.softmax(forward_segmentor(segmentor, a_out_random), dim=1)

    z_out_random1 = torch.rand(batch_size, int(cfg.MODEL.Z_LENGTH), device=images.device) * 2.0 - 1.0
    reco_random1 = decoder(a_out, z_out_random1)

    a_out_random1 = forward_a_encoder(a_encoder, reco_random1)
    seg_pred_random1 = torch.softmax(forward_segmentor(segmentor, a_out_random1), dim=1)

    masked = compute_ema_mask(masks, kernel_size=int(cfg.OIOD.KERNEL_SIZE))

    reco_loss = l1_distance(reco, images)
    reco_loss_masked = l1_distance(masked * reco, masked * images)
    recoz_loss = l1_distance(z_out_tiled, z_out)
    recoz_loss_random = l1_distance(z_out_tiled_random, z_out_random)

    a_con_loss = l1_distance(a_out_random, a_out)
    seg_con_loss = l1_distance(seg_pred_random, seg_prob)
    a_con_loss1 = l1_distance(a_out_random1, a_out)
    seg_con_loss1 = l1_distance(seg_pred_random1, seg_prob)

    kl_loss = KL_divergence(logvar_out, mu_out)

    weights = get_loss_weights(cfg)
    total_loss = (
        weights["lambda_1"] * seg_loss
        + weights["lambda_2"] * reco_loss
        + weights["lambda_3"] * reco_loss_masked
        + weights["lambda_4"] * recoz_loss
        + weights["lambda_5"] * recoz_loss_random
        + weights["lambda_6"] * a_con_loss
        + weights["lambda_7"] * seg_con_loss
        + weights["lambda_8"] * a_con_loss1
        + weights["lambda_9"] * seg_con_loss1
        + float(cfg.OIOD.KL_WEIGHT) * kl_loss
    )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Encountered non-finite total_loss. "
            f"seg={float(seg_loss.detach().item()):.6f}, "
            f"reco={float(reco_loss.detach().item()):.6f}, "
            f"reco_masked={float(reco_loss_masked.detach().item()):.6f}, "
            f"recoz={float(recoz_loss.detach().item()):.6f}, "
            f"recoz_rand={float(recoz_loss_random.detach().item()):.6f}, "
            f"a_con={float(a_con_loss.detach().item()):.6f}, "
            f"seg_con={float(seg_con_loss.detach().item()):.6f}, "
            f"a_con1={float(a_con_loss1.detach().item()):.6f}, "
            f"seg_con1={float(seg_con_loss1.detach().item()):.6f}, "
            f"kl={float(kl_loss.detach().item()):.6f}"
        )

    terms = {
        "seg_loss": float(seg_loss.detach().item()),
        "reco_loss": float(reco_loss.detach().item()),
        "reco_loss_masked": float(reco_loss_masked.detach().item()),
        "mask_ratio": float(masked.detach().mean().item()),
        "recoz_loss": float(recoz_loss.detach().item()),
        "recoz_loss_random": float(recoz_loss_random.detach().item()),
        "a_con_loss": float(a_con_loss.detach().item()),
        "seg_con_loss": float(seg_con_loss.detach().item()),
        "a_con_loss1": float(a_con_loss1.detach().item()),
        "seg_con_loss1": float(seg_con_loss1.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    return total_loss, terms


def train_one_epoch(train_loader, models, optimizers, criterions, cfg, device, logger=None, writer=None, epoch=0):
    models["m_encoder"].train()
    models["a_encoder"].train()
    models["segmentor"].train()
    models["decoder"].train()

    epoch_terms: Dict[str, float] = {}
    count = 0

    for step, batch in enumerate(train_loader):
        images, masks, _ = unpack_batch(batch)
        images = images.to(device)
        masks = masks.to(device)

        if torch.any((masks < 0) | (masks > 1)):
            if logger is not None:
                logger.warning(
                    f"Found out-of-range mask values at epoch {epoch}, step {step}; "
                    f"min={masks.min().item():.6f}, max={masks.max().item():.6f}. Clamping to [0, 1]."
                )
            masks = masks.clamp(0.0, 1.0)

        optimizers["m_optimizer"].zero_grad()
        optimizers["a_optimizer"].zero_grad()
        optimizers["s_optimizer"].zero_grad()
        optimizers["d_optimizer"].zero_grad()

        total_loss, terms = compute_loss_terms(images, masks, models, criterions, cfg)
        total_loss.backward()

        optimizers["m_optimizer"].step()
        optimizers["a_optimizer"].step()
        optimizers["s_optimizer"].step()
        optimizers["d_optimizer"].step()

        for key, value in terms.items():
            epoch_terms[key] = epoch_terms.get(key, 0.0) + float(value)
        count += 1

        if writer is not None:
            global_step = (int(epoch) - 1) * max(len(train_loader), 1) + step
            writer.add_scalar("train/loss_step", float(terms["total_loss"]), global_step)
            for term_name, term_value in terms.items():
                writer.add_scalar(f"train/{term_name}_step", float(term_value), global_step)

        if logger is not None and step % max(int(cfg.TRAIN.PRINT_FREQ), 1) == 0:
            logger.info(
                f"Epoch [{epoch}/{cfg.TRAIN.EPOCHS}] Step [{step}/{len(train_loader)}] "
                f"total={terms['total_loss']:.4f}, seg={terms['seg_loss']:.4f}, reco={terms['reco_loss']:.4f}, "
                f"reco_masked={terms['reco_loss_masked']:.4f}, mask_ratio={terms['mask_ratio']:.4f}, recoz={terms['recoz_loss']:.4f}, "
                f"recoz_rand={terms['recoz_loss_random']:.4f}, a_con={terms['a_con_loss']:.4f}, "
                f"seg_con={terms['seg_con_loss']:.4f}, a_con1={terms['a_con_loss1']:.4f}, "
                f"seg_con1={terms['seg_con_loss1']:.4f}, kl={terms['kl_loss']:.4f}"
            )

    if count > 0:
        for key in epoch_terms:
            epoch_terms[key] /= count

    if logger is not None and epoch_terms:
        logger.info(
            "Epoch summary: "
            + ", ".join([f"{name}={value:.4f}" for name, value in sorted(epoch_terms.items())])
        )

    return epoch_terms



def binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = float((pred * gt).sum())
    return (2.0 * intersection + 1.0) / (float(pred.sum()) + float(gt.sum()) + 1.0)


def safe_asd(pred_mask: np.ndarray, gt_mask: np.ndarray, default: float = 100.0):
    if binary is None:
        return default
    if np.sum(pred_mask) < 1e-4 or np.sum(gt_mask) < 1e-4:
        return default
    try:
        return float(
            binary.asd(
                np.asarray(pred_mask, dtype=np.bool_),
                np.asarray(gt_mask, dtype=np.bool_),
            )
        )
    except Exception:
        return default


def validate(test_loader, models, cfg, device, data_type: str = "optic"):
    models["a_encoder"].eval()
    models["segmentor"].eval()

    cup_scores: List[float] = []
    disc_scores: List[float] = []
    asd_oc_scores: List[float] = []
    asd_od_scores: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            images, masks, _ = unpack_batch(batch)
            images = images.to(device)
            masks = masks.to(device).clamp(0.0, 1.0)

            anatomy_feature = forward_a_encoder(models["a_encoder"], images)
            predictions = forward_segmentor(models["segmentor"], anatomy_feature)
            target_np = masks.cpu().numpy()

            for index in range(predictions.shape[0]):
                prediction_post = build_eval_prediction(
                    predictions[index],
                    cfg,
                    num_class=int(cfg.MODEL.NUM_CLASSES),
                )

                if data_type == "optic":
                    cup_dice, disc_dice = dice_coeff_2label(prediction_post, masks[index])
                    cup_scores.append(float(cup_dice))
                    disc_scores.append(float(disc_dice))

                    asd_oc_scores.append(float(safe_asd(prediction_post[2, ...], target_np[index, 2, ...])))
                    asd_od_scores.append(float(safe_asd(prediction_post[1, ...], target_np[index, 1, ...])))
                else:
                    prostate_dice, _ = dice_coeff_2label(prediction_post, masks[index])
                    disc_scores.append(float(prostate_dice))

                    pred_channel = 1 if prediction_post.shape[0] > 1 else 0
                    target_channel = 1 if target_np.shape[1] > 1 else 0
                    pred_seg = prediction_post[pred_channel, ...]
                    target_seg = target_np[index, target_channel, ...]

                    asd_od_scores.append(float(safe_asd(pred_seg, target_seg)))

    if data_type == "optic":
        val_cup_dice = float(np.mean(cup_scores)) if cup_scores else 0.0
        val_disc_dice = float(np.mean(disc_scores)) if disc_scores else 0.0
        total_asd_oc = float(np.mean(asd_oc_scores)) if asd_oc_scores else 100.0
        total_asd_od = float(np.mean(asd_od_scores)) if asd_od_scores else 100.0
        avg_f1 = (val_cup_dice + val_disc_dice) / 2.0
        return {
            "avg_f1": avg_f1,
            "val_cup_dice": val_cup_dice,
            "val_disc_dice": val_disc_dice,
            "total_asd_OC": total_asd_oc,
            "total_asd_OD": total_asd_od,
            "cup_list": cup_scores,
            "disc_list": disc_scores,
        }

    val_seg_dice = float(np.mean(disc_scores)) if disc_scores else 0.0
    total_asd_seg = float(np.mean(asd_od_scores)) if asd_od_scores else 100.0
    return {
        "avg_f1": val_seg_dice,
        "val_seg_dice": val_seg_dice,
        "total_asd_seg": total_asd_seg,
        "disc_list": disc_scores,
    }

def save_checkpoint(state: Dict, filename: str):
    torch.save(state, filename)


def serialize_cfg(cfg):
    if hasattr(cfg, "dump"):
        return cfg.dump()
    return str(cfg)


def make_checkpoint_state(models, optimizers, cfg, epoch: int, best_f1: float, best_epoch: int):
    return {
        "epoch": int(epoch),
        "m_encoder_state_dict": models["m_encoder"].state_dict(),
        "a_encoder_state_dict": models["a_encoder"].state_dict(),
        "segmentor_state_dict": models["segmentor"].state_dict(),
        "decoder_state_dict": models["decoder"].state_dict(),
        "m_optimizer_state_dict": optimizers["m_optimizer"].state_dict(),
        "a_optimizer_state_dict": optimizers["a_optimizer"].state_dict(),
        "s_optimizer_state_dict": optimizers["s_optimizer"].state_dict(),
        "d_optimizer_state_dict": optimizers["d_optimizer"].state_dict(),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "config": serialize_cfg(cfg),
    }


def load_checkpoint(weight_path: str, models, optimizers=None, map_location="cpu"):
    checkpoint = torch.load(weight_path, map_location=map_location)

    if isinstance(checkpoint, dict) and "m_encoder_state_dict" in checkpoint:
        models["m_encoder"].load_state_dict(checkpoint["m_encoder_state_dict"])
        models["a_encoder"].load_state_dict(checkpoint["a_encoder_state_dict"])
        models["segmentor"].load_state_dict(checkpoint["segmentor_state_dict"])
        models["decoder"].load_state_dict(checkpoint["decoder_state_dict"])

        if optimizers is not None:
            if "m_optimizer_state_dict" in checkpoint:
                optimizers["m_optimizer"].load_state_dict(checkpoint["m_optimizer_state_dict"])
            if "a_optimizer_state_dict" in checkpoint:
                optimizers["a_optimizer"].load_state_dict(checkpoint["a_optimizer_state_dict"])
            if "s_optimizer_state_dict" in checkpoint:
                optimizers["s_optimizer"].load_state_dict(checkpoint["s_optimizer_state_dict"])
            if "d_optimizer_state_dict" in checkpoint:
                optimizers["d_optimizer"].load_state_dict(checkpoint["d_optimizer_state_dict"])
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected keys: m_encoder_state_dict/a_encoder_state_dict/..."
        )
    return checkpoint


def write_json(file_path: str, content: Dict):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(content, file_obj, indent=2, ensure_ascii=False)
