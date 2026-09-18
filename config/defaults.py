from yacs.config import CfgNode as CN


_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = "234to1"
_C.DATASET.ROOT = "./npy_data"
_C.DATASET.TRAIN = ["D4/train", "D2/train", "D3/train"]
_C.DATASET.TEST = ["D1/test"]
_C.DATASET.TRAIN_LIST = ["filenames.txt"]
_C.DATASET.TEST_LIST = ["filenames.txt"]

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 200
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 1e-4
_C.TRAIN.WD = 1e-4
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.SEED = 42
_C.TRAIN.VAL_FREQ = 1

_C.MODEL = CN()
_C.MODEL.IMG_CH = 3
_C.MODEL.NUM_CLASSES = 3
_C.MODEL.NUM_FILTERS = 16
_C.MODEL.IMG_SIZE = 256
_C.MODEL.NDF = 16
_C.MODEL.ANATOMY_CH = 8
_C.MODEL.Z_LENGTH = 48
_C.MODEL.NORM = "batchnorm"
_C.MODEL.UPSAMPLE = "bilinear"

_C.LOSS = CN()
_C.LOSS.LAMBDA_1 = 5.0
_C.LOSS.LAMBDA_2 = 5.0
_C.LOSS.LAMBDA_3 = 10.0
_C.LOSS.LAMBDA_4 = 1.0
_C.LOSS.LAMBDA_5 = 5.0
_C.LOSS.LAMBDA_6 = 1.0
_C.LOSS.LAMBDA_7 = 1.0
_C.LOSS.LAMBDA_8 = 1.0
_C.LOSS.LAMBDA_9 = 1.0

_C.OIOD = CN()
_C.OIOD.KERNEL_SIZE = 21
_C.OIOD.THRESHOLD = 0.75
_C.OIOD.EVAL_MODE = "argmax"
_C.OIOD.KL_WEIGHT = 0.0

_C.LOG = CN()
_C.LOG.LOG_DIR = "logs/"
_C.LOG.LOG_FILE = "training_log.log"


def get_cfg_defaults():
    return _C.clone()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    print(cfg)
