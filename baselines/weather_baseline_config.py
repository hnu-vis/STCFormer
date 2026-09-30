"""Shared BasicTS configuration builder for the atmospheric benchmarks.

The eight datasets use the same training protocol.  Keeping that protocol in
one place prevents model configs from silently drifting apart.
"""

import os

from easydict import EasyDict

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mse, sedi_score
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings


WEATHER_DATASETS = {
    "French_Temperature_dim1": {"num_nodes": 234, "batch_size": 256},
    "French_U_Wind_dim1": {"num_nodes": 234, "batch_size": 256},
    "French_V_Wind_dim1": {"num_nodes": 234, "batch_size": 256},
    "HUNAN_Temperature_dim1": {"num_nodes": 3570, "batch_size": 64},
    "HUNAN_U_Wind_dim1": {"num_nodes": 3570, "batch_size": 64},
    "HUNAN_V_Wind_dim1": {"num_nodes": 3570, "batch_size": 64},
    "Global_Temperature_dim1": {"num_nodes": 3850, "batch_size": 64},
    "Global_Wind_dim1": {"num_nodes": 3850, "batch_size": 64},
}


def build_weather_cfg(data_name, model_arch, model_params, *, input_len=48,
                      output_len=None, num_epochs=30, learning_rate=3e-4):
    """Build a complete, independent EasyDict config for one model/dataset."""
    spec = WEATHER_DATASETS[data_name]
    regular = get_regular_settings(data_name)
    output_len = int(os.environ.get("WEATHER_OUTPUT_LEN", output_len or 24))
    if output_len not in (24, 72):
        raise ValueError(f"WEATHER_OUTPUT_LEN must be 24 or 72, got {output_len}")
    seed = int(os.environ.get("WEATHER_SEED", 2024))
    num_epochs = int(os.environ.get("WEATHER_EPOCHS", num_epochs))
    params = model_params(
        data_name=data_name,
        num_nodes=spec["num_nodes"],
        input_len=input_len,
        output_len=output_len,
    )

    cfg = EasyDict()
    cfg.DESCRIPTION = f"{model_arch.__name__} atmospheric forecasting on {data_name}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner
    cfg.ENV = EasyDict({
        "SEED": seed,
        "TF32": True,
        "DETERMINISTIC": True,
        "CUDNN": EasyDict({
            "ENABLED": True,
            "BENCHMARK": False,
            "DETERMINISTIC": True,
        }),
    })

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_val_test_ratio": regular["TRAIN_VAL_TEST_RATIO"],
        "input_len": input_len,
        "output_len": output_len,
    })

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_ratio": regular["TRAIN_VAL_TEST_RATIO"][0],
        "norm_each_channel": regular["NORM_EACH_CHANNEL"],
        "rescale": regular["RESCALE"],
    })

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = model_arch.__name__
    cfg.MODEL.ARCH = model_arch
    cfg.MODEL.PARAM = params
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict({
        "MAE": masked_mae,
        "MSE": masked_mse,
        "SEDI": sedi_score,
    })
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = regular["NULL_VAL"]
    cfg.METRICS.SEDI_THRESHOLD = regular["SEDI_THRESHOLD"]

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints", "weather_48_24_72_30e", model_arch.__name__, data_name,
        f"h{output_len}", f"seed{seed}",
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict({"TYPE": "AdamW", "PARAM": {
        "lr": learning_rate, "weight_decay": 1e-4,
    }})
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {
        "milestones": [15, 25], "gamma": 0.5,
    }})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    data_loader = EasyDict({
        "BATCH_SIZE": int(os.environ.get("WEATHER_BATCH_SIZE", spec["batch_size"])),
        "NUM_WORKERS": int(os.environ.get("WEATHER_NUM_WORKERS", 0)),
        "PIN_MEMORY": True,
    })
    cfg.TRAIN.DATA = EasyDict(data_loader)
    cfg.TRAIN.DATA.SHUFFLE = True

    cfg.VAL = EasyDict({"INTERVAL": 1, "DATA": EasyDict(data_loader)})
    # BaseEpochRunner always evaluates the best checkpoint after training.  An
    # interval beyond NUM_EPOCHS avoids duplicating the full test at epoch 30.
    cfg.TEST = EasyDict({"INTERVAL": num_epochs + 1, "DATA": EasyDict(data_loader)})
    cfg.EVAL = EasyDict({"USE_GPU": False, "SAVE_RESULTS": False})
    return cfg
