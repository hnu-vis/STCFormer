"""Unified configuration for the remaining atmospheric baselines.

The launcher selects a model and dataset through ``WEATHER_MODEL`` and
``WEATHER_DATASET``.  Keeping this matrix in one module avoids duplicating 80
near-identical config files and, more importantly, guarantees that every model
uses the same 48-step input, 24/72-step output, 30 epochs and evaluation
metrics.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from easydict import EasyDict

from baselines.weather_baseline_config import build_weather_cfg
from basicts.utils import load_adj


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("WEATHER_MODEL", "DLinear")
DATA_NAME = os.environ.get("WEATHER_DATASET", "French_Temperature_dim1")
INPUT_LEN = 48


def _dcrnn_params(data_name, num_nodes, input_len, output_len):
    adjacency, _ = load_adj(
        str(ROOT / "datasets" / data_name / "adj_mx.pkl"),
        "doubletransition",
    )
    return {
        "cl_decay_steps": 2000,
        "horizon": output_len,
        "input_dim": 1,
        "max_diffusion_step": 2,
        "num_nodes": num_nodes,
        "num_rnn_layers": 1,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": input_len,
        # Weather graphs contain only about 1% non-zero entries at 3.5k+ nodes.
        # Sparse supports preserve the exact diffusion operator while avoiding
        # quadratic dense graph multiplication and storage.
        "adj_mx": [torch.as_tensor(item).float().to_sparse().coalesce()
                   for item in adjacency],
        "use_curriculum_learning": True,
    }


def _corrformer_params(data_name, num_nodes, input_len, output_len):
    node_lists = {
        234: [13, 18],
        3570: [10, 357],
        3850: [10, 385],
    }
    # Corrformer's decoder keeps several B x N x (label+pred) x D tensors for
    # backward.  At 3570/3850 stations, D=256 exceeds a 24 GiB card even with
    # batch size one, so use the paper architecture at a width that is viable
    # for these much larger station sets.
    d_model = 128 if num_nodes > 1000 else 256
    d_ff = 256 if num_nodes > 1000 else 512
    return {
        "target_features": [0],
        "seq_len": input_len,
        "label_len": input_len // 2,
        "pred_len": output_len,
        "node_num": num_nodes,
        "node_list": node_lists[num_nodes],
        "moving_avg": 25,
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,
        "d_model": d_model,
        "n_heads": 4,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": d_ff,
        "root_path": str(ROOT / "datasets" / data_name),
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.1,
        "factor_temporal": 1,
        "factor_spatial": 1,
        "enc_tcn_layers": 1,
        "dec_tcn_layers": 1,
        "output_attention": False,
        "activation": "gelu",
    }


def _dlinear_params(data_name, num_nodes, input_len, output_len):
    del data_name
    return {
        "seq_len": input_len,
        "pred_len": output_len,
        "individual": False,
        "enc_in": num_nodes,
    }


def _patchtst_params(data_name, num_nodes, input_len, output_len):
    del data_name
    return {
        "enc_in": num_nodes,
        "seq_len": input_len,
        "pred_len": output_len,
        "e_layers": 1,
        "n_heads": 4,
        "d_model": 256,
        "d_ff": 512,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0.0,
        "patch_len": 6,
        "stride": 3,
        "individual": False,
        "padding_patch": "end",
        "revin": True,
        "affine": False,
        "subtract_last": False,
        "decomposition": False,
        "kernel_size": 25,
    }


def _crossformer_params(data_name, num_nodes, input_len, output_len):
    del data_name
    return {
        "data_dim": num_nodes,
        "in_len": input_len,
        "out_len": output_len,
        "seg_len": 8,
        "win_size": 2,
        "factor": 10,
        "d_model": 256,
        "d_ff": 512,
        "n_heads": 4,
        "e_layers": 1,
        "dropout": 0.2,
        "baseline": False,
    }


def _easyst_params(data_name, num_nodes, input_len, output_len):
    del data_name
    return {
        "num_nodes": num_nodes,
        "seq_len": input_len,
        "pred_len": output_len,
        "d_model": 256,
        "d_ff": 512,
        "node_dim": 64,
        "time_dim": 64,
        "num_layers": 3,
        "dropout": 0.1,
    }


def _timexer_params(data_name, num_nodes, input_len, output_len):
    del data_name
    return {
        "target_features": [0],
        "enc_in": num_nodes,
        "dec_in": num_nodes,
        "c_out": num_nodes,
        "seq_len": input_len,
        "pred_len": output_len,
        "patch_len": 6,
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 1,
        "factor": 3,
        "d_ff": 512,
        "dropout": 0.1,
        "freq": "h",
        "use_norm": True,
        "embed": "timeF",
        "activation": "gelu",
        "num_time_features": 4,
        "time_of_day_size": 24,
        "day_of_week_size": 7,
        "day_of_month_size": 31,
        "day_of_year_size": 366,
    }


def _duet_params(data_name, num_nodes, input_len, output_len):
    del data_name
    config = EasyDict({
        "CI": 1,
        "enc_in": num_nodes,
        "seq_len": input_len,
        "factor": 3,
        "dropout": 0.1,
        "output_attention": False,
        "d_model": 256,
        "n_heads": 4,
        "d_ff": 512,
        "activation": "relu",
        "e_layers": 1,
        "pred_len": output_len,
        "fc_dropout": 0.1,
        "noisy_gating": True,
        "num_experts": 4,
        "k": 2,
        "hidden_size": 256,
        "moving_avg": 13,
    })
    return {"config": config}


def _timefilter_params(data_name, num_nodes, input_len, output_len):
    del data_name
    # A larger temporal patch substantially reduces TimeFilter's LxL mask while
    # preserving its patch-specific graph filtering mechanism.
    patch_len = 12 if num_nodes > 1000 else 8
    patch_len = int(os.environ.get("WEATHER_TIMEFILTER_PATCH", patch_len))
    if patch_len < 1 or input_len % patch_len:
        raise ValueError("TimeFilter patch length must divide input_len")
    return {
        "configs": EasyDict({
            "task_name": "long_term_forecast",
            "seq_len": input_len,
            "pred_len": output_len,
            "c_out": num_nodes,
            "d_model": 256,
            "d_ff": 512,
            "patch_len": patch_len,
            "alpha": 0.1,
            "top_p": 0.5,
            "n_heads": 4,
            "e_layers": 1,
            "dropout": 0.1,
            "enc_in": num_nodes,
            "pos": True,
        })
    }


def _cdpnet_params(data_name, num_nodes, input_len, output_len):
    if num_nodes == 234:
        grid_h, grid_w = 12, 16
    else:
        grid_h, grid_w = 20, 24
    return {
        "num_nodes": num_nodes,
        "seq_len": input_len,
        "pred_len": output_len,
        "position_path": str(ROOT / "datasets" / data_name / "stations_sorted_reduced.npy"),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "d_model": 256,
        "d_ff": 512,
        "n_neighbors": 4,
        "dropout": 0.1,
    }


if MODEL_NAME == "S2Transformer":
    from baselines.S2Transformer.arch import S2Transformer as MODEL_ARCH

    def MODEL_PARAMS(data_name, num_nodes, input_len, output_len):
        return {
            "num_nodes": num_nodes, "input_len": input_len,
            "output_len": output_len, "root_path": str(ROOT / "datasets" / data_name),
            "d_model": 256, "d_ff": 512, "n_heads": 8,
            "num_layers": 2, "num_parts": 32, "dropout": 0.1,
        }
elif MODEL_NAME == "DCRNN":
    from baselines.DCRNN.arch import DCRNN as MODEL_ARCH
    MODEL_PARAMS = _dcrnn_params
elif MODEL_NAME == "Corrformer":
    from baselines.Corrformer.arch import Corrformer as MODEL_ARCH
    MODEL_PARAMS = _corrformer_params
elif MODEL_NAME == "DLinear":
    from baselines.DLinear.arch import DLinear as MODEL_ARCH
    MODEL_PARAMS = _dlinear_params
elif MODEL_NAME == "PatchTST":
    from baselines.PatchTST.arch import PatchTST as MODEL_ARCH
    MODEL_PARAMS = _patchtst_params
elif MODEL_NAME == "Crossformer":
    from baselines.Crossformer.arch import Crossformer as MODEL_ARCH
    MODEL_PARAMS = _crossformer_params
elif MODEL_NAME == "EasyST":
    from baselines.EasyST.arch import EasyST as MODEL_ARCH
    MODEL_PARAMS = _easyst_params
elif MODEL_NAME == "TimeXer":
    from baselines.TimeXer.arch import TimeXer as MODEL_ARCH
    MODEL_PARAMS = _timexer_params
elif MODEL_NAME == "DUET":
    from baselines.DUET.models.duet_model import DUETModel as MODEL_ARCH
    MODEL_PARAMS = _duet_params
elif MODEL_NAME == "TimeFilter":
    from baselines.TimeFilter.arch import TimeFilter as MODEL_ARCH
    MODEL_PARAMS = _timefilter_params
elif MODEL_NAME == "CDPNet":
    from baselines.CDPNet.arch import CDPNet as MODEL_ARCH
    MODEL_PARAMS = _cdpnet_params
else:
    raise ValueError(f"Unsupported WEATHER_MODEL: {MODEL_NAME}")


LEARNING_RATES = {
    "S2Transformer": 3e-4,
    "DCRNN": 3e-3,
    "Corrformer": 2e-4,
    "DLinear": 1e-3,
    "PatchTST": 1e-3,
    "Crossformer": 2e-4,
    "EasyST": 1e-3,
    "TimeXer": 1e-4,
    "DUET": 2e-4,
    "TimeFilter": 2e-4,
    "CDPNet": 1e-4,
}

CFG = build_weather_cfg(
    DATA_NAME,
    MODEL_ARCH,
    MODEL_PARAMS,
    input_len=INPUT_LEN,
    num_epochs=30,
    learning_rate=LEARNING_RATES[MODEL_NAME],
)

# Use stable paper-facing names (not implementation class names such as
# DUETModel) in checkpoints and aggregated result paths.
seed = int(os.environ.get("WEATHER_SEED", 2024))
horizon = int(os.environ.get("WEATHER_OUTPUT_LEN", 24))
CFG.DESCRIPTION = f"{MODEL_NAME} atmospheric forecasting on {DATA_NAME}"
CFG.MODEL.NAME = MODEL_NAME
CFG.TRAIN.CKPT_SAVE_DIR = str(
    Path("checkpoints") / "weather_48_24_72_30e" / MODEL_NAME /
    DATA_NAME / f"h{horizon}" / f"seed{seed}"
)

forward_features = [0, 1, 2, 3, 4] if MODEL_NAME in {
    "Corrformer", "TimeXer", "EasyST", "CDPNet", "S2Transformer"
} else [0]
CFG.MODEL.FORWARD_FEATURES = forward_features
CFG.MODEL.TARGET_FEATURES = [0]

batch_size = int(os.environ.get("WEATHER_BATCH_SIZE", CFG.TRAIN.DATA.BATCH_SIZE))
num_workers = int(os.environ.get("WEATHER_NUM_WORKERS", 2))
for split in (CFG.TRAIN, CFG.VAL, CFG.TEST):
    split.DATA.BATCH_SIZE = batch_size
    split.DATA.NUM_WORKERS = num_workers
    split.DATA.PIN_MEMORY = True

if MODEL_NAME == "DCRNN":
    # DCRNN creates some graph-dependent weights on its first forward pass.
    CFG.MODEL.SETUP_GRAPH = True
    CFG._ = random.randint(-1_000_000, 1_000_000)
