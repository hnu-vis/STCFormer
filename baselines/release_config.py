"""French weather benchmark: 48 observations to 24 or 72 forecasts."""

import os
from pathlib import Path

from baselines.weather_baseline_config import build_weather_cfg


ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("STC_MODEL", "STCFormer")
DATASET = os.environ.get("STC_DATASET", "French_Temperature_dim1")
HORIZON = int(os.environ.get("STC_HORIZON", "24"))
SEED = int(os.environ.get("STC_SEED", "2024"))
EPOCHS = int(os.environ.get("STC_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("STC_BATCH_SIZE", "32"))

if DATASET not in {
    "French_Temperature_dim1", "French_U_Wind_dim1", "French_V_Wind_dim1"
}:
    raise ValueError(f"Unsupported dataset: {DATASET}")
if MODEL not in {"STCFormer", "DLinear", "PatchTST", "TimeXer"}:
    raise ValueError(f"Unsupported model: {MODEL}")
if HORIZON not in (24, 72):
    raise ValueError("Horizon must be 24 or 72")


def model_and_params(dataset, nodes, history, horizon):
    if MODEL == "STCFormer":
        from baselines.STCFormer.arch import STCFormer

        return STCFormer, {
            "num_n": nodes,
            "in_len": history,
            "out_len": horizon,
            "time_patch_len": 8,
            "spatial_patch_len": 2,
            "centroids_num": [10, 10, 10],
            "d_model": 192,
            "d_ff": 384,
            "n_heads": 4,
            "layers": 3,
            "dropout": 0.1,
            "root_path": str(ROOT / "datasets" / dataset),
            "geographic_bands": 12,
            "time_feature_dim": 0,
            "revin_flag": True,
            "em_steps": 1,
            "cluster_temperature": 0.35,
            "cluster_balance": 0.0,
            "centroid_blend": 0.95,
            "gumbel_tau": 1.0,
            "use_cluster_loss": True,
            "cluster_loss_weight": 0.1,
            "anneal_epochs": 5,
        }
    if MODEL == "DLinear":
        from baselines.DLinear.arch import DLinear

        return DLinear, {
            "seq_len": history, "pred_len": horizon,
            "individual": False, "enc_in": nodes,
        }
    if MODEL == "PatchTST":
        from baselines.PatchTST.arch import PatchTST

        return PatchTST, {
            "enc_in": nodes, "seq_len": history, "pred_len": horizon,
            "e_layers": 1, "n_heads": 4, "d_model": 256, "d_ff": 512,
            "dropout": 0.2, "fc_dropout": 0.2, "head_dropout": 0.0,
            "patch_len": 6, "stride": 3, "individual": False,
            "padding_patch": "end", "revin": True, "affine": False,
            "subtract_last": False, "decomposition": False, "kernel_size": 25,
        }
    from baselines.TimeXer.arch import TimeXer

    return TimeXer, {
        "target_features": [0], "enc_in": nodes, "dec_in": nodes,
        "c_out": nodes, "seq_len": history, "pred_len": horizon,
        "patch_len": 6, "d_model": 256, "n_heads": 8,
        "e_layers": 1, "factor": 3, "d_ff": 512,
        "dropout": 0.1, "freq": "h", "use_norm": True,
        "embed": "timeF", "activation": "gelu",
        "num_time_features": 4, "time_of_day_size": 24,
        "day_of_week_size": 7, "day_of_month_size": 31,
        "day_of_year_size": 366,
    }


architecture, params = model_and_params(DATASET, 234, 48, HORIZON)
learning_rates = {
    "STCFormer": 3e-4,
    "DLinear": 1e-3,
    "PatchTST": 1e-3,
    "TimeXer": 1e-4,
}
os.environ["WEATHER_OUTPUT_LEN"] = str(HORIZON)
os.environ["WEATHER_SEED"] = str(SEED)
CFG = build_weather_cfg(
    DATASET, architecture, lambda **_: params,
    input_len=48, output_len=HORIZON, num_epochs=EPOCHS,
    learning_rate=learning_rates[MODEL],
)
CFG.MODEL.NAME = MODEL
CFG.DESCRIPTION = f"{MODEL}: {DATASET}, 48 to {HORIZON}"
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4] if MODEL in {"STCFormer", "TimeXer"} else [0]
CFG.TRAIN.CKPT_SAVE_DIR = str(
    Path("checkpoints") / MODEL / DATASET / f"h{HORIZON}" / f"seed{SEED}"
)
for split in (CFG.TRAIN, CFG.VAL, CFG.TEST):
    split.DATA.BATCH_SIZE = BATCH_SIZE
    split.DATA.NUM_WORKERS = 0
