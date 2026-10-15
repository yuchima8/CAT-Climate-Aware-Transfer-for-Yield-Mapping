import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cat.dann import run

DANN_CONFIG = {
    "seeds": [940, 941, 942, 943, 944],
    "encoder_dims": (128, 128),
    "head_dims": (64,),
    "dropout": 0.3,
    "lambda_domain": 1.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 500,
    "batch_size": 512,
    "lr_step": 150,
    "val_fraction": 0.0,
    "drop_last": True,
    "alpha_per_step": False,
    "joint_forward": True,
    "reshuffle_target_each_step": True,
    "scaling_dtype": np.float64,
}

CONFIGS = {
    "soybean": dict(DANN_CONFIG),
    "corn": dict(DANN_CONFIG),
    "winter_wheat": dict(DANN_CONFIG),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "outputs"))
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    cfg = CONFIGS[args.crop]
    predictions, metrics = run(args.crop, args.data, args.seeds or cfg["seeds"], cfg)
    os.makedirs(args.out, exist_ok=True)
    predictions.to_csv(os.path.join(args.out, f"predictions_DANN_{args.crop}.csv"), index=False)
    metrics.to_csv(os.path.join(args.out, f"metrics_DANN_{args.crop}.csv"), index=False)


if __name__ == "__main__":
    main()
