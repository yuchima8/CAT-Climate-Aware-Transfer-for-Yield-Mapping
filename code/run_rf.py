import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cat.rf import SETUPS, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", required=True, choices=sorted(SETUPS))
    parser.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "outputs"))
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()

    predictions, metrics = run(args.crop, args.data, args.seeds)
    os.makedirs(args.out, exist_ok=True)
    predictions.to_csv(os.path.join(args.out, f"rf_{args.crop}_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(args.out, f"rf_{args.crop}_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
