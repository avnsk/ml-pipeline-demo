import argparse
from pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser("ML Pipeline Demo")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = Pipeline(args.data, args.target)
    pipeline.run()


if __name__ == "__main__":
    main()
