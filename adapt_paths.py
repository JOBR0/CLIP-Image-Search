import pandas as pd
import argparse


def adapt(in_csv, out_csv, in_prefix, out_prefix):
    in_df = pd.read_csv(in_csv)

    new_paths = []

    for i, row in in_df.iterrows():
        if i % 100 == 0:
            print(f"{i}/{len(in_df)}")
        new_paths.append(row['path'].replace(in_prefix, out_prefix))


    in_df["path"] = new_paths

    in_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adapt paths in csv file.')
    parser.add_argument('--in_csv', type=str, help='Input csv file.')
    parser.add_argument('--out_csv', type=str, help='Output csv file.')
    parser.add_argument('--in_prefix', type=str, help='Input prefix.')
    parser.add_argument('--out_prefix', type=str, help='Output prefix.')
    args = parser.parse_args()

    adapt(args.in_csv, args.out_csv, args.in_prefix, args.out_prefix)
