import dbm
from ast import literal_eval
import os

import pandas as pd
import numpy as np

import argparse


def csv_to_data(csv_file, output_folder):
    print("Reading csv file...")
    df = pd.read_csv(csv_file)
    print("Done reading csv file.")

    print("String to float...")
    image_features = []
    for i, feat in enumerate(df["features"]):
        if i % 1000 == 0:
            print(f"{i}/{len(df['features'])}")
        image_features.append(literal_eval(feat))


    #image_features = [literal_eval(f) for f in df["features"]]
    image_features = np.array(image_features)
    np.save(os.path.join(output_folder, "image_features"), image_features)

    # Open database, creating it if necessary.
    with dbm.open(os.path.join(output_folder, "database"), 'c') as db:
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"{i}/{len(df)}")
            db[str(i)] = row.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="Path to the csv file")
    parser.add_argument("--output_folder", help="Path to the output folder", default=".")
    args = parser.parse_args()
    csv_to_data(args.csv_file, args.output_folder)
