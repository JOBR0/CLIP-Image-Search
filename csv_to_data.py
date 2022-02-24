import dbm
from ast import literal_eval
import os

import pandas as pd
import numpy as np


csv_file = "C:/Users/Jonas/Desktop/reps.csv"
output_folder = "."


df = pd.read_csv(csv_file)


image_features = [literal_eval(f) for f in df["features"]]
image_features = np.array(image_features)
np.save(os.path.join(output_folder, "image_features"), image_features)

# Open database, creating it if necessary.
with dbm.open(os.path.join(output_folder, "database"), 'c') as db:
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"{i}/{len(df)}")
        db[str(i)] = row.path

