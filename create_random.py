import pandas as pd
import numpy as np

n_entries = 100000

feature_dim = 512

output_file = "c:/Users/jonas/Desktop/random_numbers.csv"

path = "c:/Users/jonas/Desktop"


paths = [path] * n_entries

# Create random numbers
random_numbers = np.random.rand(n_entries, feature_dim).tolist()

df = pd.DataFrame({"path": paths, "features": random_numbers})
df.to_csv(output_file)

pass

