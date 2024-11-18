import pandas as pd
import numpy as np

# Load data into DataFrame
df = pd.read_csv("train_.csv")

# Function to sample up to 5 rows per location_id
def sample_by_location(df, max_sample_size=1000):
    # Apply sampling per group of 'location_id'
    sampled_df = df.groupby("location_id", group_keys=False).apply(lambda x: x.sample(n=min(len(x), max_sample_size)))
    return sampled_df

# Generate sampled DataFrame
sampled_df = sample_by_location(df)

# Export to CSV (here, simulated as a string to show results)
sampled_df.to_csv("sampled_by_location.csv", index=False)
sampled_df
