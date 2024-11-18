import pandas as pd

# Loading data into a DataFrame
df = pd.read_csv("./val.csv")

# Function to add the `image_path` column
def add_image_path(df, split="train"):
    df["image_path"] = df.apply(
        lambda row: f"{split}/{row['category']}/{row['category']}_{row['location_id']}/{row['category']}_{row['location_id']}_{row['image_id']}.tif", axis=1
    )
    return df

# Applying function to add image_path
df_with_path = add_image_path(df, "val")
df_with_path.to_csv("val_.csv")
