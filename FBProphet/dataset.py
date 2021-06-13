import config 
import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv(config.TRAINING_FILE_PATH)

# Lowercase column names
df.columns = [col.lower() for col in df.columns]

# Make `date` column in `datetime` format
df['date'] = pd.to_datetime(df['date'])
df['ds'] = df['date']

# Column for Active Cases
df['active'] = df.confirmed - (df.recovered + df.deceased)
# df['active'] = np.log1p(df['active'])

df['y'] = df['active']


# Create dataset with total cases in India
df2 = df.loc[df.state == "India"]
df2 = df2[["ds", "y"]]
# df2.set_index("ds", inplace = True)

# print(df2.head())