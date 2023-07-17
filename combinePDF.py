import pandas as pd

# Read the CSV files into pandas dataframes
df1 = pd.read_csv('yuval_features.csv')
df2 = pd.read_csv('matan_features.csv')

# Concatenate the dataframes vertically
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined dataframe to a new CSV file
combined_df.to_csv('yuval_and_matan_features.csv', index=False)