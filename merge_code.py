import pandas as pd
import glob

# Find all CSV files in the folder
csv_files = glob.glob('*.csv')

# Empty list to collect dataframes
dfs = []

# Read each CSV file and store it
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes into one big dataframe
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe
merged_df.to_csv('all_reviews.csv', index=False)

print(f"Merged {len(csv_files)} files into 'all_reviews.csv'!")
