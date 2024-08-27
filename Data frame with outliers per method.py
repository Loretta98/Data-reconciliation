import os
import pandas as pd
import glob

# Path to the CSV files
path = os.path.join('C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation')

# Get a list of all CSV files in the directory
files = glob.glob(os.path.join(path, "*.csv"))

# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Process the first file to initialize the DataFrame
if files:
    first_file = files.pop(0)
    print(f"Processing file: {first_file}")
    file_basename = os.path.basename(first_file).split('.')[0]
    
    # Read the first CSV file
    df = pd.read_csv(first_file)
    
    # Initialize merged_df with the 'Time' column (second column)
    merged_df['Time'] = df.iloc[:, 1]
    
    # Add the third column of the first file
    merged_df[file_basename] = df.iloc[:, 2]

# Process each remaining file
for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Add the third column to merged_df
    merged_df[file_basename] = df.iloc[:, 2]

# Create the new directory 'merged_data' if it doesn't exist
output_dir = os.path.join(path, 'merged_data')
os.makedirs(output_dir, exist_ok=True)

# Save the merged DataFrame to a new CSV file in the 'merged_data' folder
merged_df.to_csv(os.path.join(output_dir, 'merged_data.csv'), index=False)

print("Merging completed and saved as 'merged_data/merged_data.csv'")
