import pandas as pd
import os

# Define the path of the input CSV file
input_file_path = 'C:/DataRec/To_reconcile/clean_data_lagrangian/reconciled_data.csv'
output_folder = 'C:/DataRec/To_reconcile/clean_data_lagrangian/Reconciled'  # Ensure this folder exists

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Extract the Time column
time_column = df['Time']

# List of Reconciled F columns
reconciled_columns = [col for col in df.columns if col.startswith('Reconciled F')]

# Iterate through each Reconciled F column and save as separate CSV
for column in reconciled_columns:
    output_df = pd.DataFrame({'Time': time_column, column: df[column]})
    output_file_path = os.path.join(output_folder, f'{column.replace("Reconciled ", "")}.csv')
    output_df.to_csv(output_file_path, index=False)

print("Files have been saved successfully.")
