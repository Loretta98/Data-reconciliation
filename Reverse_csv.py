import pandas as pd

# Path to your problematic CSV file
input_file_path = 'C:/DataRec/To_reconcile/GC_filtered/F5_f.csv'
output_file_path = 'C:/DataRec/To_reconcile/GC_filtered/F5.csv'

# Read the CSV file
df = pd.read_csv(input_file_path)

# Add a new column 'H2O [%Vol]' with default value of 0 for all rows
df['H2O [%Vol]'] = 0

# Save the modified DataFrame back to a CSV
df.to_csv(output_file_path, index=False)

# Save the DataFrame back to a valid CSV
df.to_csv(output_file_path, index=False)
print("File has been transposed and saved successfully.")
