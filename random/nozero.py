import pandas as pd

# Load the CSV file
file_path = r'C:\DataRec\ft03_acqua.csv'
df = pd.read_csv(file_path)

# Filter out rows where VarValue equals 0
df_filtered = df[df['VarValue'] != 0]

# Save the filtered data to a new CSV file
output_file_path = r'C:\DataRec\ft03_acqua_filtered.csv'
df_filtered.to_csv(output_file_path, index=False)

print(f"Filtered data has been saved to {output_file_path}")
