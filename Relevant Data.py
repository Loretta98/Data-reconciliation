import numpy as np 
import pandas as pd 
import os 

def cut_csv_files(input_dir):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "Partial CSV")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert "Time String" column to datetime format
        df['Time String'] = pd.to_datetime(df['Time String'], format='%d.%m.%Y %H:%M:%S')
        
        # Filter the DataFrame to only include rows from 29th May onwards
        cutoff_date = pd.Timestamp('29-05-2024')  # Adjust the year accordingly
        df_filtered = df[df['Time String'] >= cutoff_date]
        
        # Save the filtered DataFrame to a new CSV file in the output directory
        output_file_path = os.path.join(output_dir, file)
        df_filtered.to_csv(output_file_path, index=False)
        print(f'Filtered file saved to {output_file_path}')

# Example usage
input_directory = 'C:/DataRec'  # Note the corrected slashes
cut_csv_files(input_directory)
