# File created on 17.07.2024 
#################### RAW DATA MANIPULATION #####################################
# The data given from the PLC isn't necessaily extracted with the correct time sequence 
# This file manipulates such data to recognize the correct time order 


import os
import pandas as pd

def reorder_csv_files(input_dir):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "Ordered CSV")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert "Time String" column to datetime format
        df['Time String'] = pd.to_datetime(df['TimeString'], format='%d.%m.%Y %H:%M:%S')
        
        # Sort the DataFrame by the "Time String" column
        df_sorted = df.sort_values(by='Time String')
        
        # Convert "Time String" column back to the original format
        df_sorted['Time String'] = df_sorted['Time String'].dt.strftime('%d.%m.%Y %H:%M:%S')
        
        # Save the sorted DataFrame to a new CSV file in the output directory
        output_file_path = os.path.join(output_dir, file)
        df_sorted.to_csv(output_file_path, index=False)
        print(f'Sorted file saved to {output_file_path}')

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024'
reorder_csv_files(input_directory)
