import numpy as np 
import pandas as pd 
import os 


# Code adjusted for the GC data format 

def cut_csv_files(input_dir):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "Partial CSV")
    os.makedirs(output_dir, exist_ok=True)

    # Get list of CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        
        print(file_path)
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Strip any potential leading/trailing spaces in the 'Injection Time' column
        df['Injection Time'] = df['Injection Time'].str.strip()

        # Convert "Time String" column to datetime format
        #df['Time String'] = pd.to_datetime(df['Time String'], format='%d.%m.%Y %H:%M:%S')
        
        df['Injection Time'] = pd.to_datetime(df['Injection Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        # Check for invalid entries
        invalid_entries = df['Injection Time'].isna().sum()
        print(f"Invalid date entries found: {invalid_entries}")

        # Display some of the entries to manually verify
        print(df['Injection Time'].head(10))

        if invalid_entries > 0:
            print("There might be an issue with some date formats or entries.")
        else:
            print("Date conversion successful!")
        # Check if data exists after conversion
        print("Earliest date in data:", df['Injection Time'].min())
        print("Latest date in data:", df['Injection Time'].max())

        # Filter the DataFrame to only include rows from 29th May onwards
        #cutoff_date = pd.Timestamp('29-05-2024')  # Adjust the year accordingly
        cutoff_date = pd.Timestamp('2024-05-29 17:02:00')  # Adjust the year accordingly
        df_filtered = df[df['Injection Time'] >= cutoff_date]
        
        # Save the filtered DataFrame to a new CSV file in the output directory
        output_file_path = os.path.join(output_dir, file)
        df_filtered.to_csv(output_file_path, index=False)
        print(f"Number of rows after filtering: {len(df_filtered)}")
        print(f'Filtered file saved to {output_file_path}')

# Example usage
input_directory = 'C:/DataRec/GC'  # Note the corrected slashes
cut_csv_files(input_directory)
