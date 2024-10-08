import os
import pandas as pd

# Define the input and output directories
input_dir = r'C:\DataRec\To_reconcile\new'
output_dir = r'C:\DataRec\To_reconcile\fifteen'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each CSV file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing file: {file_name}")

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure the 'TimeString' column is in datetime format with the specified format
            df['TimeString'] = pd.to_datetime(df['TimeString'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

            # Check for any rows where the datetime conversion failed
            if df['TimeString'].isnull().any():
                print(f"Warning: Some rows in {file_name} have invalid date formats and will be ignored.")

            # Drop rows with invalid datetime entries
            df.dropna(subset=['TimeString'], inplace=True)

            # Set 'TimeString' as the DataFrame index
            df.set_index('TimeString', inplace=True)

            # Resample the data to 15-minute intervals based on the actual timestamps and calculate the mean of 'VarValue'
            resampled_df = df.resample('15T').mean()

            # Handle NaN values: if the interval has NaN, forward fill or use existing point
            resampled_df.fillna(method='ffill', inplace=True)

            # Drop remaining NaN values if they still exist
            resampled_df.dropna(inplace=True)

            # Reset the index to turn the datetime index back into a column
            resampled_df.reset_index(inplace=True)

            # Define the output file path
            output_file_path = os.path.join(output_dir, file_name)

            # Write the resampled DataFrame to a new CSV file in the output directory
            resampled_df.to_csv(output_file_path, index=False)

            print(f"File {file_name} processed successfully. Saved to the 'fifteen' directory.\n")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print("Data resampling complete.")
