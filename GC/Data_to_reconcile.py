import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths to directories
gc_file_path = r'C:\DataRec\GC\Partial CSV\To Interpolate\Time_Complete'  # Directory containing interpolated GC files
reference_file_path = r'C:\DataRec\GC\For Reconciliation\1_media_ft02_nm3_h_cleaned_cleaned_cleaned_cleaned_cleaned.csv'  # Path to the reference CSV file with desired times
filtered_output_dir = os.path.join(gc_file_path, 'Filtered_Time')
plots_dir = os.path.join(gc_file_path, 'Filtered_Plots')  # Directory for saving filtered plots

# Create output directories if they don't exist
os.makedirs(filtered_output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Read the reference CSV file and extract the desired timestamps (ignoring seconds)
reference_df = pd.read_csv(reference_file_path)
reference_df['Injection Time'] = pd.to_datetime(reference_df['Injection Time'], format='%d.%m.%Y %H:%M:%S')
reference_df['Injection Time'] = reference_df['Injection Time'].dt.floor('T')  # Set seconds to 00

# List of interpolated CSV files from the GC
gc_csv_files = [f for f in os.listdir(gc_file_path) if f.endswith('.csv')]

for file in gc_csv_files:
    gc_file_full_path = os.path.join(gc_file_path, file)
    
    # Read the interpolated GC data
    gc_df = pd.read_csv(gc_file_full_path)

    # Convert 'Injection Time' to datetime format and ignore seconds
    gc_df['Injection Time'] = pd.to_datetime(gc_df['Injection Time'], infer_datetime_format=True)
    gc_df['Injection Time'] = gc_df['Injection Time'].dt.floor('T')  # Set seconds to 00

    # Filter the GC data to keep only rows with matching Injection Times at minute precision
    filtered_gc_df = gc_df[gc_df['Injection Time'].isin(reference_df['Injection Time'])]

    # Save the filtered data to a new CSV file
    filtered_output_path = os.path.join(filtered_output_dir, file)
    filtered_gc_df.to_csv(filtered_output_path, index=False)

    # Plotting the filtered data
    plt.figure(figsize=(10, 6))
    
    # Plot each column except 'Injection Time'
    for column in filtered_gc_df.columns[1:]:
        plt.plot(filtered_gc_df['Injection Time'], filtered_gc_df[column], label=column)
    
    plt.title(f'Filtered Values Over Time - {file}')
    plt.xlabel('Injection Time')
    plt.ylabel('Compound Concentration (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG file
    plot_path = os.path.join(plots_dir, file.replace('.csv', '_filtered.png'))
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory issues

    print(f"Filtered data saved to {filtered_output_path}")
    print(f"Filtered plot saved to {plot_path}")
