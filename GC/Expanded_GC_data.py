# The missing time data are interpolated on the base of the available data (every 15 minutes)

import pandas as pd
import os
import matplotlib.pyplot as plt

# Load your data
file_path = r'C:\DataRec\GC\Partial CSV\To Interpolate'  # Path to your CSV files
csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
output_dir = os.path.join(file_path, 'Time_Complete')
plots_dir = os.path.join(file_path, 'Plots')  # Directory for saving plots

# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

for file in csv_files:
    current_file_path = os.path.join(file_path, file)
        
    # Read the CSV file
    df = pd.read_csv(current_file_path)

    # Convert 'Injection Time' to a datetime object
    df['Injection Time'] = pd.to_datetime(df['Injection Time'], infer_datetime_format=True)

    # Set 'Injection Time' as the index
    df.set_index('Injection Time', inplace=True)

    # Resample the data to have one row per minute
    df_resampled = df.resample('1T').asfreq()

    # Perform linear interpolation to fill NaN values
    df_interpolated = df_resampled.interpolate(method='linear')
    #df_interpolated = df_resampled.interpolate(method='spline', order=3)

    # Reset the index so that 'Injection Time' becomes a column again
    df_interpolated.reset_index(inplace=True)

    # Save the interpolated data to a new CSV file
    output_path = os.path.join(output_dir, file)  # The file will be saved in the Time_Complete directory
    df_interpolated.to_csv(output_path, index=False)

    # Plotting the data
    plt.figure(figsize=(24, 10))
    
    # Plot each column except 'Injection Time'
    for column in df_interpolated.columns[1:]:
        plt.plot(df_interpolated['Injection Time'], df_interpolated[column], label=column)
    
    plt.title(f'Interpolated Values Over Time - {file}')
    plt.xlabel('Injection Time')
    plt.ylabel('Compound Concentration (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG file
    plot_path = os.path.join(plots_dir, file.replace('.csv', '.png'))
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory issues

    print(f"Interpolated data saved to {output_path}")
    print(f"Plot saved to {plot_path}")
