import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps, values, z_threshold):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'Z_score')

    # Create the 'outliers_st_dev' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    outlier_counts = []

    for k, filename in enumerate(files):
        print(f"Processing file: {filename}")
        
        # Read CSV file
        df = pd.read_csv(open(filename, 'rb'))

        # Extract the third column
        third_column = df.iloc[:, 2]

        # Detect outliers using a rolling z-score method
        z_scores = zscore(third_column, n_steps)
        
        outlier_intervals = [i for i, z in enumerate(z_scores) if abs(z) > z_threshold]

        # Count the number of outliers
        num_outliers = len(outlier_intervals)
        outlier_counts.append({'Filename': os.path.basename(filename), 'Num_Outliers': num_outliers})

        # Save the DataFrame to a CSV file
        file_basename = os.path.basename(filename).split('.')[0]

        # Plot the original data with highlighted outliers
        plt.figure(figsize=(24, 10))
        plt.scatter(third_column.index, third_column, label='Original Data', color='blue')
        #plt.plot(third_column.index,z_scores)
        if outlier_intervals:  # Check if there are outliers to plot
            plt.scatter(third_column.index[outlier_intervals], third_column[outlier_intervals], label='Outliers', color='red')

        plt.title(f'Z score ouliers detection in {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()

        results.append({
        'Filename': file_basename,
        'Number of Outliers': num_outliers})
    
    # # Save outlier counts to a CSV file
    # outlier_counts_df = pd.DataFrame(outlier_counts)
    # outlier_counts_df.to_csv(os.path.join(outliers_st_dev_path, 'outlier_counts.csv'), index=False)

    
def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    # Handle cases where the standard deviation is zero
    # Mattia !!! help
    z[ s < 0.1] = 0
    return z

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
n_steps = 20  # Ensure n_steps is an integer
values = [3.2, 6.6, 2, 1, 0.5]  # Measurement errors from provider, for each device
z_threshold = 3.0
# Initialize a list to store the results
results = []
highlight_outliers(input_directory, n_steps, values, z_threshold)


# Convert results list to DataFrame
results_df = pd.DataFrame(results)
outliers_st_dev_path = os.path.join(input_directory, 'Z_score')
# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_z.csv'), index=False)