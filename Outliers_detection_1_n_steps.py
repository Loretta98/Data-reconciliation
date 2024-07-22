import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'outliers_st_dev')

    # Create the 'outliers_st_dev' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    # Allowable range for standard deviations
    allowable_range = (-2.5, 2.5)  # Example allowable range for standard deviations

    for filename in files:
        print(f"Processing file: {filename}")

        # Read CSV file
        df = pd.read_csv(open(filename, 'rb'))
        
        # Extract the third column
        third_column = df.iloc[:, 2]
        
        # Check the data type of third column
        if third_column.dtype == 'object':
            try:
                third_column = third_column.astype(float)
            except ValueError as e:
                print(f"Error converting column to float in file {filename}: {e}")
                continue
        
        # Store the values for each n_steps in a dictionary
        all_st_devs = {}
        outlier_intervals = []

        # Calculate standard deviation for every n_steps rows
        st_devs = [np.std(third_column[i:i+n_steps]) for i in range(0, len(third_column), n_steps)]
        all_st_devs[f'n_steps_{n_steps}'] = st_devs
        
        # Detect outliers
        for i, std in enumerate(st_devs):
            if std < allowable_range[0] or std > allowable_range[1]:
                interval_start = i * n_steps
                interval_end = min(interval_start + n_steps - 1, len(third_column) - 1)  # Ensure the end index is within bounds
                outlier_intervals.append((interval_start, interval_end, n_steps))
        
        # Create a DataFrame from the dictionary
        max_length = max(len(st_devs) for st_devs in all_st_devs.values())
        for key in all_st_devs:
            all_st_devs[key] += [np.nan] * (max_length - len(all_st_devs[key]))  # Fill shorter lists with NaN
        st_devs_df = pd.DataFrame(all_st_devs)
        
        # Save the DataFrame to a CSV file
        file_basename = os.path.basename(filename).split('.')[0]
        st_dev_csv_path = os.path.join(outliers_st_dev_path, f"outliers_st_dev_{file_basename}.csv")
        st_devs_df.to_csv(st_dev_csv_path, index=False)
        
        # Plot the standard deviations and highlight outliers
        plt.figure(figsize=(35, 12))
        for key, st_devs in all_st_devs.items():
            plt.plot(st_devs, marker='o', label=key)
        
        # Highlight outlier intervals
        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(interval_start // n_steps, interval_end // n_steps, color='red', alpha=0.3, label=f'Outlier {interval_start}-{interval_end} (n_steps={n_steps})')
        
        plt.title(f'Standard Deviations for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Standard Deviation')
        #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Plot the original data with highlighted outliers
        plt.figure(figsize=(35, 12))
        plt.plot(third_column, label='Original Data')
        
        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(interval_start, interval_end, color='red', alpha=0.3, label=f'Outlier {interval_start}-{interval_end} (n_steps={n_steps})')
        
        plt.title(f'Original Data with Outliers for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Value')
        #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Report the intervals
        intervals_report_path = os.path.join(outliers_st_dev_path, f"intervals_report_{file_basename}.txt")
        with open(intervals_report_path, 'w') as report_file:
            report_file.write("Outlier Intervals (Start, End, n_steps):\n")
            for interval_start, interval_end, n_steps in outlier_intervals:
                report_file.write(f"Interval: {interval_start} - {interval_end}, n_steps: {n_steps}\n")

# Example usage
#input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV'
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
# The discretization for the data analysis should be higher or equivalent to the characteristic time of the system 
n_steps = 10 # Ensure n_steps is an integer
highlight_outliers(input_directory, n_steps)
