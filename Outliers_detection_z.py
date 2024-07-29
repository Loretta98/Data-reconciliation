################### Outliers detection through z-score rolling method ###################

import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps, values, z_threshold):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'Z_score_method')

    # Create the 'outliers_st_dev' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    k = 0
    for filename in files:
        print(f"Processing file: {filename}")
        print(f"Current file index (k): {k}")
        
        # Read CSV file
        df = pd.read_csv(open(filename, 'rb'))

        # Extract the third column
        third_column = df.iloc[:, 2]

        # Remove zero values from the third column but maintain original indices
        third_column_non_zero = third_column[third_column != 0]
        original_indices = third_column_non_zero.index

        # Check the data type of third column
        if third_column_non_zero.dtype == 'object':
            try:
                third_column_non_zero = third_column_non_zero.astype(float)
            except ValueError as e:
                print(f"Error converting column to float in file {filename}: {e}")
                continue

        # Calculate the number of full n_steps intervals
        num_intervals = len(third_column_non_zero) // n_steps

        # Store the values for each n_steps in a dictionary
        all_st_devs = {}
        outlier_intervals = []
        interpolated_values = []

        # Calculate standard deviation for every n_steps rows
        st_devs = [np.std(third_column_non_zero[i:i+n_steps]) for i in range(0, len(third_column_non_zero), n_steps)]
        all_st_devs[f'n_steps_{n_steps}'] = st_devs

        # Initialize the allowable range matrix
        allowable_range = np.ones((len(values), num_intervals + 1))
        # Adjust allowable range based on the mean of each segment
        for i in range(num_intervals):
            segment_mean = np.mean(third_column_non_zero[i * n_steps:(i + 1) * n_steps])
            if k == 0:
                allowable_range[k, i] = values[k] * segment_mean / 100
            elif k == 1:
                allowable_range[k, i] = values[k] * segment_mean / 100
            else:
                allowable_range[k, i] = values[k]

        # Detect outliers using a rolling z-score method

        z_score = zscore(third_column_non_zero,n_steps)
        print(z_score)

        # for i in range(0, len(third_column_non_zero) - n_steps + 1):
        #     window = third_column_non_zero[i:i + n_steps]
        #     mean_value = np.mean(window)
        #     st_dev = np.std(window)
        #     z_score = (third_column_non_zero[i + n_steps - 1] - mean_value) / st_dev
        z_score = np.array(z_score)
        for i in range(0,len(z_score)):
            if abs(z_score[i]) > z_threshold:
                interval_start = i + n_steps - 1
                outlier_intervals.append((interval_start, interval_start, 1))

        #         # Linear interpolation for outlier replacement
        #         if interval_start > 0 and interval_start < len(third_column_non_zero) - 1:
        #             interpolated_value = np.interp([interval_start],
        #                                            [interval_start - 1, interval_start + 1],
        #                                            [third_column_non_zero.iloc[interval_start - 1], third_column_non_zero.iloc[interval_start + 1]])
        #             third_column_non_zero.iloc[interval_start] = interpolated_value
        #             interpolated_values.append((interval_start, interval_start, interpolated_value))

        # k += 1
        # # Create a DataFrame from the dictionary
        # max_length = max(len(st_devs) for st_devs in all_st_devs.values())
        # for key in all_st_devs:
        #     all_st_devs[key] += [np.nan] * (max_length - len(all_st_devs[key]))  # Fill shorter lists with NaN
        # st_devs_df = pd.DataFrame(all_st_devs)

        # # Save the DataFrame to a CSV file
        file_basename = os.path.basename(filename).split('.')[0]
        # st_dev_csv_path = os.path.join(outliers_st_dev_path, f"outliers_st_dev_{file_basename}.csv")
        # st_devs_df.to_csv(st_dev_csv_path, index=False)

        
        # Plot the allowable range as a shaded area
        plt.fill_between(range(num_intervals + 1), allowable_range[k-1, :], -allowable_range[k-1, :], color='gray', alpha=0.2, label='Allowable Range')

        # Highlight outlier intervals
        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(interval_start // n_steps, interval_end // n_steps, color='red', alpha=0.3, label=f'Outlier {interval_start}-{interval_end} (n_steps={n_steps})')

        # Plot the original data with highlighted outliers
        plt.figure(figsize=(35, 12))
        plt.plot(original_indices, third_column_non_zero, label='Original Data')

        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(original_indices[interval_start], original_indices[interval_end], color='red', alpha=0.3, label=f'Outlier {original_indices[interval_start]}-{original_indices[interval_end]} (n_steps={n_steps})')

        # Highlight interpolated values
        for interval_start, interval_end, interpolated_value in interpolated_values:
            plt.plot(original_indices[interval_start:interval_end + 1], interpolated_value, color='blue', marker='o', linestyle='--', label='Interpolated Values')

        plt.title(f'Original Data with Outliers for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
n_steps = 15  # Ensure n_steps is an integer
values = [3.2, 6.6, 2, 1, 0.5]  # Measurement errors from provider, for each device
z_threshold = 3
highlight_outliers(input_directory, n_steps, values, z_threshold)
