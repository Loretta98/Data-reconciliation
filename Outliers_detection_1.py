import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps, values):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'outliers_st_dev')

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

        # Check the data type of third column
        if third_column.dtype == 'object':
            try:
                third_column = third_column.astype(float)
            except ValueError as e:
                print(f"Error converting column to float in file {filename}: {e}")
                continue

        # Remove zero values from the third column
        non_zero_indices = third_column != 0
        third_column = third_column[non_zero_indices]
        
        # Calculate the number of full n_steps intervals
        num_intervals = len(third_column) // n_steps
        print(f'Number of complete n_steps intervals: {num_intervals}')

        # Store the values for each n_steps in a dictionary
        all_st_devs = {}
        outlier_intervals = []

        # Calculate standard deviation for every n_steps rows
        st_devs = [np.std(third_column[i:i+n_steps]) for i in range(0, len(third_column), n_steps)]
        all_st_devs[f'n_steps_{n_steps}'] = st_devs

        # Initialize the allowable range matrix
        allowable_range = np.ones((len(values), num_intervals+1))

        # Adjust allowable range based on the mean of each segment
        for i in range(0,num_intervals):
            segment_mean = np.mean(third_column[i * n_steps:(i + 1) * n_steps])
            #print(f"Segment mean for interval {i}: {segment_mean}")
            if k == 0:
                allowable_range[k, i] = values[k] * segment_mean / 100
            elif k == 1:
                allowable_range[k, i] = values[k] * segment_mean / 100
            else:
                allowable_range[k, i] = values[k]
            #print(f"Allowable range for k={k}, i={i}: {allowable_range[k, i]}")

        #print("Final allowable range matrix:")
        #print(allowable_range)

        # Detect outliers
        for i, std in enumerate(st_devs):
            if std < -allowable_range[k, i] or std > allowable_range[k, i]:
                interval_start = i * n_steps
                interval_end = min(interval_start + n_steps - 1, len(third_column) - 1)  # Ensure the end index is within bounds
                outlier_intervals.append((interval_start, interval_end, n_steps))
        k += 1

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
        
        # Plot the allowable range as a shaded area
        plt.fill_between(range(num_intervals), allowable_range[k-1, :], -allowable_range[k-1, :], color='gray', alpha=0.2, label='Allowable Range')

        # Highlight outlier intervals
        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(interval_start // n_steps, interval_end // n_steps, color='red', alpha=0.3, label=f'Outlier {interval_start}-{interval_end} (n_steps={n_steps})')

        plt.title(f'Standard Deviations for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Standard Deviation')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()

        # Remove outliers and perform linear interpolation
        cleaned_data = third_column.copy()
        interpolated_indices = []
        for interval_start, interval_end, _ in outlier_intervals:
            if interval_start > 0 and interval_end < len(third_column) - 1:
                interpolated_values = np.interp(
                    range(interval_start, interval_end+1),
                    [interval_start-1, interval_end+1],
                    [third_column[interval_start-1], third_column[interval_end+1]]
                )
                cleaned_data[interval_start:interval_end+1] = interpolated_values
                interpolated_indices.extend(range(interval_start, interval_end+1))

        # Plot the original data with highlighted outliers
        plt.figure(figsize=(35, 12))
        plt.plot(third_column.index, third_column, label='Original Data')

        for interval_start, interval_end, n_steps in outlier_intervals:
            plt.axvspan(third_column.index[interval_start], third_column.index[interval_end], color='red', alpha=0.3, label=f'Outlier {interval_start}-{interval_end} (n_steps={n_steps})')

        plt.title(f'Original Data with Outliers for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()

        # Plot the cleaned data with highlighted interpolated values
        plt.figure(figsize=(35, 12))
        plt.plot(third_column.index, cleaned_data, label='Cleaned Data')

        # Highlight interpolated values
        plt.scatter(third_column.index[interpolated_indices], cleaned_data[interpolated_indices], color='green', label='Interpolated Values')

        plt.title(f'Cleaned Data for {file_basename}')
        plt.xlabel('Index (Time)')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(outliers_st_dev_path, f"plot_cleaned_data_{file_basename}.png")
        plt.savefig(plot_path)
        plt.close()

        # Save the cleaned data to a new CSV file
        cleaned_data_path = os.path.join(outliers_st_dev_path, f"cleaned_data_{file_basename}.csv")
        cleaned_data_df = df[non_zero_indices].copy()
        cleaned_data_df.iloc[:, 2] = cleaned_data
        cleaned_data_df.to_csv(cleaned_data_path, index=False)

        # Report the intervals
        intervals_report_path = os.path.join(outliers_st_dev_path, f"intervals_report_{file_basename}.txt")
        with open(intervals_report_path, 'w') as report_file:
            report_file.write("Outlier Intervals (Start, End, n_steps):\n")
            for interval_start, interval_end, n_steps in outlier_intervals:
                report_file.write(f"Interval: {interval_start} - {interval_end}, n_steps: {n_steps}\n")

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
# The discretization for the data analysis should be higher or equivalent to the characteristic time of the system 
n_steps = 30 # Ensure n_steps is an integer
# measurement errors from provider, for each device 
values = [3.2 , 6.6 , 2, 1, 0.5]
highlight_outliers(input_directory, n_steps, values)
