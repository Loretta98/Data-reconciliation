################### Outliers detection through rolling standard deviation method ###################

import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps_range, values, merged_df_path):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'ST.dev')

    # Create the 'outliers_st_dev' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    results = []

    # Load the merged_data file
    merged_df = pd.read_csv(merged_df_path)

    for n_steps in n_steps_range:
        print(f"Processing n_steps: {n_steps}")

        for k, filename in enumerate(files):
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

            # Calculate allowable range
            allowable_range = np.ones(len(third_column)) * np.mean(third_column[3000:]) * values / 100

            # Calculate rolling standard deviation
            rolling_std = third_column.rolling(window=n_steps).std().shift(1)
            
            # Detect outliers based on the rolling standard deviation and allowable range
            outlier_intervals = [i for i, std in enumerate(rolling_std) if std > 3 * allowable_range[i]]
            
            # Count the number of outliers
            num_outliers = len(outlier_intervals)
            results.append({
                'Filename': os.path.basename(filename),
                'n_steps': n_steps,
                'Number of Outliers': num_outliers
            })

            # Add outlier column to merged_data
            file_basename = os.path.basename(filename).split('.')[0]
            outlier_col_name = f"{file_basename}_STD_DEV_W{n_steps}"
            merged_df[outlier_col_name] = 0
            merged_df.loc[outlier_intervals, outlier_col_name] = 1

            # Plot the original data with highlighted outliers
            plt.figure(figsize=(20, 10))
            plt.scatter(third_column.index, third_column, label='Original Data', color='blue')
            
            if outlier_intervals:  # Check if there are outliers to plot
                plt.scatter(third_column.index[outlier_intervals], third_column[outlier_intervals], label='Outliers', color='red')

            plt.title(f'Rolling Std Dev outliers detection in {os.path.basename(filename)} (n_steps = {n_steps})')
            plt.xlabel('Index (Time)')
            plt.ylabel('Value')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            plt.grid(True)

            # Save the plot
            plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{os.path.basename(filename)}_n{n_steps}.png")
            plt.savefig(plot_path)
            plt.close()

            # plt.figure(figsize=(24, 10))
            # plt.plot(third_column.index, allowable_range, label='allowable')
            # plt.plot(third_column.index, 3 * allowable_range, label='threshold')
            # plt.plot(third_column.index, rolling_std, label='st_dev')
            # plt.title(f"Allowable for {os.path.basename(filename)} (n_steps = {n_steps})")
            # plt.legend()

            # # Save the plot
            # plot_path = os.path.join(outliers_st_dev_path, f"threshold_{os.path.basename(filename)}_n{n_steps}.png")
            # plt.savefig(plot_path)
            # plt.close()

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_std.csv'), index=False)

    # Save the merged data with outlier columns to a CSV file
    merged_df.to_csv(merged_df_path, index=False)

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation/ft_03'
merged_data_path = os.path.join(input_directory, 'ST.dev', 'merged_data.csv')
n_steps_range = np.arange(5, 51, 5)  # Range from 5 to 50 in steps of 5
values = 2  # Single value since only one file is processed

highlight_outliers(input_directory, n_steps_range, values, merged_data_path)
