import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def highlight_outliers(input_path, n_steps_range, values):
    path = input_path
    outliers_st_dev_path = os.path.join(path, 'ST.dev')

    # Create the 'ST.dev' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    print(f'Number of CSV files found: {len(files)}')

    results = []

    for n_steps in n_steps_range:
        print(f"Processing n_steps: {n_steps}")

        for k, filename in enumerate(files):
            print(f"Processing file: {filename}")
            
            # Read CSV file
            df = pd.read_csv(open(filename, 'rb'))

            # Extract the third column
            third_column = df.iloc[:, 2]

            # Check and convert the data type of the third column if necessary
            if third_column.dtype == 'object':
                try:
                    third_column = third_column.astype(float)
                except ValueError as e:
                    print(f"Error converting column to float in file {filename}: {e}")
                    continue

            # Calculate allowable range
            allowable_range = np.ones(len(third_column)) * values / 100

            # Calculate rolling standard deviation
            rolling_std = third_column.rolling(window=n_steps).std()
            rolling_std1 = third_column.rolling(window=10).std()
            
            # Identify steady states: rolling_std < allowable_range
            df['steady_state'] = (rolling_std1 < allowable_range).astype(int)

            # Detect outliers based on the rolling standard deviation exceeding 3 times the allowable range
            df['anomaly'] = (rolling_std > 3 * allowable_range).astype(int)

            # Identify real outliers: those that are outliers but not in steady state
            df['real_outlier'] = df.apply(lambda row: 1 if row['anomaly'] == 1 and row['steady_state'] == 0 else 0, axis=1)

            # Count real outliers and steady states marked as outliers
            real_outliers_count = df['real_outlier'].sum()
            steady_states_as_outliers_count = df[(df['anomaly'] == 1) & (df['steady_state'] == 1)].shape[0]

            results.append({
                'Filename': os.path.basename(filename),
                'n_steps': n_steps,
                'Number of Real Outliers': real_outliers_count,
                'Steady States as Outliers': steady_states_as_outliers_count
            })

            print(f"n_steps: {n_steps}")
            print(f"Real Outliers: {real_outliers_count}, Steady States as Outliers: {steady_states_as_outliers_count}")

            # Plot the original data with highlighted real outliers and steady states
            plt.figure(figsize=(20, 10))
            plt.scatter(third_column.index, third_column, label='Original Data', color='blue')
            
            # Plot real outliers
            plt.scatter(df.index[df['real_outlier'] == 1], third_column[df['real_outlier'] == 1], label='Real Outliers', color='red')
            
            # Plot steady states
            plt.scatter(df.index[df['steady_state'] == 1], third_column[df['steady_state'] == 1], label='Steady States', color='green', alpha=0.5)

            plt.title(f'Rolling Std Dev outliers detection in {os.path.basename(filename)} (n_steps = {n_steps})')
            plt.xlabel('Index (Time)')
            plt.ylabel('Value')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            plt.grid(True)

            # Save the plot
            plot_path = os.path.join(outliers_st_dev_path, f"plot_original_data_outliers_{os.path.basename(filename)}_n{n_steps}.png")
            plt.savefig(plot_path)
            plt.close()

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_std.csv'), index=False)

# Example usage
input_directory = 'C:\\DataRec\\FT_03'
n_steps_range = np.arange(5, 51, 5)  # Range from 5 to 50 in steps of 5
values = 2  # Single value since only one file is processed

highlight_outliers(input_directory, n_steps_range, values)
