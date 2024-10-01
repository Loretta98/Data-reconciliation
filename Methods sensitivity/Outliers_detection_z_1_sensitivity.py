import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob 

def highlight_outliers(input_file, n_steps_range, z_threshold_range, values):
    path = os.path.dirname(input_file)
    outliers_st_dev_path = os.path.join(path, 'Z_score')

    # Create the 'Z_score' directory if it does not exist
    os.makedirs(outliers_st_dev_path, exist_ok=True)
    files = sorted(glob.glob(path + '/*.csv'))
    print(f'Number of CSV files found: {len(files)}')
    
    for file in files: # Read CSV file (only one file now)
        df = pd.read_csv(open(file, 'rb'))
        #df = pd.read_csv(files[0])

        # Extract the third column for analysis
        third_column = df.iloc[:, 2]

        # Initialize a matrix to store the results (rows: n_steps, columns: z_threshold)
        results_matrix = pd.DataFrame(index=n_steps_range, columns=[f'Z_{z:.1f}' for z in z_threshold_range])

        # Compute rolling standard deviation to determine steady states
        window_size = 10  # Set window size for steady states calculation
        rolling_std = third_column.rolling(window=window_size).std()
        
        # Define allowable range for steady states based on some threshold
        allowable_range = np.ones(len(rolling_std)) * values/100
        df['steady_state'] = (rolling_std < allowable_range).astype(int)

        # Iterate over n_steps and z_threshold to detect outliers and compare with steady states
        for n_steps in n_steps_range:
            print(f"Processing n_steps: {n_steps}")

            for z_threshold in z_threshold_range:
                print(f"Processing z_threshold: {z_threshold}")

                # Detect outliers using rolling z-score method
                z_scores = zscore(third_column, n_steps)
                outlier_intervals = [i for i, z in enumerate(z_scores) if abs(z) > z_threshold]

                # Compare with steady states to identify real outliers
                df['real_outlier'] = 0
                for outlier_index in outlier_intervals:
                    if df['steady_state'].iloc[outlier_index] == 0:
                        df.loc[outlier_index, 'real outlier'] = 1

                # Count the number of real outliers
                num_real_outliers = df['real_outlier'].sum()
                print('outliers found:', num_real_outliers)

                # Store the result in the results matrix
                results_matrix.at[n_steps, f'Z_{z_threshold:.1f}'] = num_real_outliers

        # Save the results matrix to a CSV file
        results_matrix.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_matrix.csv'))

# Z-score calculation function
def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    # Handle cases where the standard deviation is zero
    z[s < 0.1] = 0
    return z

# Example usage
input_path = 'C:\DataRec\FT_03'
n_steps_range = range(10, 1001, 20)  # Vary n_steps from 10 to 1000 in steps of 20
z_threshold_range = np.arange(0.0, 3.5, 0.5)  # Range from 0 to 3.5 in steps of 0.5
values = 2

highlight_outliers(input_path, n_steps_range, z_threshold_range, values)
