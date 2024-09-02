import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import glob

# Base directory for the files
base_path = 'C:/DataRec/Ordered CSV/Mass Reconciliation/ft_03/KNN cleaned data'
steady_state_path = os.path.join(base_path, 'Steady States - Dalheim')
os.makedirs(steady_state_path, exist_ok=True)

# Debug: Print to verify paths
print(f"Base path: {base_path}")
print(f"Directory for steady states: {steady_state_path}")

# List CSV files and sort them
files = sorted(glob.glob(os.path.join(base_path, '*.csv')))

def detect_steady_state(data, window_size, alpha):
    # Initialize list to store results
    unsteady_state_windows = []
    steady_state_windows = []
    n = window_size
    df = n - 2
    t_critical = t.ppf(1 - alpha / 2, df)  # Get the critical t-value

    for start in range(len(data) - window_size + 1):
        end = start + window_size
        window_data = data[start:end]
        x = np.arange(n)
        z_t = window_data
        
        # Calculate the necessary sums
        sum_t = np.sum(x)
        sum_z_t = np.sum(z_t)
        sum_tz_t = np.sum(x * z_t)
        sum_t_squared = np.sum(x ** 2)
        t_average = np.mean(x)
        
        # Estimate b1
        b1_numerator = sum_tz_t - (1/n) * sum_t * sum_z_t
        b1_denominator = sum_t_squared - (1/n) * (sum_t ** 2)
        b1 = b1_numerator / b1_denominator
        
        # Estimate b0
        b0 = (1/n) * (sum_z_t - b1 * sum_t)
        
        # Calculate residuals
        residuals = z_t - b0 - b1 * x
        
        # Estimate sigma_a
        sigma_a = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        
        # Estimate sigma_b1
        sum_t_diff_squared = np.sum((x - t_average) ** 2)
        sigma_b1 = sigma_a / np.sqrt(sum_t_diff_squared)
        if sigma_b1 < 0.01: 
            t1 = 0
        else: 
        # Calculate t1
            t1 = b1 / sigma_b1
        
        # Check if the null hypothesis is rejected
        if abs(t1) > t_critical:
            unsteady_state_windows.append((start, end, b1, b0, sigma_a, sigma_b1, t1))
        else: 
            steady_state_windows.append((start, end, b1, b0, sigma_a, sigma_b1, t1))

    return [unsteady_state_windows, steady_state_windows]

alpha = 0.8
window_size = 5
k = 0 

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    
    # Read CSV file
    df = pd.read_csv(filename)
    third_column = df.iloc[:, 1]
    data = third_column.to_numpy()
    
    result = detect_steady_state(data, window_size, alpha)
    unsteady_state_windows = result[0]
    steady_states_windows = result[1]
    
    # Save steady state windows to CSV
    output_csv_path = os.path.join(steady_state_path, f'{file_basename}_steady_states.csv')
    print(f"Saving CSV to: {output_csv_path}")
    
    steady_state_df = pd.DataFrame(steady_states_windows, columns=['start', 'end', 'b1', 'b0', 'sigma_a', 'sigma_b1', 't1'])
    steady_state_df.to_csv(output_csv_path, index=False)
    
    # Plot the data and the steady states
    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column, label='Data', color='blue')
    
    # Highlight steady state points
    steady_state_points = []
    for (start, end, b1, b0, sigma_a, sigma_b1, t1) in steady_states_windows:
        steady_state_points.extend(range(start, end))
    steady_state_points = np.unique(steady_state_points)  # Remove duplicates
    plt.scatter(steady_state_points, data[steady_state_points], color='green', label='Steady State Points')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Steady State Detection for {file_basename}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(steady_state_path, f"{file_basename}_steady_states.png"))
    plt.close()
    
    k += 1
