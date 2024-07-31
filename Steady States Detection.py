import numpy as np
import pandas as pd
from scipy.stats import t
import os
import glob
import matplotlib.pyplot as plt

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
steady_states_path = os.path.join(path, 'Steady States - Dalheim')
# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(steady_states_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))

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
        
        # Calculate t1
        t1 = b1 / sigma_b1
        
        # Check if the null hypothesis is rejected
        if abs(t1) > t_critical:
            unsteady_state_windows.append((start, end, b1, b0, sigma_a, sigma_b1, t1))
        else: 
            steady_state_windows.append((start, end, b1, b0, sigma_a, sigma_b1, t1))

    return [unsteady_state_windows,steady_state_windows]

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame().values.flatten()

    window_size = 5  # Define your window size
    alpha = 0.01  # Significance level
    result = detect_steady_state(data, window_size, alpha)
    unsteady_state_windows = result[0]
    steady_states_windows = result[1]
    #print("Steady state windows:", steady_state_windows)
    
    # Save steady state windows to CSV
    unsteady_state_df = pd.DataFrame(unsteady_state_windows, columns=['start', 'end', 'b1', 'b0', 'sigma_a', 'sigma_b1', 't1'])
    unsteady_state_df.to_csv(os.path.join(steady_states_path, f"{file_basename}_steady_states.csv"), index=False)
    
    # Plot the data and the steady states
    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column, label='Data', color='blue')
    
    # Highlight steady state points
    unsteady_state_points = []
    for (start, end, b1, b0, sigma_a, sigma_b1, t1) in unsteady_state_windows:
        unsteady_state_points.extend(range(start, end))
    unsteady_state_points = np.unique(unsteady_state_points)  # Remove duplicates
    plt.scatter(unsteady_state_points, data[unsteady_state_points], color='red', label='Steady State Points')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Steady State Detection for {file_basename}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(steady_states_path, f"{file_basename}_steady_states.png"))
    plt.close()