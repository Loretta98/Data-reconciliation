import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import glob 

def ssd(df, n, t_crit):
    # Initialize the output array
    P = np.zeros(len(df))

    # Ensure n is not larger than the length of the dataframe
    if n > len(df):
        n = len(df)

    k = 0
    should_break = False

    while True:
        from_idx = k
        to_idx = from_idx + n - 1

        if to_idx >= len(df):
            to_idx = len(df) - 1
            n = to_idx - from_idx + 1
            should_break = True

        x_active = df[from_idx:to_idx+1]

        # Estimate the slope (m) of the drift component
        m = 0
        for t in range(1, n):
            m += (x_active.iloc[t] - x_active.iloc[t-1])
        m /= n

        # Equation 3
        mu = (x_active.sum() - sum(range(1, n+1)) * m) / n

        # Equation 4
        sd = 0
        for t in range(n):
            sd += (x_active.iloc[t] - m * (t + 1) - mu) ** 2
        sd = np.sqrt(sd / (n - 2))

        # Equation 5
        y = 0
        for t in range(n):
            if abs(x_active.iloc[t] - mu) <= t_crit * sd:
                y += 1
        y /= n

        P[from_idx:to_idx+1] = y

        if should_break:
            break

        k += n

    return P

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
steady_states_path = os.path.join(path, 'Steady States - Kelly')
# Create the 'Steady States - Kelly' directory if it does not exist
os.makedirs(steady_states_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))

# Example usage
t_crit = np.array([2,2,3,3,2.2])
n = np.array([15,50,50,50,50])
k = 0 
threshold = 0.8  # Threshold for identifying steady state points

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame().values.flatten()
    print(k)

    P = ssd(third_column, n[k], t_crit[k])
    
    # Identify unsteady state points
    unsteady_state_points = np.where(P < threshold)[0]

    # Plot the data and the steady states
    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column, label='Data', color='blue')
    plt.scatter(unsteady_state_points, data[unsteady_state_points], color='red', label='Unsteady State Points')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Steady State Detection for {file_basename}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(steady_states_path, f"{file_basename}_unsteady_states.png"))
    plt.close()
    
    k += 1
