####### Outliers detection with an Isolation Forest approach ######## 

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
outliers_st_dev_path = os.path.join(path, 'Isolation Forest')
steady_state_path = os.path.join(path, 'Steady States')
# Create directories if they do not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
os.makedirs(steady_state_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))

# Define parameters for rolling window analysis
window_size = 10  # Adjust based on your data
std_dev_threshold = 0.1  # Adjust based on your data
values = [3.2 , 6.6 , 2, 1, 0.5]
k = 0 
# Initialize a list to store the results
results = []

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
        # Extract the relevant column (third column)
    third_column = df.iloc[:, 2]
     
    # Calculate rolling standard deviation
    rolling_std = third_column.rolling(window=window_size).std()
    
    num_intervals1 = len(third_column) // window_size
    num_intervals = np.size(rolling_std)
    # Initialize the allowable range matrix
    allowable_range = np.ones((len(values), num_intervals+1))

    # Adjust allowable range based on the mean of each segment
    for i in range(0,num_intervals1):
        segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
        for j in range(0,window_size): 
            if k == 0:
                allowable_range[k, j+i] = values[k] * segment_mean / 100
            elif k == 1:
                allowable_range[k, j+i] = values[k] * segment_mean / 100
            else:
                allowable_range[k, j+i] = values[k]
    
    # Identify steady states where rolling standard deviation is below the threshold
    
    df['steady_state'] = (rolling_std < allowable_range[k,:-1]).astype(int)
    
    # Reshape the data to be a 2D array as required by IsolationForest
    data = third_column.values.reshape(-1, 1)
    
    # Initialize the Isolation Forest
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    
    # Fit the model
    iso_forest.fit(data)
    
    # Predict the outliers
    df['anomaly'] = iso_forest.predict(data)
    
    # -1 for outliers and 1 for inliers, converting to 0 for inliers and 1 for outliers
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})


    # # Save the result to a new CSV file
    # output_file = os.path.join(outliers_st_dev_path, f"{file_basename}_with_outliers.csv")
    # df.to_csv(output_file, index=False)
    
    # Print the number of anomalies detected
    num_anomalies = df['anomaly'].sum()
    print(f"Number of anomalies detected in {filename}: {num_anomalies}")
    
    # # Print the number of steady states detected
    # num_steady_states = df['steady_state'].sum()
    # print(f"Number of steady states detected in {filename}: {num_steady_states}")

    # Add results to the list
    results.append({
        'Filename': file_basename,
        'Number of Outliers': num_anomalies,
    })
    
    # Optionally plot the data
    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column, c=df['anomaly'], cmap='coolwarm', label='Data points') 
    plt.xlabel('Timeframe')
    plt.ylabel('Value')
    plt.title(f'Isolation Forest Outlier Detection for {file_basename}')
    plt.legend()
    plt.savefig(os.path.join(outliers_st_dev_path, f"{file_basename}_outliers.png"))
    plt.close()
    
    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column,label='Data points') 
    plt.scatter(third_column.index[df['steady_state'] == 1], third_column[df['steady_state'] == 1], color='green', label='Steady States')
    plt.xlabel('Timeframe')
    plt.ylabel('Value')
    plt.title(f'Steady States for {file_basename}')
    plt.legend()
    plt.savefig(os.path.join(outliers_st_dev_path, f"{file_basename}steady_states_plot.png"))
    plt.close()
    k = k+1

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_IF.csv'), index=False)