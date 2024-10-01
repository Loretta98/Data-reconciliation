####### Outliers detection with an Isolation Forest approach ######## 
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest

def isolation_forest_outlier_detection(input_path, values):
    path = input_path
    outliers_if_path = os.path.join(path, 'Isolation Forest')
    steady_state_path = os.path.join(path, 'Steady States')
    
    # Create directories if they do not exist
    os.makedirs(outliers_if_path, exist_ok=True)
    os.makedirs(steady_state_path, exist_ok=True)
    
    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    # # # Load the merged_data file
    # merged_df = pd.read_csv(merged_df_path)
    
    results = []

    for k, filename in enumerate(files):
        print(f"Processing file: {filename}")
        file_basename = os.path.basename(filename).split('.')[0]
        
        # Read CSV file
        df = pd.read_csv(open(filename, 'rb'))
        # Extract the third column
        third_column = df.iloc[:, 2]
        
        # Calculate rolling standard deviation
        window_size = 10  # Adjust based on your data
        rolling_std = third_column.rolling(window=window_size).std()
        
        num_intervals1 = len(third_column) // window_size
        num_intervals = np.size(rolling_std)
        # Initialize the allowable range matrix
        #allowable_range = np.ones((len(values), num_intervals+1))
        allowable_range = np.ones([1,num_intervals+1])

        # Adjust allowable range based on the mean of each segment
        for i in range(num_intervals1):
            segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
            for j in range(window_size): 
                    #allowable_range[0,j+i] = values
                # if k == 0:
                     #allowable_range[0, j+i] = values * segment_mean / 100
                # elif k == 1:
                #     allowable_range[k, j+i] = values[k] * segment_mean / 100
                # else:
                     allowable_range[k, j+i] = values
        
        # Identify steady states where rolling standard deviation is below the threshold
        df['steady_state'] = (rolling_std < allowable_range[k, :-1]).astype(int)
        
        # Reshape the data to be a 2D array as required by IsolationForest
        data = third_column.values.reshape(-1, 1)
        
        # Initialize the Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=10)
        
        # Fit the model
        iso_forest.fit(data)
        
        # Predict the outliers
        df['anomaly'] = iso_forest.predict(data)
        
        # -1 for outliers and 1 for inliers, converting to 0 for inliers and 1 for outliers
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
        
        # Add outlier column to merged_data
        outlier_col_name = f"{file_basename}_IF"
        # merged_df[outlier_col_name] = 0
        # merged_df.loc[df.index[df['anomaly'] == 1], outlier_col_name] = 1
        
        # Print the number of anomalies detected
        num_anomalies = df['anomaly'].sum()
        print(f"Number of anomalies detected in {filename}: {num_anomalies}")
        
        # Add results to the list
        results.append({
            'Filename': file_basename,
            'Number of Outliers': num_anomalies,
        })
        
        # Plot the data and highlight outliers
        plt.figure(figsize=(24, 10))
        plt.scatter(third_column.index, third_column, c=df['anomaly'], cmap='coolwarm', label='Data points') 
        plt.xlabel('Timeframe')
        plt.ylabel('Value')
        plt.title(f'Isolation Forest Outlier Detection for {file_basename}')
        plt.legend()
        plt.savefig(os.path.join(outliers_if_path, f"{file_basename}_outliers.png"))
        plt.close()
        
        plt.figure(figsize=(24, 10))
        plt.scatter(third_column.index, third_column, label='Data points') 
        plt.scatter(third_column.index[df['steady_state'] == 1], third_column[df['steady_state'] == 1], color='green', label='Steady State Points')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid()
        plt.title(f'Steady States for {file_basename} with isolation forest')
        plt.legend()
        plt.savefig(os.path.join(steady_state_path, f"{file_basename}_steady_states_plot.png"))
        plt.close()

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv(os.path.join(outliers_if_path, 'outliers_summary_IF.csv'), index=False)
    
    # # # Save the merged data with outlier columns to a CSV file
    # merged_df.to_csv(merged_df_path, index=False)

# Example usage
#input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
#input_directory = 'C:/DataRec/Ordered CSV/Mass Reconciliation/ft_03'
input_directory = 'C:\DataRec\Fuel'
#merged_data_path = os.path.join(input_directory, 'merged_data', 'merged_data.csv')
values = 2
#[3.2, 6.6, 2, 1, 0.5]

isolation_forest_outlier_detection(input_directory, values)

 # # Step 3: Calculate local neighborhood density (lnd_k)
    # lnd_k = np.zeros(len(data_scaled))
    # for i in range(len(data_scaled)):
    #     # Calculate the mean distance to the k-nearest neighbors
    #     avg_distance = np.mean(distances[i])
    #     lnd_k[i] = 1 / avg_distance if avg_distance > 0 else 0
    
    # # Step 4: Calculate average local neighborhood density of neighbors (lnd_k(N_k(p)))
    # lnd_k_neighbors = np.zeros(len(data_scaled))
    # for i in range(len(data_scaled)):
    #     # Calculate the average lnd_k of neighbors
    #     neighbor_indices = indices[i]
    #     lnd_k_neighbors[i] = np.mean(lnd_k[neighbor_indices])
    
    # # Step 5: Calculate NLOF
    # nlof_scores = lnd_k_neighbors / lnd_k
    
    # # Step 6: Sort by NLOF scores to identify the top m outliers
    # outlier_indices = np.argsort(nlof_scores)[-m:]  # Top m outliers with highest NLOF

