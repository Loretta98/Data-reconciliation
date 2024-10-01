########## THIS CODE TESTS THE SENSITIVITY OF THE KNN ALGORITHM ON TOTAL OUTLIERS, AND TOTAL STEADY STATE POINTS ACCORDING TO window size and n of neighbors#############

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os 
import glob 

def knn_outlier_detection(input_path, window_size_range, values):
    path = input_path
    outliers_knn_path = os.path.join(path, 'KNN_Sens')
    
    # Create the 'KNN' directory if it does not exist
    os.makedirs(outliers_knn_path, exist_ok=True)
    
    # List CSV files (only one file is expected)
    files = sorted(glob.glob(path + '/*.csv'))
    if len(files) != 1:
        raise ValueError("The directory must contain exactly one CSV file.")
    filename = files[0]
    
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]

    # Read the CSV file
    df = pd.read_csv(open(filename, 'rb'))
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame()

    # Initialize a list to store the results
    results = []

    # Iterate over each window size
    for window_size in window_size_range:
        print(f"Processing window_size: {window_size}")

        # Calculate rolling standard deviation for steady state detection
        window_size1 = 10  # Adjust based on your data
        rolling_std = third_column.rolling(window=window_size1).std()
        
        num_intervals1 = len(third_column) // window_size
        num_intervals = np.size(rolling_std)
        allowable_range = np.ones([1,num_intervals+1])

        for i in range(num_intervals1):
            segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
            for j in range(window_size): 
                allowable_range[0, j+i] = values/100
        
        # Identify steady states
        df['steady_state'] = (rolling_std < allowable_range[0, :-1]).astype(int)

        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # KNN method
        knn = NearestNeighbors(n_neighbors=window_size)
        knn.fit(data_scaled)
        distances, _ = knn.kneighbors(data_scaled)

        # Computing the maximum distance to the k-nearest neighbors
        max_distances = distances.max(axis=1)
        
        # Determine the threshold for outliers
        allowable_range = np.ones(len(third_column)) * values / 100
        threshold = 3 * allowable_range

        # Identify outliers
        outliers = max_distances > threshold
        df['anomaly'] = outliers.astype(int)

        # Differentiating between real outliers and steady states
        df['real_outlier'] = df.apply(lambda row: 1 if row['anomaly'] == 1 and row['steady_state'] == 0 else 0, axis=1)

        # Count real outliers and steady states incorrectly labeled as outliers
        real_outliers_count = df['real_outlier'].sum()
        steady_states_as_outliers_count = df[(df['anomaly'] == 1) & (df['steady_state'] == 1)].shape[0]
        
        print(f"Real Outliers: {real_outliers_count}, Steady States as Outliers: {steady_states_as_outliers_count}")
        
        # Add results to the list
        results.append({
            'Filename': file_basename,
            'Window_Size': window_size,
            'Real Outliers': real_outliers_count,
            'Steady States as Outliers': steady_states_as_outliers_count
        })

        # Plot the results
        plt.figure(figsize=(24, 10))
        plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
        # For real outliers
        plt.scatter(data.index[df['real_outlier'] == 1], data.loc[df['real_outlier'] == 1, data.columns[0]], color='red', label='Real Outliers')

        # For steady states
        plt.scatter(data.index[df['steady_state'] == 1], data.loc[df['steady_state'] == 1, data.columns[0]], color='green', label='Steady States')

        plt.title(f"Real Outliers and Steady States in {file_basename} (Window Size = {window_size})")
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        # Save the plot
        plot_path = os.path.join(outliers_knn_path, f"plot_outliers_knn_{file_basename}_w{window_size}.png")
        plt.savefig(plot_path)
        plt.close()

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv(os.path.join(outliers_knn_path, 'real_outliers_summary_KNN.csv'), index=False)
    print("Real outliers summary saved to CSV.")

# Example usage
input_directory = 'C:/DataRec/FT_03'
window_size_range = np.arange(5, 1015, 10)  # Range from 5 to 300 in steps of 20
values = 2  # Single value since only one file is processed

knn_outlier_detection(input_directory, window_size_range, values)
