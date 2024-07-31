import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
outliers_st_dev_path = os.path.join(path, 'KNN + ss')
# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))
window_size = 10
values = [3.2, 6.6, 2, 1, 0.5]
k = 0

# Initialize a list to store the results
results = []

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame()
    # Assuming 'data' is a DataFrame with your features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    knn = NearestNeighbors(n_neighbors=15)  # You can choose the number of neighbors
    knn.fit(data_scaled)
    distances, indices = knn.kneighbors(data_scaled)

    # Identify Steady states, which can be simplified as clusters with minimal variations.
    # Option 1: to use density-based clustering (DBSCAN) to identify clusters
    # Computing the mean and standard deviation of the distances for each point's neighbors
    # Compute the mean distance
    mean_distances = distances.mean(axis=1)

    # Calculate rolling standard deviation
    rolling_std = third_column.rolling(window=window_size).std()
    num_intervals1 = len(third_column) // window_size
    num_intervals = np.size(rolling_std)
    # Initialize the allowable range matrix
    allowable_range = np.ones((len(values), num_intervals+1))
    # Adjust allowable range based on the mean of each segment
    for i in range(0, num_intervals1):
        segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
        for j in range(0, window_size):
            if k == 0:
                allowable_range[k, j + i] = values[k] * segment_mean / 100
            elif k == 1:
                allowable_range[k, j + i] = values[k] * segment_mean / 100
            else:
                allowable_range[k, j + i] = values[k]
    # Identify steady states where rolling standard deviation is below the threshold
    df['steady_state'] = (rolling_std < allowable_range[k, :-1]).astype(int)

    # Using the maximum distance to the k-nearest neighbors
    max_distances = distances.max(axis=1)
    distance = max_distances

    # Set a threshold for what you consider an outlier
    threshold = np.mean(distance) + 2 * np.std(distance)

    # Identify outliers
    outliers = distance > threshold
    outlier_indices = np.where(outliers)[0]

    # Add results to the list
    results.append({
        'Filename': file_basename,
        'Number of Outliers': len(outlier_indices),
        'Outlier Indices': outlier_indices
    })

    print(f"Number of outliers detected: {len(outlier_indices)}")

    # Plot the data and highlight outliers
    plt.figure(figsize=(24, 10))
    plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
    plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
    plt.title(f"Data, Outliers and steady states in {file_basename}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    # Save the plot
    plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(24, 10))
    plt.scatter(third_column.index, third_column, label='Data points')
    plt.scatter(third_column.index[df['steady_state'] == 1], third_column[df['steady_state'] == 1], color='green', label='Steady States')
    plt.xlabel('Timeframe')
    plt.ylabel('Value')
    plt.title(f'Steady States for {file_basename}')
    plt.legend()
    plt.savefig(os.path.join(outliers_st_dev_path, f"{file_basename}steady_states_plot.png"))
    plt.close()

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_KNN.csv'), index=False)
