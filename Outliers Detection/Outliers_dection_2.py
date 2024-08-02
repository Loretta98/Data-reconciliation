import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
outliers_st_dev_path = os.path.join(path, 'KNN')
# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))
window_size = np.array([100,300,300,300,300])
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
    knn = NearestNeighbors(n_neighbors=window_size[k])  # You can choose the number of neighbors
    knn.fit(data_scaled)
    distances, indices = knn.kneighbors(data_scaled)

    # Calculate rolling standard deviation
    #rolling_std = third_column.rolling(window=window_size).std()

    # Computing the mean and standard deviation of the distances for each point's neighbors
    # Compute the mean distance
    mean_distances = distances.mean(axis=1)
    # Using the maximum distance to the k-nearest neighbors
    max_distances = distances.max(axis=1)
    distance = max_distances
    if k == 0 & k == 1 : 
        allowable_range = third_column*values[k]/100
    else: 
        allowable_range = np.ones(len(third_column))*values[k]/100
    # Set a threshold for what you consider an outlier
    #threshold = np.mean(distance) + 2 * np.std(distance)
    threshold = 3*allowable_range
    # Identify outliers
    outliers = distance > threshold
    outlier_indices = np.where(outliers)[0]

    # Add results to the list
    results.append({
        'Filename': file_basename,
        'Number of Outliers': len(outlier_indices),
        'Outlier Indices': outlier_indices})

    print(f"Number of outliers detected: {len(outlier_indices)}")

    # Plot the data and highlight outliers
    plt.figure(figsize=(24, 10))
    plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
    plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
    plt.title(f"Data and Outliers in {file_basename}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    # Save the plot
    plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
    plt.savefig(plot_path)
    plt.close()
    
    plt.figure(figsize=(24, 10))
    plt.plot(data.index, allowable_range,label='allowable')
    plt.plot(data.index,threshold,label='threshold')
    plt.plot(data.index, distance, label = 'distance')
    plt.title(f"Allowable for {file_basename}")
    plt.legend()
    # Save the plot
    plot_path = os.path.join(outliers_st_dev_path, f"plot_distance_threshold{file_basename}.png")
    plt.savefig(plot_path) 
    
    k = k+1
# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_KNN.csv'), index=False)
