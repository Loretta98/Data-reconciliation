################### Outliers detection through DBSCAN method ###################
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

# Paths
path = os.path.join('C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation')
outliers_st_dev_path = os.path.join(path, 'DBSCAN')

# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)

# List CSV files and sort them
files = sorted(glob.glob(os.path.join(path, '*.csv')))
eps_values = [0.01, 0.01, 0.01, 0.01, 0.01]  # DBSCAN parameter for each dataset
min_samples = 5  # Minimum number of samples in a neighborhood for a point to be considered a core point
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
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Plot k-distance graph
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    distances = np.sort(distances[:, min_samples - 1], axis=0)
    
    plt.figure(figsize=(24, 10))
    plt.plot(distances)
    plt.title(f'k-distance Graph for {file_basename}')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'Distance to {min_samples}-th Nearest Neighbor')
    plot_path = os.path.join(outliers_st_dev_path, f"plot_k_distance_{file_basename}.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Apply DBSCAN
    db = DBSCAN(eps=eps_values[k], min_samples=min_samples).fit(data_scaled)
    labels = db.labels_

    # Identify outliers (DBSCAN labels outliers as -1)
    outlier_indices = np.where(labels == -1)[0]

    # Add results to the list
    results.append({
        'Filename': file_basename,
        'Number of Outliers': len(outlier_indices),
        'Outlier Indices': outlier_indices
    })

    print(f"Number of outliers detected: {len(outlier_indices)}")

    # Plot data and outliers
    plt.figure(figsize=(24, 10))
    plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
    plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
    plt.title(f"Data and Outliers in {file_basename}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
    plt.savefig(plot_path)
    plt.close()

    k += 1

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_DBSCAN.csv'), index=False)
