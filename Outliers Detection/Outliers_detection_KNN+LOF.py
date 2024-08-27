import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
import glob 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

def knn_nlof_outlier_detection(data, k, m):
    """
    Detect outliers using a refined KNN-based NLOF approach with hierarchical adjacency.
    
    Parameters:
    - data: A NumPy array or Pandas DataFrame containing the dataset.
    - k: The number of nearest neighbors.
    - m: The number of outliers to return.
    
    Returns:
    - outliers: The indices of the top m outliers in the dataset.
    - nlof_scores: The NLOF scores for each data point.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
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


    
    # Step 1: Calculate the k-nearest neighbors
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(data_scaled)
    distances, indices = knn.kneighbors(data_scaled)
    
    # Step 2: Calculate the link distance for hierarchization
    link_distances = np.sort(distances, axis=1)
    
    # Step 3: Assign weights based on the hierarchy
    levels = np.arange(1, k + 1)
    weights = 1.0 / levels
    weighted_distances = np.sum(link_distances * weights, axis=1) / np.sum(weights)
    
    # Step 4: Calculate sequence distance (from formula in the paper)
    sequence_distances = np.zeros(len(data_scaled))
    for i in range(len(data_scaled)):
        sequence_distances[i] = np.sum(np.abs(data_scaled[i] - data_scaled[indices[i]]) * weights) / np.sum(weights)
    
    # Step 5: Calculate the average sequence distance (ASD)
    asd = np.mean(sequence_distances)
    
    # Step 6: Calculate the New Local Outlier Factor (NLOF)
    nlof_scores = sequence_distances / asd
    
    #Step 7: Sort by NLOF scores to identify the top m outliers
    outlier_indices = np.argsort(nlof_scores)[-m:]  # Top m outliers with highest NLOF
    
    return outlier_indices, nlof_scores

# Directory paths
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation/ft_03'
outliers_st_dev_path = os.path.join(path, 'KNN+LOF')
os.makedirs(outliers_st_dev_path, exist_ok=True)

# List and sort CSV files
files = sorted(glob.glob(path + '/*.csv'))

# Results storage
results = []
merged_data = pd.DataFrame()

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame()

    # Perform kNN-NLOF outlier detection
    outlier_indices, nlof_scores = knn_nlof_outlier_detection(data, k=100, m=100)

    # Store results: Save the outlier indices and corresponding scores
    result_entry = {
        "file": file_basename,
        "outlier_indices": outlier_indices.tolist(),
        "nlof_scores": nlof_scores.tolist()
    }
    results.append(result_entry)
    
    # Add outlier flag to the data
    df['is_outlier'] = 0
    df.loc[outlier_indices, 'is_outlier'] = 1
    
    # Append the dataframe with outliers marked to the merged data
    df['source_file'] = file_basename  # Add a column to indicate the source file
    merged_data = pd.concat([merged_data, df], ignore_index=True)
    
    # Visualization
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

# Save the results to a CSV file
results_df = pd.DataFrame(results)
outliers_summary_path = os.path.join(outliers_st_dev_path, 'outliers_summary_NLOF.csv')
results_df.to_csv(outliers_summary_path, index=False)

# Save the merged data with outlier columns to a CSV file
merged_df_path = os.path.join(outliers_st_dev_path, 'merged_data_with_outliers.csv')
merged_data.to_csv(merged_df_path, index=False)

print(f"Results saved to {outliers_summary_path}")
print(f"Merged data saved to {merged_df_path}")
