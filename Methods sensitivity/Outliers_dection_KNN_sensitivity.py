################### Outliers detection through KNN method ###################
import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def knn_outlier_detection(input_path, window_size_range, values):
    path = input_path
    outliers_knn_path = os.path.join(path, 'KNN_Sens')
    
    # Create the 'KNN' directory if it does not exist
    os.makedirs(outliers_knn_path, exist_ok=True)
    
    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    # Print the number of files found
    print(f'Number of CSV files found: {len(files)}')

    # Initialize a list to store the results
    results = []

    # Iterate over each window size
    for window_size in window_size_range:
        print(f"Processing window_size: {window_size}")

        # Iterate over each file
        for k, filename in enumerate(files):
            print(f"Processing file: {filename}")
            file_basename = os.path.basename(filename).split('.')[0]
            
            # Read CSV file
            df = pd.read_csv(open(filename, 'rb'))
            # Extract the third column
            third_column = df.iloc[:, 2]
            data = third_column.to_frame()
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            knn = NearestNeighbors(n_neighbors=window_size)  # Set window_size as n_neighbors
            knn.fit(data_scaled)
            distances, indices = knn.kneighbors(data_scaled)
            
            # Computing the maximum distance to the k-nearest neighbors
            max_distances = distances.max(axis=1)
            distance = max_distances
            
            # Determine allowable range and threshold
            allowable_range = np.ones(len(third_column)) * values / 100
            threshold = 3 * allowable_range
            
            # Identify outliers
            outliers = distance > threshold
            outlier_indices = np.where(outliers)[0]
            
            # Add results to the list
            results.append({
                'Filename': file_basename,
                'Window_Size': window_size,
                'Number of Outliers': len(outlier_indices),
                'Outlier Indices': outlier_indices.tolist()
            })
            
            print(f"Number of outliers detected: {len(outlier_indices)}")
        
            # Plot the data and highlight outliers
            plt.figure(figsize=(24, 10))
            plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
            plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
            plt.title(f"Data and Outliers in {file_basename} (Window Size = {window_size})")
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
    results_df.to_csv(os.path.join(outliers_knn_path, 'outliers_summary_KNN.csv'), index=False)

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation/ft_03'
window_size_range = np.arange(5, 306, 20)  # Range from 5 to 300 in steps of 20
values = 2  # Single value since only one file is processed

knn_outlier_detection(input_directory, window_size_range, values)
