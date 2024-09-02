

import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def knn_outlier_detection(input_path, window_sizes, values):#, merged_df_path):
    path = input_path
    outliers_knn_path = os.path.join(path, 'KNN')
    knn_cleaned_data_path = os.path.join(path, 'KNN cleaned data')

    # Create the directories if they do not exist
    os.makedirs(outliers_knn_path, exist_ok=True)
    os.makedirs(knn_cleaned_data_path, exist_ok=True)
    
    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    print(f'Number of CSV files found: {len(files)}')

    results = []

    # Load the merged_data file
    #merged_df = pd.read_csv(merged_df_path)
    
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
        
        # KNN model
        knn = NearestNeighbors(n_neighbors=window_sizes)
        knn.fit(data_scaled)
        distances, indices = knn.kneighbors(data_scaled)
        
        # Compute the maximum distance to the k-nearest neighbors
        max_distances = distances.max(axis=1)
        distance = max_distances
        
        # Determine allowable range and threshold
        if k == 0 or k == 1: 
            allowable_range = third_column * values / 100
        else: 
            allowable_range = np.ones(len(third_column)) * values / 100
        
        threshold = 3 * allowable_range
        
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
        
        # # Add outlier column to merged_data
        # outlier_col_name = f"{file_basename}_KNN"
        # merged_df[outlier_col_name] = 0
        # merged_df.loc[outlier_indices, outlier_col_name] = 1
        
        # # Plot the data and highlight outliers
        # plt.figure(figsize=(24, 10))
        # plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
        # plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
        # plt.title(f"Data and Outliers in {file_basename}")
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.legend()
        # # Save the plot
        # plot_path = os.path.join(outliers_knn_path, f"plot_outliers_knn_{file_basename}.png")
        # plt.savefig(plot_path)
        # plt.close()
        
        # Clean the data by removing outliers
        cleaned_df = df.drop(outlier_indices).copy()

        # Save the cleaned data
        cleaned_data_filename = os.path.join(knn_cleaned_data_path, f"{file_basename}_knn_cleaned.csv")
        cleaned_df.to_csv(cleaned_data_filename, index=False, columns=[df.columns[1], df.columns[2]])
        
        # Plot the cleaned data
        plt.figure(figsize=(24, 10))
        plt.scatter(cleaned_df[df.columns[1]], cleaned_df[df.columns[2]], label='Cleaned Data', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cleaned Data Over Time for {file_basename}')
        plt.legend()
        plt.savefig(os.path.join(knn_cleaned_data_path, f"{file_basename}_knn_cleaned_plot.png"))
        plt.close()
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv(os.path.join(outliers_knn_path, 'outliers_summary_KNN.csv'), index=False)
    
    # # Save the merged data with outlier columns to a CSV file
    # merged_df.to_csv(merged_df_path, index=False)

# Example usage
input_directory = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation/ft_03'
# merged_data_path = os.path.join(input_directory, 'merged_data', 'merged_data.csv')
window_sizes = 300 
values = 2 

knn_outlier_detection(input_directory, window_sizes, values)#, merged_data_path)
