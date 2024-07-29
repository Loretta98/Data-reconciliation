################### Outliers detection through KNN method ###################

import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os 


# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
outliers_st_dev_path = os.path.join(path, 'KNN')
# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))

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

    # The method to weight the distance for the KNN can be based on the mean or max distance

    # # Using the mean distance to the k-nearest neighbors
    # mean_distances = distances.mean(axis=1)
    
    # Using the maximum distance to the k-nearest neighbors
    max_distances = distances.max(axis=1)

    distance = max_distances
    # Set a threshold for what you consider an outlier
    threshold = np.mean(distance) + 2 * np.std(distance)

    # Identify outliers
    outliers = distance > threshold
    outlier_indices = np.where(outliers)[0]
    outliers_report_path = os.path.join(outliers_st_dev_path, f"outliers_report_{file_basename}.txt")
    with open(outliers_report_path, 'w') as report_file:
            report_file.write("Outlier Indices :\n")
            report_file.write(f"Index: {outlier_indices} \n"); report_file.write(f"Number of outliers: {len(outlier_indices)}\n")
    print(f"Number of outliers detected: {len(outlier_indices)}")
    #print("Indices of outliers:", outlier_indices)
    #print("Outlier data points:", third_column.iloc[outlier_indices])
        # Plot the data and highlight outliers
    plt.figure(figsize=(24, 10))
    plt.scatter(data.index, data.iloc[:, 0], label='Data', color='blue')
    plt.scatter(data.index[outlier_indices], data.iloc[outlier_indices, 0], color='red', label='Outliers')
    plt.title(f"Data and Outliers in {file_basename}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    #plt.show()

    # Save the plot
    plot_path = os.path.join(outliers_st_dev_path, f"plot_outliers_st_dev_{file_basename}.png")
    plt.savefig(plot_path)
    plt.close()