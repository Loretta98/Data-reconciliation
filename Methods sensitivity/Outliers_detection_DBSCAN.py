import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

# Paths
base_path = 'C:\\DataRec\\FT_03'
outliers_st_dev_path = os.path.join(base_path, 'DBSCAN')
merged_data_path = os.path.join(base_path, 'merged_data', 'merged_data.csv')

# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)

# List CSV files and sort them
files = sorted(glob.glob(os.path.join(base_path, '*.csv')))
eps_values = np.arange(0.005, 1, 0.01)  # DBSCAN eps values from 0.005 to 0.2
min_samples_range = range(5,50)  # min_samples values from 2 to 20

# Initialize a list to store the results
results = []

for filename in files[:1]:  # Process only the first file
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]

    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the third column
    third_column = df.iloc[:, 2]
    data = third_column.to_frame()
    
    # Calculate rolling standard deviation to identify steady states
    window_size = 10  # Adjust window size as needed
    rolling_std = third_column.rolling(window=window_size).std()
    
    # Define steady state condition
    allowable_range = 0.02  # Adjust this value as needed
    df['steady_state'] = (rolling_std < allowable_range).astype(int)

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    for eps in eps_values:
        for min_samples in min_samples_range:
            # Apply DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
            labels = db.labels_

            # Identify outliers (DBSCAN labels outliers as -1)
            outlier_indices = np.where(labels == -1)[0]

            # Check if outliers are real (not in a steady state)
            df['anomaly'] = 0
            df.loc[outlier_indices, 'anomaly'] = 1
            df['real_outlier'] = df.apply(lambda row: 1 if row['anomaly'] == 1 and row['steady_state'] == 0 else 0, axis=1)

            # Count real outliers and steady states marked as outliers
            real_outliers_count = df['real_outlier'].sum()
            steady_states_as_outliers_count = df[(df['anomaly'] == 1) & (df['steady_state'] == 1)].shape[0]

            # Add results to the list
            results.append({
                'Filename': file_basename,
                'eps': eps,
                'min_samples': min_samples,
                'Real Outliers': real_outliers_count,
                'Steady States as Outliers': steady_states_as_outliers_count
            })

            print(f"eps: {eps}, min_samples: {min_samples}")
            print(f"Real Outliers: {real_outliers_count}, Steady States as Outliers: {steady_states_as_outliers_count}")

    # Save the updated dataframe with outlier columns
    outlier_col_name = f"{file_basename}_DBSCAN"

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv(os.path.join(outliers_st_dev_path, 'outliers_summary_DBSCAN.csv'), index=False)

