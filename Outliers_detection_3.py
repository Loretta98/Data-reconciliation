################### Outliers detection through isolation forest ###################

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os 
from sklearn.ensemble import IsolationForest

# List CSV files and sort them
path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024/Ordered CSV/Mass Reconciliation'
outliers_st_dev_path = os.path.join(path, 'Isolation Forest')
# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
files = sorted(glob.glob(path + '/*.csv'))

for filename in files:
    print(f"Processing file: {filename}")
    file_basename = os.path.basename(filename).split('.')[0]
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the relevant column (third column)
    third_column = df.iloc[:, 2]
    
    # Reshape the data to be a 2D array as required by IsolationForest
    data = third_column.values.reshape(-1, 1)
    
    # Initialize the Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    
    # Fit the model
    iso_forest.fit(data)
    
    # Predict the outliers
    df['anomaly'] = iso_forest.predict(data)
    
    # -1 for outliers and 1 for inliers, converting to 0 for inliers and 1 for outliers
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    
    # Save the result to a new CSV file
    output_file = os.path.join(outliers_st_dev_path, f"{file_basename}_with_outliers.csv")
    df.to_csv(output_file, index=False)
    
    # Print the number of anomalies detected
    num_anomalies = df['anomaly'].sum()
    print(f"Number of anomalies detected in {file_basename}: {num_anomalies}")
    
    # Optionally plot the data
    plt.figure(figsize=(24, 10))
    plt.scatter(range(len(third_column)), third_column, c=df['anomaly'], cmap='coolwarm', label='Data points') # Assuming first column is the timeframe
    plt.xlabel('Timeframe')
    plt.ylabel('Value')
    plt.title(f'Isolation Forest Outlier Detection for {file_basename}')
    plt.legend()
    plt.savefig(os.path.join(outliers_st_dev_path, f"{file_basename}_outliers_plot.png"))
    plt.close()
