import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest

def isolation_forest_outlier_detection(filename, values, outliers_if_path, steady_state_path, clean_data_path):
    file_basename = os.path.basename(filename).split('.')[0]
    
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the third column (assumed to be the values of interest)
    third_column = df.iloc[:, 2]
    
    # Calculate rolling standard deviation
    window_size = 10  # Adjust based on your data
    rolling_std = third_column.rolling(window=window_size).std()
    
    num_intervals1 = len(third_column) // window_size
    num_intervals = np.size(rolling_std)
    allowable_range = np.ones((len(values), num_intervals+1))

    # Adjust allowable range based on the mean of each segment
    for i in range(num_intervals1):
        segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
        for j in range(window_size): 
            allowable_range[0, j+i] = values[0] * segment_mean / 100
            #allowable_range[0, j+i] = values[0] * segment_mean / 100

    # Identify steady states
    df['steady_state'] = (rolling_std < allowable_range[0, :-1]).astype(int)
    
    # Reshape the data for IsolationForest
    data = third_column.values.reshape(-1, 1)
    
    # Initialize the Isolation Forest
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    iso_forest.fit(data)
    
    # Predict the outliers
    df['anomaly'] = iso_forest.predict(data)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    
    # Collect indices for outliers from Isolation Forest
    outliers_index = df.index[df['anomaly'] == 1].tolist()
    
    # Add indices for rows where the value is 0 or outside the range [-0.2, 0.2]
    zero_out_of_range_indices = df.index[(df[df.columns[2]] == 0) | (df[df.columns[2]].abs() < 0.2)].tolist()
    
    # Combine the two lists of indices
    outliers_index.extend(zero_out_of_range_indices)
    outliers_index = sorted(set(outliers_index))  # Remove duplicates and sort the indices

    # Remove outliers and values outside the range (-0.2, 0.2)
    cleaned_df = df.drop(outliers_index).copy()

    # Save the cleaned data
    cleaned_data_filename = os.path.join(clean_data_path, f"{file_basename}_cleaned.csv")
    cleaned_df.to_csv(cleaned_data_filename, index=False, columns=[df.columns[1], df.columns[2]])

    # Plot the cleaned data
    plt.figure(figsize=(24, 10))
    plt.plot(cleaned_df[df.columns[1]], cleaned_df[df.columns[2]], label='Cleaned Data', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Cleaned Data Over Time for {file_basename}')
    plt.legend()
    plt.savefig(os.path.join(clean_data_path, f"{file_basename}_cleaned_plot.png"))
    plt.close()

    return outliers_index


# Example usage
input_directory = 'C:/DataRec/Ordered CSV/Mass Reconciliation/ft_04'
values = [3.2, 6.6, 2, 1, 0.5]

# Create directories for saving outputs
outliers_if_path = os.path.join(input_directory, 'Isolation Forest')
steady_state_path = os.path.join(input_directory, 'Steady States')
clean_data_path = os.path.join(input_directory, 'clean data')
os.makedirs(outliers_if_path, exist_ok=True)
os.makedirs(steady_state_path, exist_ok=True)
os.makedirs(clean_data_path, exist_ok=True)

# List CSV files and sort them
files = sorted(glob.glob(input_directory + '/*.csv'))
print(f'Number of CSV files found: {len(files)}')

# Process the first file and get the combined outliers index
first_filename = files[0]
outliers_index = isolation_forest_outlier_detection(first_filename, values, outliers_if_path, steady_state_path, clean_data_path)

# Apply the outliers index from the first file to all files in the "other variables" folder
input_directory_1 = 'C:/DataRec/Ordered CSV/Mass Reconciliation/ft_04/other variables'
other_files = sorted(glob.glob(input_directory_1 + '/*.csv'))

clean_data_path_other = os.path.join(input_directory_1, 'clean data')
os.makedirs(clean_data_path_other, exist_ok=True)

for filename in other_files:
    df_other = pd.read_csv(open(filename, 'rb'))
    print(filename)
    # Check if the number of rows in df_other is sufficient to match outliers_index
    
    if len(df_other) > len(outliers_index):
        cleaned_df_other = df_other.drop(outliers_index[:len(df_other)]).copy()
    else:
        cleaned_df_other = df_other.drop(outliers_index).copy()
    
    # Save the cleaned data for the "other variables" files
    file_basename_other = os.path.basename(filename).split('.')[0]
    cleaned_data_filename_other = os.path.join(clean_data_path_other, f"{file_basename_other}_cleaned.csv")
    
    cleaned_df_other.to_csv(cleaned_data_filename_other, index=False)

    # # Plot the cleaned data for other variables
    # plt.figure(figsize=(24, 10))
    # plt.scatter(cleaned_df_other[df_other.columns[1]], cleaned_df_other[df_other.columns[2]], label='Cleaned Data', color='blue')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title(f'Cleaned Data Over Time for {file_basename_other}')
    # plt.legend()
    # plt.savefig(os.path.join(clean_data_path_other, f"{file_basename_other}_cleaned_plot.png"))
    # plt.close()

print("Cleaning completed for 'other variables' files.")
