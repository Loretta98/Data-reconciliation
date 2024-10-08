import numpy as np
import pandas as pd
import glob
import os
from sklearn.ensemble import IsolationForest

def isolation_forest_outlier_detection(input_path, values, contamination_list, random_states_list):
    path = input_path
    outliers_if_path = os.path.join(path, 'Isolation Forest')
    steady_state_path = os.path.join(path, 'Steady States')
    results_path = os.path.join(path, 'Results_with_Outliers_and_SteadyStates')
    
    # Create directories if they do not exist
    os.makedirs(outliers_if_path, exist_ok=True)
    os.makedirs(steady_state_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # List CSV files and sort them
    files = sorted(glob.glob(path + '/*.csv'))
    print(f'Number of CSV files found: {len(files)}')

    results = []

    for k, filename in enumerate(files):
        print(f"Processing file: {filename}")
        file_basename = os.path.basename(filename).split('.')[0]
        
        # Read CSV file
        df = pd.read_csv(open(filename, 'rb'))
        second_column = df.iloc[:, 1]  # Assuming second column is the time column
        third_column = df.iloc[:, 2]
        
        window_size = 10  # Adjust based on your data
        rolling_std = third_column.rolling(window=window_size).std()

        num_intervals1 = len(third_column) // window_size
        num_intervals = np.size(rolling_std)
        allowable_range = np.ones([1,num_intervals+1])

        for i in range(num_intervals1):
            segment_mean = np.mean(third_column[i * window_size:(i + 1) * window_size])
            for j in range(window_size): 
                allowable_range[k, j+i] = values/100

        df['steady_state'] = (rolling_std < allowable_range[k, :-1]).astype(int)
        data = third_column.values.reshape(-1, 1)

        for contamination in contamination_list:
            random_state = random_states_list
            # Initialize the Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
            iso_forest.fit(data)
            df['anomaly'] = iso_forest.predict(data)
            df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

            # Cross-check: If a point is both a steady state and an outlier, it's not a real outlier
            df['real_outlier'] = df.apply(lambda row: 1 if row['anomaly'] == 1 and row['steady_state'] == 0 else 0, axis=1)

            # Count real outliers and steady states marked as outliers
            real_outliers_count = df['real_outlier'].sum()
            steady_states_as_outliers_count = df[(df['anomaly'] == 1) & (df['steady_state'] == 1)].shape[0]
            
            print(f"Contamination: {contamination}, Random State: {random_state}")
            print(f"Real Outliers: {real_outliers_count}, Steady States as Outliers: {steady_states_as_outliers_count}")

            # Append results
            results.append({
                'Filename': file_basename,
                'Contamination': contamination,
                'Random State': random_state,
                'Real Outliers': real_outliers_count,
                'Steady States as Outliers': steady_states_as_outliers_count
            })

            # Save the original data with added steady_state and real_outlier columns
            output_filename = os.path.join(results_path, f"{file_basename}_cont{contamination}_rs{random_state}_with_outliers_and_steady_states.csv")
            df[['steady_state', 'real_outlier']].to_csv(output_filename, index=False)
            print(f"Results saved to {output_filename}")

    # Save the summary of results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outliers_if_path, 'real_outliers_summary_IF_PROVA.csv'), index=False)
    print("Real outliers summary saved to CSV.")

# Example usage
input_directory = 'C:/DataRec/FT_03'
values = 2 
contamination_list = np.arange(0.01, 0.51, 0.01) # should be between zero and 0.5

random_states_list = 42

isolation_forest_outlier_detection(input_directory, values, contamination_list, random_states_list)
