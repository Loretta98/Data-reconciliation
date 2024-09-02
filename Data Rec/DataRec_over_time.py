from Mass_reconciliation_minimizer import MassReconciliation
import numpy as np 
import glob 
import os 
import pandas as pd
import matplotlib.pyplot as plt

# Define the path where your CSV files are located
path = 'C:/DataRec/Ordered CSV/Mass Reconciliation/DATA TO RECONCILE'
files = sorted(glob.glob(path + '/*.csv'))
reconciled_data_path = os.path.join(path, 'clean data')

# Ensure the directory for clean data exists
os.makedirs(reconciled_data_path, exist_ok=True)

# Mapping of variables to filenames
variable_order = ['F4', 'F6', 'F7', 'F1', 'F2']
variable_to_file = dict(zip(variable_order, files))

# Initialize lists to hold the extracted data for each variable
extracted_data = {var: [] for var in variable_order}
time_values = None  # Assuming all files have the same time column

# Extract the second column for each variable and store it
for var, filename in variable_to_file.items():
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Extract the first column (assumed to be time)
    if time_values is None:
        time_values = df.iloc[:, 0].values  # Store time once
    
    # Extract the second column (values to reconcile)
    extracted_data[var] = df.iloc[:, 1].values

# Reorder the extracted data according to x0 = F1, F2, F4, F6, F7
x0_order = ['F1', 'F2', 'F4', 'F6', 'F7']
y0_order = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

# Initialize a list to hold the reconciled values and errors for each variable over time
reconciled_values = {var: [] for var in y0_order}
errors = {var: [] for var in y0_order}

# Loop over each time step
for t in range(len(time_values)):
    # Extract the values for each variable at this time step
    x0 = np.array([extracted_data[var][t] for var in x0_order])
    
    # Perform mass reconciliation for this time step
    Fr, Eps = MassReconciliation(x0, x0)
    
    # Store the reconciled values and errors for each variable
    for i, var in enumerate(y0_order):
        reconciled_values[var].append(Fr[i])
    for i,var in enumerate(x0_order):
        errors[var].append(Eps[i])

# Convert the reconciled values to a DataFrame for saving
output_df = pd.DataFrame({
    'Time': time_values,
    'Reconciled F1': reconciled_values['F1'],
    'Reconciled F2': reconciled_values['F2'],
    'Reconciled F3': reconciled_values['F3'],
    'Reconciled F4': reconciled_values['F4'],
    'Reconciled F5': reconciled_values['F5'],
    'Reconciled F6': reconciled_values['F6'],
    'Reconciled F7': reconciled_values['F7']
})
output_csv = os.path.join(reconciled_data_path, 'reconciled_data.csv')
output_df.to_csv(output_csv, index=False)

# Convert the errors to a DataFrame for saving
error_df = pd.DataFrame({
    'Time': time_values,
    'Error F1': errors['F1'],
    'Error F2': errors['F2'],
    'Error F4': errors['F4'],
    'Error F6': errors['F6'],
    'Error F7': errors['F7']
})
error_csv = os.path.join(reconciled_data_path, 'errors.csv')
error_df.to_csv(error_csv, index=False)

# Plotting the reconciled results
plt.figure(figsize=(10, 6))
for var in y0_order:
    plt.plot(time_values, reconciled_values[var], label=f'Reconciled {var}')
plt.xlabel('Time')
plt.ylabel('Reconciled Values')
plt.title('Reconciled Data Over Time')
plt.legend()
plt.grid(True)

# Plotting the errors
plt.figure(figsize=(10, 6))
for var in x0_order:
    plt.plot(time_values, errors[var], label=f'Error {var}')
plt.xlabel('Time')
plt.ylabel('Error Values')
plt.title('Error Data Over Time')
plt.legend()
plt.grid(True)
plt.show()
