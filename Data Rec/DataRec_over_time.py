from Mass_reconciliation_minimizer import MassReconciliation, Lagrangian
import numpy as np 
import glob 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from Data_Rec_Lin_lagrangian import MassReconciliation_Projection, MassReconciliation_Abs
import statistics
from scipy.linalg import qr
import seaborn as sns 
import scipy.stats as stats 


# Define the path where your CSV files are located
#path = 'C:/DataRec/Ordered CSV/Mass Reconciliation/DATA TO RECONCILE'
path = r'C:\DataRec\To_reconcile\fortyfive'
files = sorted(glob.glob(path + '/*.csv'))
reconciled_data_path = os.path.join(path, 'clean_data_lagrangian_abs')

# Ensure the directory for clean data exists
os.makedirs(reconciled_data_path, exist_ok=True)

# Covariance Matrix V according to Narasimhan et al.: diagonal elements = variance, non diagonal = null 
V = np.zeros([5,5])
#V = np.identity(5)
# Mapping of variables to filenames
variable_order = ['F1', 'F2', 'F4', 'F6', 'F7']
variable_to_file = dict(zip(variable_order, files))

# Initialize lists to hold the extracted data for each variable
extracted_data = {var: [] for var in variable_order}
time_values = None  # Assuming all files have the same time column

# Extract the second column for each variable and store it
for i, (var, filename) in enumerate(variable_to_file.items()):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Extract the first column (assumed to be time)
    if time_values is None:
        time_values = df.iloc[:, 0].values  # Store time once
    
    # Extract the second column (values to reconcile)
    extracted_data[var] = df.iloc[:, 1].values
    
    # Update the diagonal of the covariance matrix with the variance
    V[i, i] = 1/np.var(extracted_data[var])  # Variance for variable var
    #V[i, i] = np.std(extracted_data[var])  # Variance for variable var

V[1,1] = V[0,0] 
print(V)

# Reorder the extracted data according to x0 = F1, F2, F4, F6, F7
x0_order = ['F1', 'F2', 'F4', 'F6', 'F7']
y0_order = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

# Initialize a list to hold the reconciled values and errors for each variable over time
reconciled_values = {var: [] for var in y0_order}
errors = {var: [] for var in x0_order}

MW_gas = 0.6*(12+4) +0.4*(12+16*2)  #g/mol
rho_w = 997/1000    #kg/lt
#F1,F2,F4,F6,F7
conversion_factor = np.array([ 0.044*MW_gas ,rho_w, rho_w, 1,1]) # Nm3/h, lt/h, lt/h, kg/h, kg/h
# Balance Matrix of measured variables F1,F2,F4,F6,F7
Ay = np.array([ [1, 1, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, -1, -1]])

# Balance Matrix of unmeasured variables F3,F5
Az = np.array([ [-1, 0],
                [1, -1],
                [0, 1],])

Q,R  = qr(Az)
# Q = [Q1 Q2], Q2 has the shape = Q.shape-R.rank
Q2 = Q[:,-1]
Q1 = Q[:,:2]
P = Q2.T 
PA2 = np.dot(P,Az)
# Calculate G = PA1
G = np.dot(P, Ay)

# Loop over each time step to solve the mass data reconciliation
for t in range(len(time_values)):
    # Extract the values for each variable at this time step
    x0 = np.array([extracted_data[var][t] for var in x0_order])
    
    # Perform mass reconciliation for this time step
    Fr, Eps = MassReconciliation_Projection(x0, x0,V,conversion_factor,Ay,Az,G,Q1,R)
    #Fr, Eps = MassReconciliation_Abs(x0, x0,V,conversion_factor,Ay,Az,G,Q1,R,P)

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

# Example: Convert time_values to numeric (elapsed time in seconds or minutes)
if isinstance(time_values[0], str):  # If time_values is a list of strings
    #time_values = pd.to_datetime(time_values,format='%d.%m.%Y %H:%M:%S', dayfirst=True)  # Convert to datetime objects
    time_values = pd.to_datetime(time_values, format='%Y-%m-%d %H:%M:%S', dayfirst=False)
    time_values_numeric = (time_values - time_values[0]).total_seconds()  # Convert to elapsed seconds

# Set up a grid for subplots
fig, axs = plt.subplots(len(error_df.columns)-1, 1, figsize=(14, 20))  # Two columns: one for error over time, another for normal distribution

for i, var in enumerate(['Error F1', 'Error F2', 'Error F4', 'Error F6', 'Error F7']):
    # Plot the error over time
    axs[i].plot(time_values, error_df[var], label=f'{var} over Time')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Error Value')
    axs[i].set_title(f'{var} Over Time')
    axs[i].legend()
    axs[i].grid(True)
    
    # # Fit and plot the normal distribution of the errors
    # mean = np.mean(error_df[var])
    # std_dev = np.std(error_df[var])
    # normal_dist = stats.norm.pdf(time_values_numeric, mean, std_dev)
    
    # axs[i, 1].plot(error_df[var], normal_dist, label=f'Normal Distribution of {var}')
    # axs[i, 1].set_xlabel('Error')
    # axs[i, 1].set_ylabel('Density')
    # axs[i, 1].set_title(f'Normal Distribution of {var}')
    # axs[i, 1].legend()
    # axs[i, 1].grid(True)

# Adjust layout for better spacing
# plt.figure()
# ORDER = ['F1','F2']
# for var in ORDER:
#     plt.plot(time_values, reconciled_values[var], label=f'Reconciled {var}')
# plt.plot(time_values, reconciled_values['F2']/reconciled_values['F1'])

plt.tight_layout()
plt.show()

# # Plotting the errors
# plt.figure(figsize=(10, 6))
# for var in x0_order:
#     plt.plot(time_values, errors[var], label=f'Error {var}')
# plt.xlabel('Time')
# plt.ylabel('Error Values')
# plt.title('Error Data Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting the normal distribution of errors
# plt.figure(figsize=(10, 6))
# for var in ['Error F1', 'Error F2', 'Error F4', 'Error F6', 'Error F7']:
#     sns.kdeplot(error_df[var], label=f'Normal Distribution of {var}')

# plt.xlabel('Error Values')
# plt.ylabel('Density')
# plt.title('Normal Distribution of Errors')
# plt.legend()
# plt.grid(True)
# plt.show()