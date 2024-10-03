import pandas as pd
import numpy as np
import glob
import os
from Opt_Am_problem import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from Data_Rec_Lin_lagrangian import*
from scipy.linalg import qr

# Define paths where your CSV files are located
flowrate_path = 'C:/DataRec/To_reconcile/clean_data_lagrangian/Reconciled'
reconciled_path = 'C:/DataRec/To_reconcile/clean_data_lagrangian/Reconciled/results'
composition_path = 'C:/DataRec/To_reconcile/GC_filtered'
flowrate_files = sorted(glob.glob(flowrate_path + '/*.csv'))
composition_files = sorted(glob.glob(composition_path + '/*.csv'))

time_values = None  # Assuming all files have the same time column

# Covariance Matrix V according to Narasimhan et al.: diagonal elements = variance, non diagonal = null 
V = np.zeros([5,5])
# Covariance Matrixes of measurements over CH4, CO2, CO, H2, H2O 
Wc = np.zeros([5,5])
Wh = np.zeros([5,5])
Wo = np.zeros([5,5])

# Mapping stream names to filenames
variable_order = ['F2', 'F4','F5', 'F6', 'F7']
x0_order = [ 'F2', 'F4','F5', 'F6', 'F7']
flowrate_files_dict = dict(zip(variable_order, flowrate_files))
composition_files_dict = dict(zip(variable_order, composition_files))

# Initialize lists to hold the extracted data for each variable
extracted_data = {var: [] for var in variable_order}

# Molar masses of the compounds (g/mol)
molar_masses = {
    'CH4': 16.04,     # g/mol
    'CO': 28.01,      # g/mol
    'CO2': 44.01,     # g/mol
    'CH3OH': 32.04,   # g/mol
    'H2': 2.016,      # g/mol
    'O2': 32.00,      # g/mol
    'H2O': 18.01528   # g/mol
}

# Initialize dictionaries to hold Nc, Nh, and No data for each stream
Nc_dict = {}
Nh_dict = {}
No_dict = {}

# Contribution factors per molecule
nC_CH4 = 1
nC_CO = 1
nC_CO2 = 1
nC_CH3OH = 1

nH_CH4 = 4
nH_H2 = 2
nH_CH3OH = 4
nH_H2O = 2

nO_O2 = 2
nO_CO = 1
nO_CO2 = 2
nO_CH3OH = 1
nO_H2O = 1

F_mean = np.zeros(5)
F_std = np.zeros(5)

# Read data for each stream
for k,var in enumerate(variable_order):
    #print('Flowrate', var); print('Mass csv',flowrate_files[k]); print('Composition csv',composition_files[k])
    # Read mass flowrate data
    flowrate_df = pd.read_csv(flowrate_files_dict[var])
    # Extract the first column (assumed to be time)
    if time_values is None:
        time_values = flowrate_df.iloc[:-1, 0].values  # Store time o
    # Extract the second column (values to reconcile)
    extracted_data[var] = flowrate_df.iloc[:, 1].values
    
    mass_flowrates = flowrate_df.iloc[:, 1].values  # Assuming flowrates are in the second column
    rho_w = 997/1000    #kg/lt
    #F1,F2,F4,F6,F7
    #conversion_factor = np.array([0.044,rho_w, rho_w, 1,1]) # Nm3/h, lt/h, lt/h, kg/h, kg/h
    # F2, F4, F5, F6, F7 
    conversion_factor = np.array([rho_w, rho_w, 1, 1,1])
    mass_flowrates = mass_flowrates*conversion_factor[k]
    # Read composition data
    composition_df = pd.read_csv(composition_files_dict[var])
    # Extract molar fractions and convert from percentage to fraction
    xCH4 = composition_df['CH4 (A) [%Vol]'].values / 100
    xCO = composition_df['CO (A) [%]'].values / 100
    xCO2 = composition_df['CO2 (B) [%Vol]'].values / 100
    #xCH3OH = composition_df['CH3OH (B) [%]'].values / 100
    xH2 = composition_df['H2 (A) [%]'].values / 100
    #xO2 = composition_df['O2 (A) [%Vol]'].values / 100
    xH2O = composition_df['H2O [%Vol]'].values / 100

        # Calculate the total molar mass using the composition values
    MW = (xCH4 * molar_masses['CH4']/1000 +
                        xCO * molar_masses['CO']/1000 +
                        xCO2 * molar_masses['CO2']/1000 +
                        xH2 * molar_masses['H2']/1000 +
                        xH2O * molar_masses['H2O']/1000)  #kg/mol
    
    # Convert mass flowrate to molar flowrate: Fi = Mi / (sum(xj * MWj))
    molar_flowrates = mass_flowrates[:-1] / MW # mol/h

    # Calculate Nc, Nh, and No using molar flowrates and molar fractions
    Nc = molar_flowrates * (xCH4 * nC_CH4 + xCO * nC_CO + xCO2 * nC_CO2) # + xCH3OH * nC_CH3OH)
    Nh = molar_flowrates * (xCH4 * nH_CH4 + xH2 * nH_H2  + xH2O * nH_H2O)
    No = molar_flowrates * (xCO * nO_CO + xCO2 * nO_CO2  + xH2O * nO_H2O)

    # Store results in dictionaries
    Nc_dict[var] = Nc
    Nh_dict[var] = Nh
    No_dict[var] = No
    
    # Update the diagonal of the covariance matrix with the variance of the mass flowrate [kg/h]
    V[k, k] = np.var(mass_flowrates)  # Variance for variable var
    
    # To avoid big values of the covariance matrixes, trial with only st.dev. TENTATIVO
    #V[k, k] = np.std(mass_flowrates)  # Variance for variable var
    
    F_mean[k], F_std[k] = np.mean(mass_flowrates), np.std(mass_flowrates)

# Create DataFrames to store results
Nc_df = pd.DataFrame(Nc_dict)
Nh_df = pd.DataFrame(Nh_dict)
No_df = pd.DataFrame(No_dict)

# Save each DataFrame to separate CSV files
Nc_output_path = os.path.join(flowrate_path, 'Nc.csv')
Nh_output_path = os.path.join(flowrate_path, 'Nh.csv')
No_output_path = os.path.join(flowrate_path, 'No.csv')

Nc_df.to_csv(Nc_output_path, index=False)
Nh_df.to_csv(Nh_output_path, index=False)
No_df.to_csv(No_output_path, index=False)

print(f"CSV files 'Nc.csv', 'Nh.csv', and 'No.csv' have been created at {flowrate_path}")

# Loop through each stream to calculate variance and populate W matrices
for idx, var in enumerate(variable_order):
    # Calculate variance for each stream in Nc, Nh, and No
    Wc[idx, idx] = np.var(Nc_dict[var])
    Wh[idx, idx] = np.var(Nh_dict[var])
    Wo[idx, idx] = np.var(No_dict[var])
    # TENTATIVO 
    # Wc[idx, idx] = np.std(Nc_dict[var])
    # Wh[idx, idx] = np.std(Nh_dict[var])
    # Wo[idx, idx] = np.std(No_dict[var])

print('V',V); print('Wc',Wc); print('Wh',Wh); print('Wo',Wo)

# Define the paths for the reconciled data files
Nc_m_path = os.path.join(flowrate_path, 'Nc.csv')
Nh_m_path = os.path.join(flowrate_path, 'Nh.csv')
No_m_path = os.path.join(flowrate_path, 'No.csv')

# Load measured Nc, Nh, and No from CSV files
Nc_m_df = pd.read_csv(Nc_m_path)
Nh_m_df = pd.read_csv(Nh_m_path)
No_m_df = pd.read_csv(No_m_path)

F_list = []
Nc_list = []
Nh_list = []
No_list = []
steps = []

Nc_mean = np.zeros(5); Nh_mean = np.zeros(5); No_mean = np.zeros(5)
Nc_std = np.zeros(5); Nh_std = np.zeros(5); No_std  = np.zeros(5)

for i in range (0,5): # Scaling the Ni variables (mean and std can be calculated from the measured data, one for each csv column)
    Nc_mean[i], Nc_std[i] = np.mean(Nc_m_df.iloc[:,i]), np.std(Nc_m_df.iloc[:,i])
    Nh_mean[i], Nh_std[i] = np.mean(Nh_m_df.iloc[:,i]), np.std(Nh_m_df.iloc[:,i])
    No_mean[i], No_std[i] = np.mean(No_m_df.iloc[:,i]), np.std(No_m_df.iloc[:,i])


# First loop for the carbon molecules, the matrixes are the same 

# Balance Matrix of measured variables F2,F4,F5,F6,F7
Ay = np.array([ [1, 0, 0, 0, 0],
                [0, -1, -1, 0, 0],
                [0, 0, 1, -1, -1]])

# Balance Matrix of unmeasured variables F1,F3
Az = np.array([ [1, -1],
                [0, 1],
                [0, 0]])
Q,R  = qr(Az)
# Q = [Q1 Q2], Q2 has the shape = Q.shape-R.rank
Q2 = Q[:,-1]
Q1 = Q[:,:2]
P = Q2.T 
PA2 = np.dot(P,Az)
# Calculate G = PA1
G = np.dot(P, Ay)

y0_order = ['N1C', 'N2C', 'N3C', 'N4C', 'N5C', 'N6C', 'N7C']
reconciled_values = {var: [] for var in y0_order}
errors = {var: [] for var in x0_order}

# The integration must be carried out three times 
# Loop over each time step to solve the mass data reconciliation
for t in range(len(time_values)):
    # Placeholder measured data (replace with actual measured values)
    Nc_m = Nc_m_df.iloc[t].values  # Get Nc measured values for F1, F2, F4, F6, and F7
    # Due to the large difference scale of the parameters and values, it is better to work on scaled factors
    #Nc_0 = np.abs((Nc_m-Nc_mean) / Nc_std)

    # Replace NaNs with zero
    #Nc_0 = replace_nan_with_zero(Nc_0)
    x0 = np.array(Nc_m)
    Fr, Eps = AtomicRec_lin(x0, x0,Wc,Ay,Az,G,Q1,R,P)
    for i, var in enumerate(y0_order):
        reconciled_values[var].append(Fr[i])
    for i,var in enumerate(x0_order):
        errors[var].append(Eps[i])

print('Succesfull reconciliation for NC')
# Convert the reconciled values to a DataFrame for saving
output_df = pd.DataFrame({
    'Time': time_values,
    'Reconciled N1C': reconciled_values['N1C'],
    'Reconciled N2C': reconciled_values['N2C'],
    'Reconciled N3C': reconciled_values['N3C'],
    'Reconciled N4C': reconciled_values['N4C'],
    'Reconciled N5C': reconciled_values['N5C'],
    'Reconciled N6C': reconciled_values['N6C'],
    'Reconciled N7': reconciled_values['N7C']
})
# Save to CSV
output_csv = os.path.join(reconciled_path, 'Carbon_reconciled.csv')
output_df.to_csv(output_csv, index=False)
# Convert the errors to a DataFrame for saving
error_df = pd.DataFrame({
    'Time': time_values,
    'Error N2C': errors['F2'],
    'Error N4C': errors['F4'],
    'Error N5C': errors['F5'],
    'Error N6C': errors['F6'],
    'Error N7C': errors['F7']
})
output_csv = os.path.join(reconciled_path, 'Carbon_error.csv')
error_df.to_csv(output_csv, index=False)

y0_order = ['N1H', 'N2H', 'N3H', 'N4H', 'N5H', 'N6H', 'N7H']
reconciled_values = {var: [] for var in y0_order}
errors = {var: [] for var in x0_order}
# Loop over each time step to solve the mass data reconciliation
for t in range(len(time_values)):
    # Placeholder measured data (replace with actual measured values)
    Nh_m = Nh_m_df.iloc[t].values  # Get Nh measured values for F2, F4, F5, F6, and F7

    # Due to the large difference scale of the parameters and values, it is better to work on scaled factors
    #Nh_0 = np.abs((Nh_m-Nh_mean)/ Nh_std)
    # Replace NaNs with zero
    #Nh_0 = replace_nan_with_zero(Nh_0)
    
    x0 = np.array(Nh_m)

    Nr, Eps = AtomicRec_lin(x0, x0, Wh ,Ay,Az,G,Q1,R,P)
    # Store the reconciled values and errors for each variable
    for i, var in enumerate(y0_order):
        reconciled_values[var].append(Fr[i])
    for i,var in enumerate(x0_order):
        errors[var].append(Eps[i])
# Convert the reconciled values to a DataFrame for saving (Nitrogen and Oxygen)

print('Succesfull reconciliation for NH')
output_df = pd.DataFrame({
    'Time': time_values,
    'Reconciled N1H': reconciled_values['N1H'],
    'Reconciled N2H': reconciled_values['N2H'],
    'Reconciled N3H': reconciled_values['N3H'],
    'Reconciled N4H': reconciled_values['N4H'],
    'Reconciled N5H': reconciled_values['N5H'],
    'Reconciled N6H': reconciled_values['N6H'],
    'Reconciled N7': reconciled_values['N7H']
})
output_csv = os.path.join(reconciled_path, 'Hydrogen_reconciled.csv')
output_df.to_csv(output_csv, index=False)
# Convert the errors to a DataFrame for saving (Oxygen)
error_df = pd.DataFrame({
    'Time': time_values,
    'Error N2O': errors['F2'],
    'Error N4O': errors['F4'],
    'Error N5O': errors['F5'],
    'Error N6O': errors['F6'],
    'Error N7O': errors['F7']
})
output_csv = os.path.join(reconciled_path, 'Hydrogen_error.csv')
error_df.to_csv(output_csv, index=False)

y0_order = ['N1O', 'N2O', 'N3O', 'N4O', 'N5O', 'N6O', 'N7O']
reconciled_values = {var: [] for var in y0_order}
errors = {var: [] for var in x0_order}

# Loop over each time step to solve the mass data reconciliation
for t in range(len(time_values)):
    # Placeholder measured data (replace with actual measured values)
    No_m = No_m_df.iloc[t].values  # Get No measured values for F1, F2, F4, F6, and F7
    
    # Due to the large difference scale of the parameters and values, it is better to work on scaled factors
    #No_0 = np.abs((No_m-No_mean) / No_std) 
    # Replace NaNs with zero
    #No_0 = replace_nan_with_zero(No_0)
    
    x0 = np.array(No_m)
    Fr, Eps = AtomicRec_lin(x0, x0, Wo ,Ay,Az,G,Q1,R,P)
    for i, var in enumerate(y0_order):
        reconciled_values[var].append(Fr[i])
    for i,var in enumerate(x0_order):
        errors[var].append(Eps[i])

print('Succesfull reconciliation for NO')
# Convert the reconciled values to a DataFrame for saving
output_df = pd.DataFrame({
    'Time': time_values,
    'Reconciled N1O': reconciled_values['N1O'],
    'Reconciled N2O': reconciled_values['N2O'],
    'Reconciled N3O': reconciled_values['N3O'],
    'Reconciled N4O': reconciled_values['N4O'],
    'Reconciled N5O': reconciled_values['N5O'],
    'Reconciled N6O': reconciled_values['N6O'],
    'Reconciled N7O': reconciled_values['N7O']
})
# Save to CSV
output_csv = os.path.join(reconciled_path, 'Oxygen_reconciled.csv')
output_df.to_csv(output_csv, index=False)
# Convert the errors to a DataFrame for saving (Oxygen)
error_df = pd.DataFrame({
    'Time': time_values,
    'Error N2O': errors['F2'],
    'Error N4O': errors['F4'],
    'Error N5O': errors['F5'],
    'Error N6O': errors['F6'],
    'Error N7O': errors['F7']
})
output_csv = os.path.join(reconciled_path, 'Oxygen_error.csv')
error_df.to_csv(output_csv, index=False)



