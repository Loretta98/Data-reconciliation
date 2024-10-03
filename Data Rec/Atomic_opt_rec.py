import pandas as pd
import numpy as np
import glob
import os
from Opt_Am_problem import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

# Define paths where your CSV files are located
flowrate_path = 'C:/DataRec/To_reconcile'
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
variable_order = ['F1', 'F2', 'F4', 'F6', 'F7']
x0_order = ['F1', 'F2', 'F4', 'F6', 'F7']
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
    conversion_factor = np.array([0.044,rho_w, rho_w, 1,1]) # Nm3/h, lt/h, lt/h, kg/h, kg/h
    mass_flowrates = mass_flowrates*conversion_factor[k]
    # Read composition data
    composition_df = pd.read_csv(composition_files_dict[var])
    # Extract molar fractions and convert from percentage to fraction
    xCH4 = composition_df['CH4 (A) [%Vol]'].values / 100
    xCO = composition_df['CO (A) [%]'].values / 100
    xCO2 = composition_df['CO2 (B) [%Vol]'].values / 100
    xCH3OH = composition_df['CH3OH (B) [%]'].values / 100
    xH2 = composition_df['H2 (A) [%]'].values / 100
    xO2 = composition_df['O2 (A) [%Vol]'].values / 100
    xH2O = composition_df['H2O [%Vol]'].values / 100

        # Calculate the total molar mass using the composition values
    MW = (xCH4 * molar_masses['CH4']/1000 +
                        xCO * molar_masses['CO']/1000 +
                        xCO2 * molar_masses['CO2']/1000 +
                        xCH3OH * molar_masses['CH3OH']/1000 +
                        xH2 * molar_masses['H2']/1000 +
                        xH2O * molar_masses['H2O']/1000)  #kg/mol
    if k == 0:
        molar_flowrates = mass_flowrates[:-1]*1000 #mol/h
        mass_flowrates = mass_flowrates[:-1]*(0.6*16+0.4*44) #kg/h
    else: 
        # Convert mass flowrate to molar flowrate: Fi = Mi / (sum(xj * MWj))
        molar_flowrates = mass_flowrates[:-1] / MW # mol/h

    # Calculate Nc, Nh, and No using molar flowrates and molar fractions
    Nc = molar_flowrates * (xCH4 * nC_CH4 + xCO * nC_CO + xCO2 * nC_CO2 + xCH3OH * nC_CH3OH)
    Nh = molar_flowrates * (xCH4 * nH_CH4 + xH2 * nH_H2 + xCH3OH * nH_CH3OH + xH2O * nH_H2O)
    No = molar_flowrates * (xCO * nO_CO + xCO2 * nO_CO2 + xCH3OH * nO_CH3OH + xH2O * nO_H2O)

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

#print(f"CSV files 'Nc.csv', 'Nh.csv', and 'No.csv' have been created at {flowrate_path}")

# Loop through each stream to calculate variance and populate W matrices
for idx, var in enumerate(variable_order):
    # Calculate variance for each stream in Nc, Nh, and No
    # Wc[idx, idx] = np.var(Nc_dict[var])
    # Wh[idx, idx] = np.var(Nh_dict[var])
    # Wo[idx, idx] = np.var(No_dict[var])
    # TENTATIVO 
    Wc[idx, idx] = np.std(Nc_dict[var])
    Wh[idx, idx] = np.std(Nh_dict[var])
    Wo[idx, idx] = np.std(No_dict[var])


# Additional line for F5 in the covariance matrix 
n = 100 # weight for the minimization 

Wc = add_row_col(Wc,n)
Wh = add_row_col(Wh,n)
Wo = add_row_col(Wo,n)

#print('V',V); print('Wc',Wc); print('Wh',Wh); print('Wo',Wo)

# Define the paths for the reconciled data files
Nc_m_path = os.path.join(flowrate_path, 'Nc.csv')
Nh_m_path = os.path.join(flowrate_path, 'Nh.csv')
No_m_path = os.path.join(flowrate_path, 'No.csv')

# Load measured Nc, Nh, and No from CSV files
Nc_m_df = pd.read_csv(Nc_m_path)
Nh_m_df = pd.read_csv(Nh_m_path)
No_m_df = pd.read_csv(No_m_path)

F5_meas = pd.read_csv('C:\DataRec\To_reconcile\GC_filtered\excluded\F5.csv') 
xCH4_5 = F5_meas['CH4 (A) [%Vol]'].values / 100
xCO_5 = F5_meas['CO (A) [%]'].values / 100
xCO2_5 = F5_meas['CO2 (B) [%Vol]'].values / 100
xCH3OH_5 = F5_meas['CH3OH (B) [%]'].values / 100
xH2_5 = F5_meas['H2 (A) [%]'].values / 100
xO2_5 = F5_meas['O2 (A) [%Vol]'].values / 100

def wrapper_function(x, V, Wc, Wh, Wo, x_CO2_F5, x_CH4_F5, x_CO_F5,x_H2_F5, F_measured, Nc_measured, Nh_measured, No_measured):
    phi, constraints = objective(x, V, Wc, Wh, Wo, x_CO2_F5, x_CH4_F5, x_CO_F5,x_H2_F5, F_measured, Nc_measured, Nh_measured, No_measured)
    return phi

optimized_F_list = []
optimized_Nc_list = []
optimized_Nh_list = []
optimized_No_list = []
time_steps = []

Nc_mean = np.zeros(5); Nh_mean = np.zeros(5); No_mean = np.zeros(5)
Nc_std = np.zeros(5); Nh_std = np.zeros(5); No_std  = np.zeros(5)

for i in range (0,5): # Scaling the Ni variables (mean and std can be calculated from the measured data, one for each csv column)
    Nc_mean[i], Nc_std[i] = np.mean(Nc_m_df.iloc[:,i]), np.std(Nc_m_df.iloc[:,i])
    Nh_mean[i], Nh_std[i] = np.mean(Nh_m_df.iloc[:,i]), np.std(Nh_m_df.iloc[:,i])
    No_mean[i], No_std[i] = np.mean(No_m_df.iloc[:,i]), np.std(No_m_df.iloc[:,i])

# Loop over each time step to solve the mass data reconciliation
for t in range(len(time_values)):
    # Placeholder measured data (replace with actual measured values)
    Nc_m = Nc_m_df.iloc[t].values  # Get Nc measured values for F1, F2, F4, F6, and F7
    Nh_m = Nh_m_df.iloc[t].values  # Get Nh measured values for F1, F2, F4, F6, and F7
    No_m = No_m_df.iloc[t].values  # Get No measured values for F1, F2, F4, F6, and F7
    
    conversion_factor[0] = conversion_factor[0]*(0.6*16+0.4*44)
    # Extract the values for each variable at this time step
    F_m = np.array([extracted_data[var][t] for var in x0_order])*conversion_factor #kg/h
    F_0 = np.abs((F_m-F_mean)/F_std)
    
    # Composition measured values for stream F5 (replace these placeholders with actual values)
    x_CO2_F5 = xCO2_5[t]  # Mole fraction
    x_CH4_F5 = xCH4_5[t]
    x_CO_F5 = xCO_5[t]
    x_H2_F5 = xH2_5[t]
    
    F5guess  = np.abs((25-27.75703)/0.668318) #kg/h         # from the mass based reconciliation
    F3guess =  np.abs((71-71.69286)/0.4033 ) 
    
    # Calculate N5C, N5H, N5O using F5 and composition values
    MW_mix = MW_mixture_F5(x_CO2_F5, x_CH4_F5, x_CO_F5,x_H2_F5)
    
    N5C = F5guess * (x_CH4_F5 * 1 + x_CO2_F5 * 1 + x_CO_F5 * 1) * 1000 / MW_mix # mol/h
    N5H = F5guess * (x_CH4_F5 * 4) * 1000 / MW_mix
    N5O = F5guess * (x_CO2_F5 * 2 + x_CO_F5 * 1)*1000 / MW_mix
    
    # Build full arrays with N5C, N5H, N5O
    Nc_m = np.concatenate([Nc_m, [N5C]])
    Nh_m = np.concatenate([Nh_m, [N5H]])
    No_m = np.concatenate([No_m, [N5O]])

    # Build full arrays with N5C, N5H, N5O
    Nc_mean = np.concatenate([Nc_mean, [N5C]])
    Nh_mean = np.concatenate([Nh_mean, [N5H]])
    No_mean = np.concatenate([No_mean, [N5O]])

    Nc_std = np.concatenate([Nc_std, [N5C]])
    Nh_std = np.concatenate([Nh_std, [N5H]])
    No_std = np.concatenate([No_std, [N5O]])

    # Due to the large difference scale of the parameters and values, it is better to work on scaled factors
    Nc_0 = np.abs((Nc_m-Nc_mean) / Nc_std)
    Nh_0 = np.abs((Nh_m-Nh_mean)/ Nh_std)
    No_0 = np.abs((No_m-No_mean) / No_std) 

    # Replace NaNs with zero
    Nc_0 = replace_nan_with_zero(Nc_0)
    Nh_0 = replace_nan_with_zero(Nh_0)
    No_0 = replace_nan_with_zero(No_0)

    # Initial guess for F3, F5, and all Nc, Nh, No values
    initial_guess = np.concatenate([F_0, [F3guess, F5guess], Nc_0[:-1], Nh_0[:-1], No_0[:-1]])

    # Optimize
    # Pass constraints dynamically during optimization
    result = minimize(
        fun=wrapper_function,                     # Use the wrapper function for optimization
        x0=initial_guess,                         # Initial guess
        args=(V, Wc, Wh, Wo, x_CO2_F5, x_CH4_F5, x_CO_F5,x_H2_F5, F_m, Nc_m, Nh_m, No_m),  # Additional arguments
        constraints=objective(initial_guess, V, Wc, Wh, Wo, x_CO2_F5, x_CH4_F5, x_CO_F5, x_H2_F5, F_m, Nc_m, Nh_m, No_m)[1],  # Pass constraints
        method='trust-constr', options={'disp': True})#, 'ftol': 1e-6, 'maxiter': 500}
            

    # Assuming this is part of your optimization script
    if result.success:
        # If optimization is successful, do not print any results
        print('Optimization has succeded!!')
        # Extract optimized values
        optimized_values = result.x
        optimized_F = optimized_values[:7]
        optimized_Nc = optimized_values[7:12]
        optimized_Nh = optimized_values[12:17]
        optimized_No = optimized_values[17:22]

        # Append results to lists
        optimized_F_list.append(optimized_F)
        optimized_Nc_list.append(optimized_Nc)
        optimized_Nh_list.append(optimized_Nh)
        optimized_No_list.append(optimized_No)
        time_steps.append(t)

    else:
        # If optimization fails, print the error message
        print("Optimization failed:", result.message)

# Convert results to DataFrame for easier CSV saving and plotting
df = pd.DataFrame({
    'time': time_steps,
    'F1': [opt[0] for opt in optimized_F_list],
    'F2': [opt[1] for opt in optimized_F_list],
    'F3': [opt[5] for opt in optimized_F_list],
    'F4': [opt[2] for opt in optimized_F_list],
    'F5': [opt[6] for opt in optimized_F_list],
    'F6': [opt[3] for opt in optimized_F_list],
    'F7': [opt[4] for opt in optimized_F_list],
    'Nc1': [opt[0] for opt in optimized_Nc_list],
    'Nc2': [opt[1] for opt in optimized_Nc_list],
    'Nc3': [opt[2] for opt in optimized_Nc_list],
    'Nc4': [opt[3] for opt in optimized_Nc_list],
    'Nc5': [opt[4] for opt in optimized_Nc_list],
    'Nh1': [opt[0] for opt in optimized_Nh_list],
    'Nh2': [opt[1] for opt in optimized_Nh_list],
    'Nh3': [opt[2] for opt in optimized_Nh_list],
    'Nh4': [opt[3] for opt in optimized_Nh_list],
    'Nh5': [opt[4] for opt in optimized_Nh_list],
    'No1': [opt[0] for opt in optimized_No_list],
    'No2': [opt[1] for opt in optimized_No_list],
    'No3': [opt[2] for opt in optimized_No_list],
    'No4': [opt[3] for opt in optimized_No_list],
    'No5': [opt[4] for opt in optimized_No_list]
})

# Save to CSV
df.to_csv('optimized_results_over_time.csv', index=False)

# Plot results
plt.figure(figsize=(12, 6))
# Plot flow rates (F)
plt.subplot(1, 2, 1)
plt.plot(time_steps, df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']])
plt.legend(['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'])
plt.title('Optimized Flow Rates (F) Over Time')
plt.xlabel('Time Step')
plt.ylabel('Flow Rate')

# Plot atomic balances (Nc, Nh, No)
plt.subplot(1, 2, 2)
plt.plot(time_steps, df[['Nc1', 'Nh1', 'No1', 'Nc5', 'Nh5', 'No5']])
plt.legend(['Nc1', 'Nh1', 'No1', 'Nc5', 'Nh5', 'No5'])
plt.title('Optimized Atomic Balances (Nc, Nh, No) Over Time')
plt.xlabel('Time Step')
plt.ylabel('Atomic Balances')

plt.tight_layout()
plt.show()


