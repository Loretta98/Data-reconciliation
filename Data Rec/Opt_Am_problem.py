import numpy as np 
import pandas as pd 


# Atomic balance 
# N1C = N3C 
# N3C = N4C + N5C 
# N5C = N6C + N7C 
# MWmix = sum(xi*MWi)
# N5C = (xCO2+xCH4+xCO)*F5/MWmix

# Define molar masses (g/mol) for calculating N5C, N5H, N5O
molar_masses = {'CH4': 16.04, 'CO': 28.01, 'CO2': 44.01, 'CH3OH': 32.04, 'H2': 2.016, 'O2': 32.00, 'H2O': 18.01528}

# Function to calculate molar weight of the mixture in F5
def MW_mixture_F5(x_CO2, x_CH4, x_CO, x_H2):
    return x_CO2 * molar_masses['CO2'] + x_CH4 * molar_masses['CH4'] + x_CO * molar_masses['CO'] + x_H2*molar_masses['H2']

def replace_nan_with_zero(scaled_array):
    """
    Replace NaN values in the scaled array with zero.
    
    Parameters:
    - scaled_array (numpy array): The array with possible NaN values.
    
    Returns:
    - numpy array: The array with NaN values replaced by zeros.
    """
    return np.nan_to_num(scaled_array, nan=0.0)

def add_row_col(matrix, n):
    # Get the shape of the original matrix
    rows, cols = matrix.shape
    
    # Create a new matrix with an extra row and column filled with zeros
    new_matrix = np.zeros((rows + 1, cols + 1))
    
    # Copy the original matrix into the top-left part of the new matrix
    new_matrix[:rows, :cols] = matrix
    
    # Set the bottom-right value to n
    new_matrix[rows, cols] = n
    
    return new_matrix


# Objective function with Nc, Nh, and No treated as variables
def objective(x, V, Wc, Wh, Wo, x_CO2_F5, x_CH4_F5, x_CO_F5,x_H2_F5,  F_m, Nc_m, Nh_m, No_m):
 
    F   = x[:5]  # Measured flow rates: F1, F2, F4, F6, F7 [kg/h]
    F3  = x[5]
    F5  = x[6]

    Nc_vars = x[7:12]  # Includes Nc for F1, F2, F4, F6, F7
    Nh_vars = x[12:17]
    No_vars = x[17:22]
    
    # Calculate N5C, N5H, N5O using F5 and composition values
    MW_mix = MW_mixture_F5(x_CO2_F5, x_CH4_F5, x_CO_F5, x_H2_F5)
    N5C = F5 * (x_CH4_F5 * 1 + x_CO2_F5 * 1 + x_CO_F5 * 1)*1000 / MW_mix
    N5H = F5 * (x_CH4_F5 * 4 + x_H2_F5*2)*1000 / MW_mix
    N5O = F5 * (x_CO2_F5 * 2 + x_CO_F5 * 1)*1000 / MW_mix
    
    # Build full arrays with N5C, N5H, N5O
    Nc_i = np.concatenate([Nc_vars, [N5C]])
    Nh_i = np.concatenate([Nh_vars, [N5H]])
    No_i = np.concatenate([No_vars, [N5O]])

    # Define constraints using the current values of N5C, N5H, N5O
    constraints = create_constraints_with_N5C_N5H_N5O(N5C, N5H, N5O)

    # Objective function
    phi = (F_m - F).T @ V @ (F_m - F)
    phi += (Nc_m - Nc_i).T @ Wc @ (Nc_m - Nc_i)
    phi += (Nh_m - Nh_i).T @ Wh @ (Nh_m - Nh_i)
    phi += (No_m - No_i).T @ Wo @ (No_m - No_i)
    
    return phi, constraints


def create_constraints_with_N5C_N5H_N5O(N5C, N5H, N5O):
    constraints = [
        # Mass balance constraints
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] - x[5]},   # F1 + F2 - F3 = 0
        {'type': 'eq', 'fun': lambda x: x[5] - x[2] - x[6]},   # F3 - F4 - F5 = 0
        {'type': 'eq', 'fun': lambda x: x[6] - x[3] - x[4]},   # F5 - F6 - F7 = 0
        
        # Atomic balance constraints (Carbon)
        {'type': 'eq', 'fun': lambda x: x[7] +x[8] - (x[10] + N5C)},  # Nc1 + Nc2 = Nc4 + N5C
        {'type': 'eq', 'fun': lambda x: N5C - (x[11] + x[12])}, # Nc5 = Nc6 + Nc7

        # Atomic balance constraints (Hydrogen)
        {'type': 'eq', 'fun': lambda x: x[12]+x[13] - (x[15] + N5H)},  # Nh1 + Nh2 = Nh4 + Nh5
        {'type': 'eq', 'fun': lambda x: N5H - (x[16] + x[17])},  # Nh5 = Nh6 + Nh7

        # Atomic balance constraints (Oxygen)
        {'type': 'eq', 'fun': lambda x: x[17]+x[18] - (x[20] + N5O)},  # No1 +No2 = No4 + No5
        {'type': 'eq', 'fun': lambda x: N5O - (x[21] + x[-1])},  # No5 = No6 + No7

        {'type': 'ineq', 'fun': lambda x: x}  # Ensure all variables are positive

        # Normalization Equation 
    ]
    return constraints
