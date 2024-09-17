############################################################################
##### RIGOROUS DATA RECONCILIATION FOR GIOVE (UMBRIA, FAT) DEMO PLANT ######
##### Created on 20/06/2024, author = Loretta Salano #######################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

import glob 
import os
import pandas as pd 

#### MASS  DATA RECONCILIATION --> LINEAR ######
#### Variables redundancy classification #######

# Ay x + Az 2u = 0; x = measured variables; u = unmeasured variables 
# DoR = 5 + 3 - 7 = 1 > 0 
# Mass Balances 
# F1 + F2 - F3 = 0
# F3 - F4 - F5 = 0 
# F5 - F6 - F7 = 0 
# x = F1,F2,F4,F6,F7; u = F3,F5 

# Given measured data mean values
y = np.array([24.07, 48.71, 40.11, 8.516215893, 18.95019524, 10.67, 8.977])


def MassReconciliation_Projection(y, Measured_values, V, conversion_factor, Ay,Az, G, Q1,R):
    # Extract the variables and scale Measured_values by the conversion factors
    y = np.array(y)*conversion_factor

    # All variables are mass-based for the reconciliation
    Measured_values = np.array(Measured_values) * conversion_factor
    
    G_V_G_T = G @ V @ G.T
    
    # Step 2: Since G^T V G is a scalar, we can invert it using 1/ instead of np.linalg.inv()
    G_V_G_T_inv = 1.0 / G_V_G_T  # Inversion of the scalar (1x1 value)
    
    # Step 3: Compute the full correction term
    correction = np.dot(V,np.dot(G.T,np.dot(G_V_G_T_inv,np.dot(G,y))))
    
    # Step 4: Compute the reconciled data
    y_hat = y - correction
    R1 = R[:-1,:]
    term1 = np.linalg.inv(R1)
    u_hat = - np.dot(term1, np.dot(Q1.T, np.dot(Ay,y_hat)))
    
    # Compute the residuals (for error analysis or constraints checking)
    Fm = y_hat  # measured reconciled data
    F1r = Fm[0]; F2r = Fm[1]; F4r = Fm[2]; F6r=Fm[3]; F7r=Fm[4]
    Eps = y - y_hat  # Errors
    Fu = u_hat 
    Fr = np.array([F1r , F2r , Fu[0], F4r, Fu[1], F6r, F7r])

    return Fr, Eps
      
