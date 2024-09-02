############################################################################
##### RIGOROUS DATA RECONCILIATION FOR GIOVE (UMBRIA, FAT) DEMO PLANT ######
##### Created on 20/06/2024, author = Loretta Salano #######################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.linalg import qr

#### MASS  DATA RECONCILIATION --> LINEAR ######
#### Variables redundancy classification #######

# A1x + Ax2u = 0; x = measured variables; u = unmeasured variables 
# DoR = 7 + 5 - 10 = 2 > 0 
# Mass Balances 
# F1 + F2 - F3 = 0
# F3 - F4 - F5 = 0 
# F5 - F6 - F7 = 0 
# F7 - F8 =  0
# F8 -F9 - F10 = 0 
# x = F1,F2,F4,F6,F7,F9,F10; u = F3,F5,F8 

import numpy as np
from scipy.optimize import minimize

# Given measured data mean values
y = np.array([24.07, 48.71, 40.11, 8.516215893, 18.95019524, 10.67, 8.977])

# Balance Matrix of measured variables 
A1 = np.array([[1, 1, 0, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0],
               [0, 0, 0, -1, -1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, -1, -1]])

# Balance Matrix of unmeasured variables 
A2 = np.array([[-1, 0, 0],
               [0, -1, 0],
               [0, 1, 0],
               [0, 0, -1],
               [0, 0, 1]])

# Objective function to minimize the weighted sum of squared errors
def objective(x):
    return np.sum((x - y) ** 2)

# Constraints (Ax = 0)
constraints = [{'type': 'eq', 'fun': lambda x: np.dot(A1, x)}]

# Initial guess for the measured variables (can start with the measured values)
x0 = y.copy()

# Perform the minimization
result = minimize(objective, x0, constraints=constraints, method='SLSQP')

# Reconciled measured variables
x_reconciled = result.x

print("Reconciled measured variables:", x_reconciled)

# Calculate the unmeasured variables using the pseudo-inverse
u_estimated = np.dot(np.linalg.pinv(A2), -np.dot(A1, x_reconciled))

print("Estimated unmeasured variables:", u_estimated)
