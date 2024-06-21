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
# x = F1,F2,F4,F6,F7,F9,10; u = F3,F5,F8 

x = np.array([1,2,4,6,7,9,10])

A1 = np.array([[1,1,0,0,0,0,0],
              [0,0,-1,0,0,0,0],
              [0,0,0,-1,-1,0,0],
              [0,0,0,0,1,0,0],
              [0,0,0,0,0,-1,-1]])
A2 = np.array([[-1,0,0],
              [0,-1,0],
              [0,1,0],
              [0,0,-1],
              [0,0,1]])


# Projection matrix P so that P = Q2.T, Q = [Q1,Q2], P(A1x+A2u) = PA1x = Gx = 0, A2 = QR using the orthogonalization (Q orthogonal matrix and R upper trangular matrix) 

# Perform QR decomposition of A2
Q, R = qr(A2, mode="full")

# Print results of QR decomposition
print("A2:")
print(A2)
print("\nQ:")
print(Q)
print("\nR:")
print(R)

# Verify the QR decomposition
check = np.dot(Q, R)
print("\nCheck (Q * R):")
print(check)

# Split Q into Q1 and Q2
Q1 = Q[:, :A2.shape[1]]  # First 3 columns of Q (same number as columns in A2)
Q2 = Q[:, A2.shape[1]:]  # Remaining columns of Q

# Form the projection matrix P
P = Q2.T
#P = np.dot(Q2, Q2.T)
print("\nProjection matrix P:")
print(P)

PA2 = np.dot(P,A2)
print("\nPA2:")
print(PA2)


# Calculate G = PA1
G = np.dot(P, A1)
print("\nMatrix G (P * A1):")
print(G)

# Define the vector x corresponding to the measured variables F1, F2, F4, F6, F7, F9, F10
x = np.array([1, 2, 4, 6, 7, 9, 10])

# Compute Gx to see the balances
Gx = np.dot(G, x)
print("\nGx (Balances obtained from Gx):")
print(Gx)

# Mapping of x to F variables
F_vars = ['F1', 'F2', 'F4', 'F6', 'F7', 'F9', 'F10']

# Interpretation of Gx
print("\nInterpreting the balances:")

# Interpretation of Gx
balances = []
print("\nInterpreting the balances:")
for i, balance in enumerate(Gx):
    balance_terms = [f"{G[i, j]}*{F_vars[j]}" for j in range(len(F_vars)) if G[i, j] != 0]
    balance_eq = " + ".join(balance_terms)
    print(f"Balance {i+1}: {balance_eq} = 0")
    balances.append(balance_terms)

# Summing the balances
combined_balance_terms = {}
for balance in balances:
    for term in balance:
        coef, var = term.split('*')
        coef = float(coef)
        if var in combined_balance_terms:
            combined_balance_terms[var] += coef
        else:
            combined_balance_terms[var] = coef

combined_balance_eq = " + ".join(f"{coef}*{var}" for var, coef in combined_balance_terms.items() if coef != 0)
print("\nCombined Balance Equation:")
print(f"{combined_balance_eq} = 0")

# Identify redundant and remaining variables
redundant_vars = [var for var, coef in combined_balance_terms.items() if coef != 0]
remaining_vars = [var for var in F_vars if var not in redundant_vars]

print("\nRedundant Variables:")
print(redundant_vars)
print("\nNon-Redundant Variables:")
print(remaining_vars)


# for i, balance in enumerate(Gx):
#     balance_eq = " + ".join(f"{G[i, j]}*{F_vars[j]}" for j in range(len(F_vars)) if G[i, j] != 0)
#     print(f"Balance {i+1}: {balance_eq} = 0")