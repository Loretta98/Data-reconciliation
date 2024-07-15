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

# Balance Matrix of measured variables 
A1 = np.array([[1,1,0,0,0,0,0],
              [0,0,-1,0,0,0,0],
              [0,0,0,-1,-1,0,0],
              [0,0,0,0,1,0,0],
              [0,0,0,0,0,-1,-1]])

# Balance Matrix of unmeasured variables 
A2 = np.array([[-1,0,0],
              [0,-1,0],
              [0,1,0],
              [0,0,-1],
              [0,0,1]])

# rank of the matrix msut be = number of streams 
M = np.array([[1,1,-1,0,0,0,0,0,0,0],
              [0,0,1,-1,-1,0,0,0,0,0],
              [0,0,0,0,1,-1,-1,0,0,0],
              [0,0,0,0,0,0,1,-1,0,0],
              [0,0,0,0,0,0,0,1,-1,-1],
              [1,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,1]
              ])

rank = np.linalg.matrix_rank(M)
print("\nrank of matrix M: "), print(rank)
# Projection matrix P so that P = Q2.T, Q = [Q1,Q2], P(A1x+A2u) = PA1x = Gx = 0, A2 = QR using the orthogonalization (Q orthogonal matrix and R upper trangular matrix) 

# Perform QR decomposition of A2
Q, R = qr(A2, mode="full")

min_dim = min(A2.shape)
R1 = R[:min_dim, :min_dim]

# # Print results of QR decomposition
# print("A2:"); print(A2); print("\nQ:"); print(Q); print("\nR:"); print(R)

# # Verify the QR decomposition
# check = np.dot(Q, R)
# print("\nCheck (Q * R):"); print(check)

# Slit R into R1 and 0 

# Split Q into Q1 and Q2
Q1 = Q[:, :min_dim]  # First 3 columns of Q (same number as columns in A2)
Q2 = Q[:, min_dim:]  # Remaining columns of Q

#print(R)
#print(R1)

# Form the projection matrix P
P = Q2.T
#P = np.dot(Q2, Q2.T)
# print("\nProjection matrix P:"); print(P)

PA2 = np.dot(P,A2)
print("\nPA2:"); print(PA2)

# Calculate G = PA1
G = np.dot(P, A1)
print("\nMatrix G (P * A1):"); print(G)

# Define the vector x corresponding to the measured variables F1, F2, F4, F6, F7, F9, F10
x1 = np.array([1, 2, 4, 6, 7, 9, 10])

# Compute Gx to see the balances
Gx = np.dot(G, x1)
# print("\nGx (Balances obtained from Gx):"); print(Gx)

# Mapping of x to F variables
F_vars = ['F1', 'F2', 'F4', 'F6', 'F7', 'F9', 'F10']

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

# Identify redundant and non redundant variables
redundant_vars = [var for var, coef in combined_balance_terms.items() if coef != 0]
remaining_vars = [var for var in F_vars if var not in redundant_vars]

print("\nRedundant Variables:")
print(redundant_vars)
print("\nNon-Redundant Variables:")
print(remaining_vars)

######### Resolution of the Data Reconciliation problem with Lagrange multipliers #############

# First the suproblem with the measured variables is resolved 
# min (y-x)^T phi^-1 (y-x) s.t. Gx = 0
# xr = y - phi G^T (GphiG)^-1 Gy 
# ur = -r1^1Q1^TA1 xr
# x = F1,F2,F4,F6,F7,F9,10; u = F3,F5,F8 ; x = y- eps

# Measured data mean value for one of the process steady states 
y = np.array([24.07,	48.71,	40.11,	8.516215893,	18.95019524,	10.67,	8.977])
phi = np.zeros([np.size(y),np.size(y)]) # variance matrix
sigma = np.array([0.004269825,	0.01871219,	0.000214119,	0.112119912,	0.627503283,	2.36339497,	0.000145749
]) # measurements variance

for i in range(0,np.size(y)):
    phi[i,i] = sigma[i] 

# # First guess for values to be reconciled 
x0 = np.array([24.07,	48.71,	40.11,	8.516215893,	18.95019524,	10.67,	8.977])
# lambda0 = np.zeros(G.shape[0])
# initial_guess = np.concatenate([x0, lambda0])

# def lagrangian(x_lambda):
#     # Split the input vector into x and lambda
#     x = x_lambda[:len(y)]
#     lambd = x_lambda[len(y):]
    
#     # Compute the Lagrangian
#     L = (y - x).T @ np.linalg.inv(phi) @ (y - x) + lambd.T @ G @ x
    
#     # Compute the gradients
#     grad_x = -2 * np.linalg.inv(phi) @ (y - x) + G.T @ lambd
#     grad_lambda = G @ x
    
#     # Concatenate gradients
#     return np.concatenate([grad_x, grad_lambda])
# # Solve the system of equations
# solution = fsolve(lagrangian, initial_guess)

# # Extract x and lambda from the solution
# x_opt = solution[:len(y)]
# lambda_opt = solution[len(y):]

# # Print the results
# print("Optimal x:", x_opt)
# print("Optimal lambda:", lambda_opt)

def f(x): 
    return np.dot((y-x).T,np.dot(np.linalg.inv(phi),(y-x)))

cons = ({'type':'eq', 
         'fun' : lambda x: np.dot(G,x) }, # Equality constraint
         {'type': 'ineq', 'fun': lambda x: x}  )          # Inequality constraint for x >= 0)

res = minimize(f,x0,constraints=cons, method='SLSQP')
print(res.x)
# Check if the optimization was successful
if not res.success:
    print("Optimization failed:", res.message)

# Ensure x_opt is in the positive domain
x_opt = np.maximum(res.x, 0)

# Print the results
print("Optimal x (adjusted for positivity):", x_opt)

# # Compute G^T
# G_T = G.T

# # Compute the product G * Phi * G^T
# G_Phi_G_T = np.dot(np.dot(G, phi), G_T)

# # Compute the inverse of the product
# G_Phi_G_T_inv = np.linalg.inv(G_Phi_G_T)

# # Compute the product of G * y
# G_y = np.dot(G, y)

# # Compute the product of Phi * G^T * (G * Phi * G^T)^-1 * G * y
# Phi_G_T_G_Phi_G_T_inv_G_y = np.dot(np.dot(np.dot(phi, G_T), G_Phi_G_T_inv), G_y)

# # Compute the final result: y - Phi * G^T * (G * Phi * G^T)^-1 * G * y
# # reconciled measured variables 
# x_r = y - Phi_G_T_G_Phi_G_T_inv_G_y 
# eps = y-x_r

# R1_inv = np.linalg.inv(R1)

# # Compute the transpose of Q1
# Q1_T = Q1.T

# # Compute the product Q1^T * A1 * x
# Q1_T_A1_x = np.dot(np.dot(Q1_T, A1), x_r)

# # Compute the product -R1^(-1) * Q1^T * A1 * x
# u_hat = -np.dot(R1_inv, Q1_T_A1_x)

# # reconciled unmeasured variables 
# print(u_hat)
# print(x_r)
# print(eps)
