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

# def QR_Decomposition(A2):
#     n, m = A2.shape # get the shape of A

#     Q = np.zeros((n, n)) # initialize matrix Q
#     u = np.zeros((n, n)) # initialize matrix u

#     u[:, 0] = A2[:, 0]
#     Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

#     for i in range(1, m):

#         u[:, i] = A2[:, i]
#         for j in range(i):
#             u[:, i] -= (A2[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

#         Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

#     R = np.zeros((n, m))
#     for i in range(n):
#         for j in range(i, m):
#             R[i, j] = A2[:, j] @ Q[:, i]

#     return Q, R

# def diag_sign(A2):
#     "Compute the signs of the diagonal of matrix A"

#     D = np.diag(np.sign(np.diag(A2)))

#     return D

# def adjust_sign(Q, R):
#     """
#     Adjust the signs of the columns in Q and rows in R to
#     impose positive diagonal of Q
#     """

#     D = diag_sign(Q)

#     Q[:, :] = Q @ D
#     R[:, :] = D @ R

#     return Q, R

# Q, R = adjust_sign(*QR_Decomposition(A2))

# print(A2);print(Q),print(R)