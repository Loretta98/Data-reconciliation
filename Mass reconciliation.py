# QUESTO CODICE CONTIENE DATA RECONCILIATION IMPIANTO FAT
# 12/02/2024

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

def AbsError(x,F1,F2,F3,F4,F6,F7,F8):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F3r = x[1]
    F4r = x[2]
    F7r = x[3]
    F8r = x[4]

    phi = (F4r+F8r+F7r+F3r-F2r - F1)**2 + (F2r - F2)**2 + (F3r - F3)**2 + (F4r - F4)**2 + (F7r+F8r - F6)**2 + (F7r - F7)**2 + (F8r - F8)**2

    return phi

def WeightAbsError(x,F1,F2,F3,F4,F6,F7,F8,omega1, omega2, omega3, omega4, omega6, omega7, omega8):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F3r = x[1]
    F4r = x[2]
    F7r = x[3]
    F8r = x[4]

    phi = omega1*(F4r+F8r+F7r+F3r-F2r - F1)**2 + omega2(F2r - F2)**2 + omega3(F3r - F3)**2 + omega4(F4r - F4)**2 + omega6(F7r+F8r - F6)**2 + omega7(F7r - F7)**2 + omega8(F8r - F8)**2

    return phi

def RelError(x,F1,F2,F3,F4,F6,F7,F8):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F3r = x[1]
    F4r = x[2]
    F7r = x[3]
    F8r = x[4]

    phi = (F4r+F8r+F7r+F3r-F2r - F1)**2/F1 + (F2r - F2)**2/F2 + (F3r - F3)**2/F3 + (F4r - F4)**2/F4 + (F7r+F8r - F6)**2/F6 + (F7r - F7)**2/F7 + (F8r - F8)**2/F8

    return phi

################################################################################
# Schema di processo : 1 + 2 -> 10 (Reformer)     10 -> 3 + 5 (Separazione H2O)         5 -> 4 + 6 (PSWA)         6 -> 9  (Sintesi)        9 -> 8 + 7 (Purifica)


F1 = 24.02                                  # Biogas feed [Nm3/h]
F2 = 48.71                                  # Acqua feed [L/h]
F3 = 41.92                                  # Condensa post reformer [L/h]
F4 = 10                                     # CO2 da PSWA [kg]
# F5 =
F6 = 20                                     # Feed reattore [kg/h]
F7 = 13.29                                  # Offgas sintesi [kg]
F8 = 6.7                                    # Raw product [L/h] (valore maggiore del set di dati)
# F9 =
# F10 =

#########################
# Conversioni

FirstGuess = np.array([48, 38.4, 8, 4.85, 4.6]) # F2r, F3r, F4r, F7r, F8r

root = minimize(RelError, x0=FirstGuess, args=(F1,F2,F3,F4,F6,F7,F8), bounds= ((0,100),(0,100),(0,100),(0,100),(0,100)))

F2r = root.x[0]
F3r = root.x[1]
F4r = root.x[2]
F7r = root.x[3]
F8r = root.x[4]

F1r  = F4r + F8r + F7r + F3r - F2r
F10r = F4r + F8r + F7r + F3r
F5r  = F4r + F8r + F7r
F6r  = F8r + F7r
F9r  = F8r + F7r

Fr = np.array([F1r , F2r , F3r, F4r, F5r, F6r, F7r, F8r, F9r, F10r])
Eps = np.array([(F1-F1r)/F1r, (F2-F2r)/F2r , (F3-F3r)/F3r, (F4-F4r)/F4r, (F6-F6r)/F6r, (F7-F7r)/F7r, (F8-F8r)/F8r])*100

print(Eps)
# PRODUCTION





################################################################
# Plotting