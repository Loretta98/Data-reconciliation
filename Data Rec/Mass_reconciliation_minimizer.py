# QUESTO CODICE CONTIENE DATA RECONCILIATION IMPIANTO
# 02/09/2024
# F1 a F7 (esclusa sezione di sintesi e purifica)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize


########################### Objective functions definition ###########################

def AbsError(x,F1,F2,F4,F6,F7):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F4r = x[1]
    F6r = x[2]
    F7r = x[3]

    phi = (F4r+F6r+F7r-F2r - F1)**2 + (F2r - F2)**2  + (F4r - F4)**2 + (F6r - F6)**2/F6 + (F7r - F7)**2 

    return phi

def WeightAbsError(x,F1,F2,F4,F6,F7,omega1, omega2, omega4, omega6, omega7):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F4r = x[1]
    F6r = x[2]
    F7r = x[3]

    phi = omega1*(F4r+F6r+F7r-F2r - F1)**2 + omega2*(F2r - F2)**2+ omega4*(F4r - F4)**2 + omega6*(F6r - F6)**2 + omega7*(F7r - F7)**2 

    return phi

def RelError(x,F1,F2,F4,F6,F7):

    # x = [F2r, F3r, F4r, F7r, F8r]
    F2r = x[0]
    F4r = x[1]
    F6r = x[2]
    F7r = x[3]

    phi = (F4r+F6r+F7r-F2r - F1)**2/F1 + (F2r - F2)**2/F2  + (F4r - F4)**2/F4 + (F6r - F6)**2/F6 + (F7r - F7)**2/F7 

    return phi

def Lagrangian(x, lambdas, F1, F2, F4, F6, F7):
    # Reconciled values
    F2r, F4r, F6r, F7r = x
    λ1, λ2 = lambdas
    
    # Define the mass balance constraints
    constraint1 = F4r + F6r + F7r - F2r - F1  # Should equal 0 for balance
    constraint2 = F2r + F4r + F6r + F7r       # Another possible constraint, if applicable
    
    # Original objective function (sum of squared relative errors)
    phi = ((F4r + F6r + F7r - F2r - F1)**2 / F1 + 
           (F2r - F2)**2 / F2 + 
           (F4r - F4)**2 / F4 + 
           (F6r - F6)**2 / F6 + 
           (F7r - F7)**2 / F7)
    
    # Lagrangian function: objective + sum(lambda * constraint)
    lagrangian = phi + λ1 * constraint1 + λ2 * constraint2
    
    return lagrangian
################################################################################
# Schema di processo : 1 + 2 -> 3 (Reformer)     3 -> 4 + 5 (Separazione H2O)         5 -> 7 + 6 (PSWA)       
# 24.07, 48.71, 40.11, 8.516215893, 18.9501952 [kg/h] 

def MassReconciliation(y, Measured_values): 
    MW_gas = 0.6*(12+4) +0.4*(12+16*2)  #g/mol
    rho_w = 997/1000    #kg/lt
    CONVERSIONE = 0.044*MW_gas
    print(CONVERSIONE)

    F1 = y[0]*0.044*MW_gas
    F2 = y[1]*rho_w
    F4 = y[2]*rho_w*60
    F6 = y[3]
    F7 = y[4]

    # F1 = 24.02                                  # Biogas feed [Nm3/h]
    # F2 = 48.71                                  # Acqua feed [L/h]
    # #F3 = 
    # F4 = 40.911                                 # Condensa post reformer [L/h]
    # F6 = 8.51                                   # CO2 da PSWA [kg]
    # # F5 =
    # F7 = 18.95                                     # Feed reattore [kg/h]

    #Measured_values = np.array([24.02, 48, 40.911, 8.51, 18.95 ])
    
    ################# Conversion to Mass ##########################################
    #x = np.array([[0,0.6,0.4,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[]]) # Offgas composition
    # H2 CH4 CO CO2 H2O
    
    MW_bg = 0.6*(12+4)+0.4*(44) # g/mol Biogas
    rho_mix = 0.96*(12+4+16) + 0.04*(2+16) # g/mol MeOH+H2O  
    
    #print('Measurement');print(Measured_values)

    for i in range(0,5):
        #F1,F2,F4,F6,F7
        conversion_factor = np.array([ 0.044*MW_gas ,rho_w, rho_w*60, 1,1]) # Nm3/h, lt/h, lt/min, kg/h, kg/h
        Measured_values[i] = Measured_values[i]*conversion_factor[i]
    #print('Mass base');print(Measured_values)
    
    FirstGuess = Measured_values[1:] # F2r, F4r, F6r, F7r
    #print('First Guess');print(FirstGuess)
    #root = minimize(RelErr, x0=FirstGuess, args=(F1,F2,F4,F6,F7), bounds= ((0,100),(0,100),(0,100),(0,100)))
    root = minimize(Lagrangian, x0=FirstGuess, args=(F1, F2, F4, F6, F7), bounds=[(0, None)] * len(FirstGuess))
    F2r = root.x[0]
    F4r = root.x[1]
    F6r = root.x[2]
    F7r = root.x[3]

    F1r  = F4r+F6r+F7r-F2r
    F5r = F6r + F7r
    F3r = F4r + F6r + F7r
    #F3r = F1r + F2r

    Fr = np.array([F1r , F2r , F3r, F4r, F5r, F6r, F7r])
    #print('Reconciled Values: '); print(Fr)
    Eps = np.array([(F1-F1r)/F1r, (F2-F2r)/F2r , (F4-F4r)/F4r, (F6-F6r)/F6r, (F7-F7r)/F7r])*100
    #print(Eps)
    return Fr, Eps



