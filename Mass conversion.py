import numpy as np 
import glob 
import pandas as pd 
import csv 
import os

################ Data acquisition from excel #################################
df1 = pd.read_excel("C:\Users\lsalano\OneDrive - Politecnico di Milano\Desktop\FAT\Riconciliazione dati\MicroGC\Dati ordinati stazionario 2024.xlsx", sheet_name="Syngas")


x = np.array() # Offgas composition
# H2 O2 N2 CH4 CO CO2 CH3OH
################# Conversion to Mass ##########################################
rho_w = 997/1000    #kg/lt
MW_bg = 0.6*(12+4)+0.4*(44) # g/mol Biogas
rho_mix = 0.96*(12+4+16) + 0.04*(2+16) # g/mol MeOH+H2O  
# property based on composition 
MW_offgas = x[0]*(2) + x[4]*(12+16) +x[3]*(12+4) + x[5]*(12+16*2) + x[6]*(12+4+16)
conversion_factor = np.array([rho_w*60, 1, 0.044*MW_offgas, rho_mix*60, 1,0.044*MW_bg, rho_w]) # lt/min, kg/h, Nm3/h, lt/min, kg/h, Nm3/h, lt/h

################## Data acquisition in Python as arrays #########################
path = "C:\Users\lsalano\OneDrive - Politecnico di Milano\Desktop\FAT\Riconciliazione dati\PLC\Maggio 2024"
files = glob.glob(path + '/*.csv') 

for filename in (files): 
    df = pd.read_csv('')
    array1 = df["TimeString"].values
    array2 = df["VarValue"].values
        
