# This code allows to generate the Missing measurement in terms of composition with a standard deviation of = 0.2 (based on the available data)

import pandas as pd
import numpy as np


# Load the previously generated F5.csv to match the Injection Time column
f5_path = r'C:\DataRec\To_reconcile\GC_filtered\F5.csv'  # Assuming F5.csv is in the same path where we saved the others
df_f5 = pd.read_csv(f5_path)

# Create a new DataFrame for F1 with the same Injection Time column
df_f1 = pd.DataFrame()

df_f1['Injection Time'] = df_f5['Injection Time']

# List of compounds
compounds = ['H2 (A) [%]', 'O2 (A) [%Vol]', 'N2 (A) [%Vol]', 'CH4 (A) [%Vol]', 
             'CO (A) [%]', 'CO2 (B) [%Vol]', 'CH3OH (B) [%]', 'H2O [%Vol]','Total (%)']

# Initialize all compounds with zero values
for compound in compounds:
    df_f1[compound] = 0

# Generate normal distributions for CH4 and CO2
np.random.seed(0)  # Set seed for reproducibility
df_f1['CO2 (A) [%Vol]'] = np.random.normal(loc=100, scale=0.08, size=len(df_f1))


# Save the F1.csv file
f1_path = 'C:\DataRec\To_reconcile\GC_filtered\F6.csv'
df_f1.to_csv(f1_path, index=False)

f1_path
