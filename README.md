# Data reconciliation

Mass Reconciliation is based on simple scipy minimization of the quadratic difference between the values and the mass balances. 

Mass conversion is a file set up (to be finished) to convert all the given raw data to mass unit (kg/h) for the mass reconciliation 

Data_Rec_Lin tries to use the Lagrangian multipliers to find the reconciled variables, after recognizing the redudant ones. 

Outliers_detection1 is based on the standard deviation of the data sets to identify outliers and stationary intervals
