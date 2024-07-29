# Data reconciliation

Mass Reconciliation is based on simple scipy minimization of the quadratic difference between the values and the mass balances. 

Mass conversion is a file set up (to be finished) to convert all the given raw data to mass unit (kg/h) for the mass reconciliation 

Data_Rec_Lin_lagrangian tries to use the Lagrangian multipliers to find the reconciled variables, after recognizing the redudant ones. 

Outliers_detection_for_n_steps is based on the standard deviation of the data sets to identify outliers and stationary intervals with variable n_steps

Outliers_detection_1 sorts the files by standard deviation, according to the maximum measurement device error

Outliers_detection_2 sorts the files using the KNN method, with the scikit library

Outliers_detection_3 sorts the files using an Isolation Forest method, with the scikit library