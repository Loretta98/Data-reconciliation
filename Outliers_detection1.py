import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024'
st_dev_path = os.path.join(path, 'st_dev')

# Create the 'st_dev' directory if it does not exist
os.makedirs(st_dev_path, exist_ok=True)

# List CSV files 
files = glob.glob(path + '/*.csv')

for filename in files:
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the third column
    third_column = df.iloc[:, 2]
    
    # Calculate standard deviation for every 50 rows
    st_devs = [np.std(third_column[i:i+50]) for i in range(0, len(third_column), 50)]
    
    # Store the values in a new array
    file_basename = os.path.basename(filename).split('.')[0]
    st_dev_array = os.path.join(st_dev_path, f"st_dev_{file_basename}.csv")
    
    # Plot the standard deviations
    plt.figure()
    plt.plot(st_devs, marker='o')
    plt.title(f'Standard Deviation for {file_basename}')
    plt.xlabel('Index (50 rows per point)')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(st_dev_path, f"{file_basename}_std_dev_plot.png"))
    plt.show()
    
    # Save the standard deviations to a new CSV file
    pd.DataFrame(st_devs, columns=['Standard Deviation']).to_csv(st_dev_array, index=False)
