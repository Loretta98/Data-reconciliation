import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'C:/Users/lsalano/OneDrive - Politecnico di Milano/Desktop/FAT/Riconciliazione dati/PLC/Maggio 2024/31 Maggio 2024'
outliers_st_dev_path = os.path.join(path, 'outliers_st_dev')

# Create the 'outliers_st_dev' directory if it does not exist
os.makedirs(outliers_st_dev_path, exist_ok=True)
# List CSV files 
files = glob.glob(path + '/*.csv')

# Range of n_steps to test
n_steps_range = range(10, 121,10)

for filename in files:
    # Read CSV file
    df = pd.read_csv(open(filename, 'rb'))
    
    # Extract the third column
    third_column = df.iloc[:, 2]
    
    # Store the values for plotting
    all_st_devs = {}
    
    for n_steps in n_steps_range:
        # Calculate standard deviation for every n_steps rows
        st_devs = [np.std(third_column[i:i+n_steps]) for i in range(0, len(third_column), n_steps)]
        all_st_devs[n_steps] = st_devs
    
    # Create a DataFrame from the dictionary
    max_length = max(len(st_devs) for st_devs in all_st_devs.values())
    for key in all_st_devs:
        all_st_devs[key] += [np.nan] * (max_length - len(all_st_devs[key]))  # Fill shorter lists with NaN
    st_devs_df = pd.DataFrame(all_st_devs)
    
    # Save the DataFrame to a CSV file
    file_basename = os.path.basename(filename).split('.')[0]
    st_dev_csv_path = os.path.join(outliers_st_dev_path, f"outliers_st_dev_{file_basename}.csv")
    st_devs_df.to_csv(st_dev_csv_path, index=False)
    # Plot the standard deviations for each n_steps
    
    # plt.figure()
    # for n_steps, st_devs in all_st_devs.items():
    #     plt.plot(st_devs, marker='o', label=f'n_steps={n_steps}')
    # plt.title(f'Standard Deviations for {os.path.basename(filename)}')
    # plt.xlabel('Index')
    # plt.ylabel('Standard Deviation')
    # plt.legend()
    # plt.grid(True)
    
    # # Save the plot
    # file_basename = os.path.basename(filename).split('.')[0]
    # plot_path = os.path.join(path, f"{file_basename}_std_dev_plots.png")
    # plt.savefig(plot_path)
    # plt.show()
