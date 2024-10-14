import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set font to Segoe UI if available
mpl.rcParams['font.family'] = 'Segoe UI'

# File path
file_path = r'C:\DataRec\To_reconcile\fortyfive\clean_data_lagrangian_abs\results.csv'

# Load the data with the correct delimiter
data = pd.read_csv(file_path, delimiter=';')

# List of variables to plot, including the additional ones
variables = ['F1', 'F2', 'F4', 'F6', 'F7']
additional_variables = ['Reconciled F3', 'Reconciled F5']

# Colors for each variable to ensure distinct colors across plots
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'magenta']

# Loop over each variable to create individual plots
for i, var in enumerate(variables):
    # Special case for variable F2 to create two subplots
    if var == 'F2':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract reconciled and original data
        original_data = data[var].dropna()
        reconciled_data = data[f'Reconciled {var}'].dropna()

        # Plot histogram and normal distribution for F2 on the left subplot
        axes[0].hist(original_data, bins=30, density=True, alpha=0.6, color=colors[i], label=f'{var}', edgecolor='black')
        mean, std = np.mean(original_data), np.std(original_data)
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
        axes[0].plot(x, y, color=colors[i], linestyle='-', label=f'Normal Distribution - {var}')

        # Titles, labels, legend, and grid for the first subplot (F2)
        #axes[0].set_title(f'Histogram and Normal Distribution: {var}')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True)

        # Plot histogram and normal distribution for Reconciled F2 on the right subplot
        axes[1].hist(reconciled_data, bins=30, density=True, alpha=0.6, color=colors[i], label=f'Reconciled {var}', edgecolor='black', hatch='/')
        mean_reconciled, std_reconciled = np.mean(reconciled_data), np.std(reconciled_data)
        x_reconciled = np.linspace(mean_reconciled - 3*std_reconciled, mean_reconciled + 3*std_reconciled, 100)
        y_reconciled = (1 / (std_reconciled * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_reconciled - mean_reconciled) / std_reconciled)**2)
        axes[1].plot(x_reconciled, y_reconciled, color=colors[i], linestyle='--', label=f'Normal Distribution - Reconciled {var}')

        # Titles, labels, legend, and grid for the second subplot (Reconciled F2)
        axes[1].set_title(f'Histogram and Normal Distribution: Reconciled {var}')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True)

        # Adjust layout and show the plot
        plt.tight_layout()
    else:
        # Create a new figure for each variable (excluding F2)
        plt.figure(figsize=(8, 6))
        
        # Extract reconciled and original data
        original_data = data[var].dropna()
        reconciled_data = data[f'Reconciled {var}'].dropna()

        # Plot histograms with matching colors
        plt.hist(original_data, bins=30, density=True, alpha=0.6, color=colors[i], label=f'{var}', edgecolor='black')
        plt.hist(reconciled_data, bins=30, density=True, alpha=0.4, color=colors[i], label=f'Reconciled {var}', edgecolor='black', hatch='/')

        # Fit and plot normal distributions with matching colors
        for dataset, linestyle, label in zip([original_data, reconciled_data], ['-', '--'], [f'{var}', f'Reconciled {var}']):
            mean, std = np.mean(dataset), np.std(dataset)
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
            plt.plot(x, y, color=colors[i], linestyle=linestyle, label=f'Normal Distribution - {label}')

        # Labels, title, legend, and grid
        #plt.title(f'Histogram and Normal Distribution: {var} and Reconciled {var}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.tight_layout()


# Plot histograms for additional variables (Reconciled F3 and Reconciled F5) with normal distribution
for j, var in enumerate(additional_variables, len(variables)):
    plt.figure(figsize=(8, 6))
    
    # Extract data for the additional variables
    data_var = data[var].dropna()

    # Plot histogram and normal distribution for the additional variables
    plt.hist(data_var, bins=30, density=True, alpha=0.6, color=colors[j], label=f'{var}', edgecolor='black')

    mean, std = np.mean(data_var), np.std(data_var)
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    plt.plot(x, y, color=colors[j], linestyle='-', label=f'Normal Distribution - {var}')

    # Labels, title, legend, and grid
    #plt.title(f'Histogram and Normal Distribution: {var}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()

plt.show()
