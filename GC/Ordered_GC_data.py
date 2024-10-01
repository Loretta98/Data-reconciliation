import pandas as pd

# Use the raw string to handle backslashes correctly
data = r'C:\DataRec\GC\Partial CSV\GC_Maggio_2024.csv'  # Replace 'your_filename.csv' with the actual CSV filename

# Create a DataFrame
df = pd.read_csv(data)

# Convert 'Injection Time' to datetime
df['Injection Time'] = pd.to_datetime(df['Injection Time'], format='%d/%m/%Y %H:%M')

# Extract the base stream names (removing trailing numbers) for easier filtering
df['Base Analysis'] = df['Analysis'].str.extract(r'(^[\w\s]+)_\d+$')[0]

# Define the mapping for base Analysis to stream names
analysis_to_stream = {
    'Co2 separata': 'F6',
    'Syngas': 'F5',
    'post lavaggio': 'F7'
}

# Replace Analysis values with stream names using the base analysis name
df['Stream'] = df['Base Analysis'].replace(analysis_to_stream)

# Select relevant columns
compounds = ['Injection Time', 'H2 (A) [%]', 'O2 (A) [%Vol]', 'N2 (A) [%Vol]', 'CH4 (A) [%Vol]', 'CO (A) [%]', 'CO2 (B) [%Vol]', 'CH3OH (B) [%]']

# Iterate over each unique stream to create separate CSV files
for base_name, Fi_name in analysis_to_stream.items():
    # Filter the data for the current base name
    df_filtered = df[df['Base Analysis'] == base_name][['Injection Time'] + compounds[1:]]  # Keep Injection Time and the relevant compounds

    # Save the filtered data to a CSV file named after the Fi value
    output_path = f'C:\\DataRec\GC\Partial CSV\\{Fi_name}.csv'  # Change to your desired output directory
    df_filtered.to_csv(output_path, index=False)
    
    print(f"CSV file for {Fi_name} saved to: {output_path}")
