import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from config import paths


# Get path
folder_paths = paths.get_paths()

# Set folder
csv_folder_path = folder_paths["reconstructions_by_parameters"]
files = os.listdir(csv_folder_path)

columns = [
    'Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio',
    'Elongation', 'Flatness', 'Sphericity', 'Compactness',
    'Components_number', 'Point_density', 'Measured_leaf_area'
]

for file in files:
    file_path = os.path.join(csv_folder_path, file)
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        print(file)
        df = pd.read_csv(file_path)
        df_scaled = df.copy()  # Make a copy of the DataFrame
        scaler = StandardScaler()
        for column in columns:
            scaler.fit(df[[column]])
            df_scaled[column] = scaler.transform(df[[column]])  # Transform and assign scaled values
        print(df.to_string())
        print(df_scaled.to_string())
        df_scaled.to_csv(file_path, index=False)
