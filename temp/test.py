import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import paths
import os


folder_paths = paths.get_paths()
folder_path = folder_paths["data"]
file_path = os.path.join(folder_path, '003final.csv')
df = pd.read_csv(file_path)



# Sample data
data = {
    'p1': [0.5, 0.6, 0.2],
    'p2': [0.5, 0.4, 0.7]
}

# Create parameter values
parameter_values = [1, 2, 3]

# Create subplots
fig, axs = plt.subplots(len(data), 1, figsize=(6, 6), sharex=True)

for i, (key, values) in enumerate(data.items()):
    X, Y = np.meshgrid(parameter_values, [0, 1])
    Z = np.array([values])
    axs[i].pcolormesh(X, Y, Z)
    axs[i].set_yticks([0.5, 1.5])
    axs[i].set_yticklabels([key])
    axs[i].set_xticks(parameter_values)
    axs[i].set_xlabel('Parameter value')
    axs[i].set_ylabel('Parameter name')
    axs[i].set_title(f'{key}')

fig.colorbar(axs[0].pcolormesh(X, Y, Z), ax=axs, label='R2')
plt.tight_layout()
plt.show()
