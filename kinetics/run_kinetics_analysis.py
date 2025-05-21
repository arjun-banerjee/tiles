import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from tqdm import tqdm  
import numpy as np
import sys
import kinetics_utils as ku

#1: Read Dataset
file_path = sys.argv[1]
write_path = sys.argv[2]
df = ku.read_dataset(file_path)

#2: Normalize Dataset
df = ku.normalized_dataset(df)

#3: Fit Model and save results
col_groups = [df.iloc[:, i:i+2] for i in range(1, df.shape[1], 2)]
for i,group in enumerate(col_groups):
    best_loss, best_popt = ku.fit_model(group["Time"], group["Fluorescence"])
    #write results to file
    with open(f'{write_path}/results.csv', "a") as f:
        np.savetxt(f, np.insert(best_popt, 0, best_loss), delimiter=",", fmt="%.5f")
    ku.plot_results(group["Time"], group["Fluorescence"], best_popt, f'{write_path}/plot{i}.png')

