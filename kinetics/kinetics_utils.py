import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from tqdm import tqdm  
import numpy as np


def read_dataset(file_path):
    """
    Reads a dataset from a csv file.
    """
    return pd.read_csv(file_path)


def normalized_dataset(data, normalization_type = "max"):
    """
    Normalizes a dataset of the form:
    Col | Time | Dimer #1 | Tile #1 | Dimer #2 | Tile #2 | ...
    
    Args:
    data: The dataset to normalize.
    normalization_type: The type of normalization to perform. Can be "max" or "mean" or "standard".
    
    Returns:
    Col | Time | Dimer #1 | Tile #1 | Dimer #2 | Tile #2 | ...

    Where the return df is in divided by the normalization factor
    """
    #initliaze the dataset
    returning_dataset = pd.DataFrame()
    returning_dataset["Time"] = data["Time"]
    #enumerate all groups
    col_groups = [data.iloc[:, i:i+2] for i in range(1, data.shape[1], 2)]
    #normalize each group
    for group in col_groups:
        if normalization_type == "max":
            returning_dataset[group.columns[0]] = group[group.columns[0]] / group[group.columns[0]].max()
            returning_dataset[group.columns[1]] = group[group.columns[1]] / group[group.columns[0]].max()
        elif normalization_type == "mean":
            returning_dataset[group.columns[0]] = group[group.columns[0]] / group[group.columns[0]].mean()
            returning_dataset[group.columns[1]] = group[group.columns[1]] / group[group.columns[0]].mean()
        elif normalization_type == "standard":
            returning_dataset[group.columns[0]] = group[group.columns[0]] / group[group.columns[0]]
            returning_dataset[group.columns[1]] = group[group.columns[1]] / group[group.columns[0]]
    
    return returning_dataset


def dPdt(t, z, khyb, k1, k2):
    """
    A system of ODEs for the fluorescence of a H3 tile.
    
    Args:
    t: The time array.
    z: The initial conditions.
    khyb: The hybridization rate.
    k1: The binding rate.
    k2: The dissociation rate.
    
    Returns:
    The time derivative of the system.
    """
    x, y = z  
    dxdt = - (k1 + khyb) * x + k2 * y
    dydt = k1 * x - k2 * y
    return [dxdt, dydt]

# Wrapper for fluorescence model
def fluorescence_H3_dynamic(t_array, fgood, khyb, k1, k2):
    """
    A system of ODEs for the fluorescence of a H3 tile.
    
    Args:
    t_array: The time array.
    fgood: The fraction of good tiles.
    khyb: The hybridization rate.
    k1: The binding rate.
    k2: The dissociation rate.
    
    Returns:
    The fluorescence of the system.
    """
    # Initial conditions
    x0 = fgood
    y0 = 0
    z0 = [x0, y0]

    # Solve ODEs
    sol = solve_ivp(dPdt, [t_array[0], t_array[-1]], z0,
                    t_eval=t_array, args=(khyb, k1, k2), vectorized=True, method="LSODA")

    x, y = sol.y
    nonfunctional = (1 - fgood)
    fluorescence = (x + y + nonfunctional) 
    return fluorescence

# Fixed parameters (other than k1, k2)
def eval_loss_grid(k1_vals, k2_opt, fgood_opt, khyb_vals, t_all, y_all):
    """
    Evaluates the loss grid for a given set of parameters.
    
    Args:
    k1_vals: The values of k1.
    k2_opt: The optimal value of k2.
    fgood_opt: The optimal value of fgood.
    khyb_vals: The values of khyb.
    t_all: The time array.
    y_all: The fluorescence data.
    
    Returns:
    The loss grid.
    """
    K1, K2 = np.meshgrid(k1_vals, khyb_vals)
    Z = np.zeros_like(K1)
    for i in tqdm(range(K1.shape[0])):
        for j in range(K1.shape[1]):
            k1 = K1[i, j]
            khyb = K2[i, j]
            try:
                y_pred = fluorescence_H3_dynamic(t_all, fgood_opt, khyb, k1, k2_opt)
                Z[i, j] = np.mean((y_all - y_pred) ** 2)
            except:
                print("a")
                Z[i, j] = np.nan  # If the solver fails
    print(K1, K2, Z)
    return K1, K2, Z

def fit_model(time, fluorescence):
    """
    Fits the model to the data.
    
    Args:
    time: The time array.
    fluorescence: The fluorescence data.
    
    Returns:
    
    """
    def wrapper(t, fgood, khyb, k1, k2):
        """
        A wrapper to match curve_fit input format
        Args:
        t: The time array.
        fgood: The fraction of good tiles.
        khyb: The hybridization rate.
        k1: The binding rate.
        k2: The dissociation rate.
        
        Returns:
        best_loss: The best loss.
        best_popt: The best parameters.
        """

        return fluorescence_H3_dynamic(t, fgood, khyb, k1, k2)
    
    best_loss = np.inf
    best_popt = None
    losses = []
    params = []

    # Set bounds for each parameter (adjust as needed)
    fgood_range  = (0.0, 1.0)
    khyb_range   = (1e-5, 1e2)
    k1_range     = (1e-6, 10)
    k2_range     = (1e-3, 100)

    # Fit to downsampled data
    for _ in tqdm(range(200)):
        # Random p0 sampling
        p0 = [
            np.random.uniform(*fgood_range),
            np.random.uniform(*khyb_range),
            np.random.uniform(*k1_range),
            np.random.uniform(*k2_range)
        ]

        try:
            popt, _ = curve_fit(wrapper, time, fluorescence, p0=p0,
                                bounds=(
                                    [*map(lambda r: r[0], [fgood_range, khyb_range, k1_range, k2_range])],
                                    [*map(lambda r: r[1], [fgood_range, khyb_range, k1_range, k2_range])]
                                ),
                                maxfev=5000)
            
            y_pred = wrapper(t, *popt)
            loss = np.mean((y_pred - y) ** 2)

            losses.append(loss)
            params.append(popt)

            if loss < best_loss:
                best_loss = loss
                best_popt = popt

        except Exception as e:
            continue
    
    return best_loss, best_popt
    

def plot_results(time, fluorescence, best_popt, path):
    """
    Plots the results of the fit.
    
    Args:
    time: The time array.
    fluorescence: The fluorescence data.
    best_popt: The best parameters.
    """
    plt.plot(time, fluorescence, 'o', label='Data')
    plt.plot(time, wrapper(time, *best_popt), '-', label='Best Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence')
    plt.title("Fluorescence vs. Time")
    plt.legend()
    #Save results in path
    plt.savefig(path)
    plt.close()
        
    
    



