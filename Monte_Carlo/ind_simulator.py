import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Orchestrator(distributions=[], num_samples=10000, iterations=1000, T=0):
    initial_data = []
    for dist in distributions:
        data = create_data(num_samples=num_samples, 
                    distribution=dist["dist_name"],
                    params=dist["params"])    
        initial_data.append(data)
    
    ## Run the simulation
    results = run_simulation(iterations=iterations, data_set=initial_data)
    statistics = get_summary_statistics(results,
                                        quantiles=[0.25, 0.5, 0.75],
                                        value=T)
    
    plot_data(results, title='Monte Carlo Simulation Results', statistics=statistics)
    return statistics    


## Monte Carlo Simulation for Data Generation.
def create_data(num_samples=10000, distribution='normal', params=None):
    if distribution == 'normal':
        mu, std = params["mu"], params["std"]
        data = np.random.normal(mu, std, num_samples)
    elif distribution == 'Gamma':
        shape, scale = params["shape"], params["scale"]
        data = np.random.Gamma(shape, scale, num_samples)
    return data

## Function that plots the generated data with improved aesthetics
def plot_data(data, title='Data Distribution', statistics=None):
    # Convertir data a un arreglo de NumPy si no lo es
    data = np.array(data).flatten()  # Asegurarse de que sea un arreglo 1D numérico
    
    # Set the aesthetic style
    sns.set_theme(style="whitegrid")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Use passed statistics if available, otherwise calculate them
    if statistics:
        mean_val = statistics['mean']
        std_val = statistics['std']
        q1, median_val, q3 = statistics['quantiles']
    else:
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        q1, q3 = np.percentile(data, [25, 75])
    
    # Plot histogram with KDE in the top subplot
    sns.histplot(data, bins=50, kde=True, color='royalblue', alpha=0.7, 
                edgecolor='none', ax=ax1)
    
    # Add vertical lines for statistics
    ax1.axvline(mean_val, color='crimson', linestyle='--', linewidth=2, 
              label=f'Media: {mean_val:.2f}')
    ax1.axvline(median_val, color='forestgreen', linestyle='-', linewidth=2, 
              label=f'Mediana: {median_val:.2f}')
    
    # Add statistical annotations
    textstr = (f'Media: {mean_val:.2f}\n'
               f'Mediana: {median_val:.2f}\n'
               f'Desv. Estándar: {std_val:.2f}\n'
               f'Q1: {q1:.2f}\n'
               f'Q3: {q3:.2f}')
    
    # Add probability if available
    if statistics and 'probability_less_than' in statistics:
        prob = statistics['probability_less_than']
        textstr += f'\nP(X < T): {prob:.4f}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Enhance the plot appearance
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Valor', fontsize=14)
    ax1.set_ylabel('Frecuencia', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Add boxplot in the bottom subplot
    sns.boxplot(x=data, ax=ax2, color='royalblue')
    ax2.set_title('Diagrama de Caja', fontsize=14)
    
    plt.tight_layout()
    plt.show()

## Sum two data sets
def independent_sum(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Data sets must be of the same length.")
    return data1 + data2

## Function to run the simulation
def run_simulation(iterations=1000, data_set=[]):
    final_results = []
    
    # For each iteration, we'll randomly select one sample from each dataset
    # and sum them to get a single result for that iteration
    for _ in range(iterations):
        # For each iteration, randomly select indices to sample from each distribution
        idx = np.random.randint(0, len(data_set[0]))
        
        # Sum the selected samples from each distribution
        iteration_sum = sum(data[idx] for data in data_set)
        final_results.append(iteration_sum)
    
    return final_results


def get_summary_statistics(data, quantiles=[0.25, 0.5, 0.75], value=None):
    # Convertir data a un arreglo de NumPy si no lo es
    data = np.array(data)
    
    summary = {
        'mean': np.mean(data),
        'std': np.std(data),
        'quantiles': np.quantile(data, quantiles)
    }
    if value is not None:
        summary['probability_less_than'] = np.mean(data < value)
    return summary

