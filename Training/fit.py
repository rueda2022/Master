import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d

def fit_and_generate_from_normal(x_values, cdf_values, num_samples=10000):
    
    
    def normal_cdf_error(params, x_values, cdf_values):
        mu, sigma = params
        predicted_cdf = norm.cdf(x_values, loc=mu, scale=sigma)
        return np.sum((predicted_cdf - cdf_values)**2)
    
    # Estimación inicial usando mediana y IQR
    sorted_idx = np.argsort(cdf_values)
    x_sorted = np.array(x_values)[sorted_idx]
    cdf_sorted = np.array(cdf_values)[sorted_idx]
    
    # Función para estimar percentiles
    get_percentile = interp1d(cdf_sorted, x_sorted, bounds_error=False, 
                             fill_value=(np.min(x_sorted), np.max(x_sorted)))
    
    # Estimar media y desviación estándar iniciales
    mu_est = get_percentile(0.5)  # mediana
    q25 = get_percentile(0.25)
    q75 = get_percentile(0.75)
    sigma_est = (q75 - q25) / 1.35  # Aproximación basada en IQR
    
    # Optimizar para encontrar mejores parámetros
    initial_guess = [mu_est, sigma_est]
    result = minimize(normal_cdf_error, initial_guess, 
                     args=(x_values, cdf_values), 
                     method='Nelder-Mead')
    
    mu_opt, sigma_opt = result.x
    
    # Generar muestras directamente de la distribución normal ajustada
    samples = np.random.normal(loc=mu_opt, scale=sigma_opt, size=num_samples)
    
    return samples, mu_opt, sigma_opt