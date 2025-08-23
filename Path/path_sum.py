import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import norm, gamma
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import torch  # Add PyTorch import
import matplotlib.pyplot as plt
import seaborn as sns

def sum_edges(arcs, model, dist_type='normal'):
    """
    Progressively sum arcs in a path (a+b, then (a+b)+c, etc.)
    
    Args:
        arcs: List of arcs, each with mean_time and std_time
        model: Prediction model for combining arcs
        dist_type: Type of distribution to fit ('normal' or 'gamma')
        
    Returns:
        dict_edges: Dictionary of edge indices and their combinations
        dict_sums: Dictionary of progressive sums with their properties
    """
    dict_edges = {}
    dict_sums = {}
    
    # Check if the model is a PyTorch model (has 'forward' method)
    is_torch_model = hasattr(model, 'forward')
    
    for i in range(len(arcs)-1):
        if i == 0:
            # First combination: arc[0] + arc[1]
            arc1 = arcs[i]
            arc2 = arcs[i+1]
            # Store index combinations instead of using dicts as keys
            dict_edges[i] = (0, 1)  # Indicating we're combining arc[0] and arc[1]
        elif i >= 1:
            # Subsequent combinations: previous_result + next_arc
            arc1 = dict_sums[i]  # Previous combined result
            arc2 = arcs[i+1]     # Next arc in the path
            # Store index combinations
            dict_edges[i] = (f'combined_{i}', i+1)  # Combining previous result with next arc
            
        # Prepare parameters for the model
        params_np = np.array([
            arc1['mean_time'], arc1['std_time'],
            arc2['mean_time'], arc2['std_time']
        ]).reshape(1, 4)
        
        # Generate percentiles for prediction
        percentiles = np.array(list(np.arange(0.05, 1.01, 0.05)))
        
        try:
            # Handle PyTorch models (convert numpy to tensor)
            if is_torch_model:
                # Convert numpy array to PyTorch tensor
                params_tensor = torch.tensor(params_np, dtype=torch.float32)
                
                # Set model to evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                    
                # Use with torch.no_grad() to disable gradient calculation
                with torch.no_grad():
                    # Forward pass
                    prediction_tensor = model(params_tensor)
                    
                    # Convert prediction back to numpy for further processing
                    if isinstance(prediction_tensor, torch.Tensor):
                        prediction = prediction_tensor.cpu().numpy()
                    else:
                        prediction = prediction_tensor
            # Handle other model types
            elif hasattr(model, 'predict'):
                prediction = model.predict(params_np)
            elif hasattr(model, '__call__'):
                prediction = model(params_np)
            elif hasattr(model, 'predict_values'):
                prediction = model.predict_values(params_np)
            else:
                # Last resort
                prediction = np.array([model.predict_single(p) for p in params_np])
        except Exception as e:
            raise ValueError(f"Could not get prediction from model: {e}. Available methods: {dir(model)}")
        
        # Ensure prediction is a numpy array
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        elif not isinstance(prediction, np.ndarray):
            prediction = np.array(prediction)
        
        # Fix: Safely handle prediction flattening without index errors
        # Instead of trying to access prediction[0] which might not exist,
        # check the shape and extract appropriately
        if prediction.ndim > 1 and prediction.shape[0] == 1:
            # If prediction has shape (1, n), extract the first row
            prediction_flattened = prediction[0]
        else:
            # Otherwise, use the prediction as is
            prediction_flattened = prediction
            
        # Ensure prediction_flattened is properly shaped for the fit function
        if len(prediction_flattened) != len(percentiles):
            # If shapes don't match, we need to reshape or broadcast
            if len(prediction_flattened) == 1:
                # If prediction is a single value, broadcast to match percentiles
                prediction_flattened = np.full_like(percentiles, prediction_flattened.item())
            elif len(prediction_flattened) > len(percentiles):
                # If too many values, truncate
                prediction_flattened = prediction_flattened[:len(percentiles)]
            else:
                # If too few values, pad with the last value
                padding = np.full(len(percentiles) - len(prediction_flattened), prediction_flattened[-1])
                prediction_flattened = np.concatenate([prediction_flattened, padding])
        
        # Fit the appropriate distribution to the prediction
        if dist_type.lower() == 'gamma':
            direct_samples, alpha, beta = fit_and_generate_from_gamma(prediction_flattened, percentiles)
            dict_sums[i+1] = {'samples': direct_samples, 'mean_time': alpha/beta, 'std_time': np.sqrt(alpha)/beta, 
                             'alpha': alpha, 'beta': beta, 'dist_type': 'gamma'}
        else:
            # Default to normal distribution
            direct_samples, mu, sigma = fit_and_generate_from_normal(prediction_flattened, percentiles)
            dict_sums[i+1] = {'samples': direct_samples, 'mean_time': mu, 'std_time': sigma, 'dist_type': 'normal'}

    return dict_edges, dict_sums


def fit_and_generate_from_normal(x_values, cdf_values, num_samples=10000):
    """
    Fit a normal distribution to given percentiles and generate samples.
    
    Args:
        x_values: Predicted values for each percentile
        cdf_values: Percentile values (0.05, 0.10, ..., 1.00)
        num_samples: Number of samples to generate
        
    Returns:
        samples: Generated samples from the fitted normal distribution
        mu: Fitted mean parameter
        sigma: Fitted standard deviation parameter
    """
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


def fit_and_generate_from_gamma(x_values, cdf_values, num_samples=10000):
    """
    Fit a gamma distribution to given percentiles and generate samples.
    
    Args:
        x_values: Predicted values for each percentile
        cdf_values: Percentile values (0.05, 0.10, ..., 1.00)
        num_samples: Number of samples to generate
        
    Returns:
        samples: Generated samples from the fitted gamma distribution
        alpha: Fitted shape parameter
        beta: Fitted rate parameter
    """
    def gamma_cdf_error(params, x_values, cdf_values):
        alpha, beta = params
        # Ensure parameters are valid
        if alpha <= 0 or beta <= 0:
            return 1e10  # High error for invalid parameters
        
        predicted_cdf = gamma.cdf(x_values, a=alpha, scale=1.0/beta)
        return np.sum((predicted_cdf - cdf_values)**2)
    
    # Ordenar valores para interpolación
    sorted_idx = np.argsort(cdf_values)
    x_sorted = np.array(x_values)[sorted_idx]
    cdf_sorted = np.array(cdf_values)[sorted_idx]
    
    # Función para estimar percentiles
    get_percentile = interp1d(cdf_sorted, x_sorted, bounds_error=False, 
                             fill_value=(np.min(x_sorted), np.max(x_sorted)))
    
    # Estimar estadísticas iniciales
    mean_est = get_percentile(0.5)  # Usar mediana como aproximación inicial
    q25 = get_percentile(0.25)
    q75 = get_percentile(0.75)
    std_est = (q75 - q25) / 1.35  # Aproximación de desviación estándar basada en IQR
    
    # Estimar parámetros iniciales de gamma basados en método de momentos
    # Para gamma: media = alpha/beta, varianza = alpha/beta^2
    var_est = std_est**2
    if var_est > 0 and mean_est > 0:
        alpha_est = (mean_est**2) / var_est
        beta_est = mean_est / var_est
    else:
        # Valores predeterminados si las estimaciones no son útiles
        alpha_est = 2.0
        beta_est = 1.0
    
    # Restricciones para parámetros (ambos deben ser positivos)
    bounds = [(0.01, None), (0.01, None)]
    
    # Optimizar para encontrar mejores parámetros
    initial_guess = [alpha_est, beta_est]
    try:
        result = minimize(gamma_cdf_error, initial_guess, 
                         args=(x_values, cdf_values), 
                         method='L-BFGS-B',
                         bounds=bounds)
        
        alpha_opt, beta_opt = result.x
        
        # Si la optimización falla o da valores no razonables, usar estimaciones iniciales
        if not result.success or alpha_opt <= 0 or beta_opt <= 0:
            alpha_opt, beta_opt = alpha_est, beta_est
    except:
        # Fallback en caso de error en la optimización
        alpha_opt, beta_opt = alpha_est, beta_est
    
    # Generar muestras de la distribución gamma ajustada
    # Nota: En scipy.stats.gamma, scale = 1/beta (inverso de la tasa)
    samples = gamma.rvs(a=alpha_opt, scale=1.0/beta_opt, size=num_samples)
    
    return samples, alpha_opt, beta_opt


def plot_resulting_norm(dict_sums, arcs):
    """
    Plot the resulting distribution (normal or gamma) from summing edges.
    
    Args:
        dict_sums: Dictionary with progressive sums info
        arcs: List of arcs used in the path
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Get data from the final combination
    result_data = dict_sums[len(arcs)-1]
    data = result_data['samples']  
    dist_type = result_data.get('dist_type', 'normal')
    
    # Calculate basic statistics for the histogram title
    mean = np.mean(data)
    std_dev = np.std(data)

    # Plot histogram with density curve
    sns.histplot(data, kde=True, color='skyblue', alpha=0.6, label="Distribución calculada")
    
    # Add vertical line for the mean
    plt.axvline(x=mean, color='blue', linestyle='--')
    
    # Different labels based on distribution type
    if dist_type.lower() == 'gamma' and 'alpha' in result_data and 'beta' in result_data:
        alpha = result_data['alpha']
        beta = result_data['beta']
        plt.axvline(x=mean, color='blue', linestyle='--', 
                    label=f'Gamma: α={alpha:.2f}, β={beta:.2f}, μ={mean:.2f}, σ={std_dev:.2f}')
    else:
        plt.axvline(x=mean, color='blue', linestyle='--', 
                    label=f'Normal: μ={mean:.2f}, σ={std_dev:.2f}')

    plt.legend()
    plt.title(f'Distribución {dist_type.capitalize()} para la Suma de {len(arcs)} Arcos')
    plt.xlabel('Tiempo')
    plt.ylabel('Densidad')

    plt.tight_layout()
    plt.show()