import numpy as np
# from scipy.stats import Gamma
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def filter_outliers(data, method='iqr', multiplier=1.5):
    """
    Filter outliers from a dataset.
    
    Parameters:
    -----------
    data : list or array
        The data to filter
    method : str, default='iqr'
        The method to use for filtering outliers ('iqr' only for now)
    multiplier : float, default=1.5
        The multiplier for the IQR (standard is 1.5)
        
    Returns:
    --------
    filtered_data : array
        The data with outliers removed
    outliers : array
        The identified outliers
    thresholds : tuple
        (lower_bound, upper_bound) thresholds used for filtering
    """
    data = np.array(data)
    
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Identify which points are outliers
        is_outlier = (data < lower_bound) | (data > upper_bound)
        
        # Separate outliers and non-outliers
        filtered_data = data[~is_outlier]
        outliers = data[is_outlier]
        
        return filtered_data, outliers, (lower_bound, upper_bound)
    else:
        raise ValueError(f"Method '{method}' not supported")


def get_parameters(G, filter_outliers_data=False):
    params = {}
    for u, v, data in G.edges(data=True):
        time_vals = data.get("Time", None)
        time_limits = (None, None)
        if time_vals is not None:
            try:
                if isinstance(time_vals, (list, tuple, np.ndarray)):
                    if len(time_vals) > 0:
                        tmin = min(time_vals)
                        tmax = max(time_vals)
                        time_limits = (tmin, tmax)
                else:
                    time_limits = (time_vals, time_vals)
            except (ValueError, TypeError):
                time_limits = (None, None)

        params[(u, v)] = {
            "Distance": data["Distance"],
            "Time limits": time_limits,
        }

    all_times = []
    for _, _, data in G.edges(data=True):
        t = data.get("Time", None)
        if t is None:
            continue
        if isinstance(t, (list, tuple, np.ndarray)):
            all_times.extend([x for x in t if x is not None])
        else:
            all_times.append(t)

    overall_min_time = min(all_times) if len(all_times) > 0 else None
    overall_max_time = max(all_times) if len(all_times) > 0 else None

    # Extract all distance values
    all_distances = [data["Distance"] for _, _, data in G.edges(data=True)]
    
    # Calculate descriptive statistics for distances (no filtering)
    dist_stats = {
        "median": np.median(all_distances),
        "q1": np.percentile(all_distances, 25),
        "q3": np.percentile(all_distances, 75),
        "iqr": np.percentile(all_distances, 75) - np.percentile(all_distances, 25),
        "min": np.min(all_distances),
        "max": np.max(all_distances)
    }
    
    # Calculate time statistics if we have time data
    time_stats = None
    if len(all_times) > 0:
        if filter_outliers_data:
            filtered_times, time_outliers, _ = filter_outliers(all_times)
            # Use filtered data for statistics
            all_times_for_stats = filtered_times
            time_stats = {
                "median": np.median(filtered_times),
                "q1": np.percentile(filtered_times, 25),
                "q3": np.percentile(filtered_times, 75),
                "iqr": np.percentile(filtered_times, 75) - np.percentile(filtered_times, 25),
                "min": np.min(filtered_times),
                "max": np.max(filtered_times),
                "outliers_removed": len(time_outliers)
            }
        else:
            time_stats = {
                "median": np.median(all_times),
                "q1": np.percentile(all_times, 25),
                "q3": np.percentile(all_times, 75),
                "iqr": np.percentile(all_times, 75) - np.percentile(all_times, 25),
                "min": np.min(all_times),
                "max": np.max(all_times),
                "outliers_removed": 0
            }

    overall_params = {
        "Overall Distance": np.mean([data["Distance"] for u, v, data in G.edges(data=True)]),
        "Overall Time limits": (overall_min_time, overall_max_time),
        "Distance Statistics": dist_stats,
        "Time Statistics": time_stats
    }
    return overall_params


def create_boxplot(G, show_plot=True, save_path=None, remove_time_outliers=True, compare=False):
    """
    Create boxplots for Time and Distance values in the graph that properly handle outliers.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph containing edge data with Time and Distance attributes
    show_plot : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    remove_time_outliers : bool, default=True
        Whether to remove outliers from time data before plotting (default changed to True)
    compare : bool, default=False
        If True, show both original and filtered time data side by side
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    stats : dict
        Statistical values shown in the boxplot
    """
    # Extract data
    distances = [data["Distance"] for _, _, data in G.edges(data=True)]
    
    # Extract all time values (flatten lists if needed)
    times = []
    for _, _, data in G.edges(data=True):
        t = data.get("Time", None)
        if t is None:
            continue
        if isinstance(t, (list, tuple, np.ndarray)):
            times.extend([x for x in t if x is not None])
        else:
            times.append(t)
    
    # Filter time outliers if requested (distance outliers are not filtered)
    times_filtered = times
    time_outliers = []
    if times and remove_time_outliers:
        times_filtered, time_outliers, time_bounds = filter_outliers(times)
    
    # Determine number of plots based on compare parameter
    if compare and times:
        # 2 columns: distance and time (original vs filtered time)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Distance plot (no filtering)
        dist_stats = boxplot_with_stats(axs[0], distances, "Edge Distances")
        
        # Time plots with comparison
        time_stats_orig = boxplot_with_stats(axs[1], times, "Original Times")
        time_stats_filt = boxplot_with_stats(axs[2], times_filtered, 
                                    f"Filtered Times\n({len(time_outliers)} outliers removed)")
    else:
        # Just distance and time (filtered if requested)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Distance plot (no filtering)
        dist_stats = boxplot_with_stats(axs[0], distances, "Edge Distances")
        
        # Time plot (filtered by default)
        time_stats = None
        if times:
            title = f"Edge Times{' (Outliers Removed)' if remove_time_outliers else ''}"
            time_stats = boxplot_with_stats(axs[1], times_filtered, title)
        else:
            axs[1].set_title("No Time Data Available")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    # Return stats based on which version we created
    if compare and times:
        return fig, axs, {
            "Distance": dist_stats,
            "Time": {"original": time_stats_orig, "filtered": time_stats_filt}
        }
    else:
        return fig, axs, {
            "Distance": dist_stats,
            "Time": time_stats
        }


def boxplot_with_stats(ax, data, title):
    """
    Create a boxplot with statistical annotations.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to draw the boxplot on
    data : list or array
        The data to plot
    title : str
        The title for the plot
        
    Returns:
    --------
    stats : dict
        Statistical values shown in the boxplot
    """
    # Calculate statistics
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = max(np.min(data), q1 - 1.5 * iqr)
    upper_whisker = min(np.max(data), q3 + 1.5 * iqr)
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, vert=True, whis=1.5)
    
    # Customize boxplot appearance
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    
    plt.setp(bp['boxes'], facecolor='lightblue')
    plt.setp(bp['fliers'], markerfacecolor='red', markersize=5)
    
    # Add statistics as text
    stats_text = (
        f"Median: {median:.2f}\n"
        f"Q1: {q1:.2f}\n"
        f"Q3: {q3:.2f}\n"
        f"IQR: {iqr:.2f}\n"
        f"Lower whisker: {lower_whisker:.2f}\n"
        f"Upper whisker: {upper_whisker:.2f}\n"
        f"Outliers: {sum(1 for x in data if x < lower_whisker or x > upper_whisker)}"
    )
    
    # Position the text box in the top right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    stats = {
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers_count": sum(1 for x in data if x < lower_whisker or x > upper_whisker)
    }
    
    return stats

# def create_random_params(dist, corr):
#     parameters = []
#     p1 = p2 = p3 = p4 = p5 = 0

#     if dist == ('normal', 'normal') and not corr:
#         p1 = np.random.uniform(0, 5)      # mu1
#         p2 = np.random.uniform(0.1, 1.5)  # sigma1
#         p3 = np.random.uniform(0, 5)      # mu2
#         p4 = np.random.uniform(0.1, 1.5)  # sigma2
#         parameters = [p1, p2, p3, p4]
#     elif dist == ('normal', 'normal') and corr:
#         p1 = np.random.uniform(0, 5)
#         p2 = np.random.uniform(0.1, 1.5)
#         p3 = np.random.uniform(0, 5)
#         p4 = np.random.uniform(0.1, 1.5)
#         p5 = np.random.uniform(0, 0.9)
#         parameters = [p1, p2, p3, p4, p5]
#     elif dist == ('Gamma', 'Gamma') and not corr:
#         p1 = np.random.uniform(0.1, 5)    # alpha1 > 0
#         p2 = np.random.uniform(0.1, 1.5)  # beta1
#         p3 = np.random.uniform(0.1, 5)    # alpha2 > 0
#         p4 = np.random.uniform(0.1, 1.5)  # beta2
#         parameters = [p1, p2, p3, p4]
#     else:
#         raise ValueError(f"Combinación no soportada: dist={dist}, corr={corr}")
    
#     return parameters

# def create_distribution(dist, corr, n, parameters):
#     X1 = np.zeros(n)
#     X2 = np.zeros(n)
    
#     if dist == ('normal', 'normal', ) and not corr:
#         X1 = np.random.normal(parameters[0], parameters[1], size=n)
#         X2 = np.random.normal(parameters[2], parameters[3], size=n)
#     elif dist == ('normal', 'normal') and corr == True :
#         Z1 = np.random.normal(0, 1, size=n)
#         Z2 = np.random.normal(0, 1, size=n)
#         X1 = parameters[0] + parameters[1] * Z1
#         X2 = parameters[2] + parameters[3] * (parameters[4] * Z1 + np.sqrt(1 - parameters[4]**2) * Z2)
#     elif dist == ('Gamma', 'Gamma') and not corr:
#         X1 = Gamma.rvs(a=parameters[0], scale=1/parameters[1], size=n)
#         X2 = Gamma.rvs(a=parameters[2], scale=1/parameters[3], size=n)
#     else:
#         raise ValueError(f"Combinación no soportada: dist={dist}, corr={corr}")
    
#     return X1, X2

# def data_simulation(n, dist=('normal', 'normal'), corr=False):
#     N = 10000
#     inputs = []
#     outputs = []
#     percentiles = np.arange(0.05, 1.01, 0.05)

#     for i in range(N):
#         params = create_random_params(dist, corr)
#         inputs.append(params)  
        
#         X1, X2 = create_distribution(dist, corr, n, params)
#         Y = X1 + X2
#         Y = np.sort(Y)
        
#         ni = [np.percentile(Y, p * 100) for p in percentiles]
#         outputs.append(ni)

#     inputs = np.array(inputs)
#     outputs = np.array(outputs)
    
#     return {'inputs': inputs, 'outputs': outputs, 'percentiles': percentiles}