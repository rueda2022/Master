from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import networkx as nx
from datetime import datetime, timedelta
import re
import os
from Training import random_data
from Training import train
from Path import path_sum
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
# Add new imports for map background
import contextily as ctx
from pyproj import Transformer
from matplotlib.offsetbox import AnchoredText
import Orchestator


# Add: custom graph classes to allow G[(u, v)] access
class AccessDiGraph(nx.DiGraph):
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            u, v = key
            return self.get_edge_data(u, v)
        return super().__getitem__(key)


class AccessGraph(nx.Graph):
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            u, v = key
            return self.get_edge_data(u, v)
        return super().__getitem__(key)


def read_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read data from a CSV file, handling comments and column stripping.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame or None if reading fails
    """
    try:
        df = pd.read_csv(file_path, comment="/")
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def infer_time_columns(data: pd.DataFrame, distance_col: str = "Distance") -> List[str]:
    """
    Infer which columns contain time data based on the position of the Distance column.
    
    Args:
        data: DataFrame with the data
        distance_col: Name of the distance column
        
    Returns:
        List of column names that contain time data
    """
    cols = list(map(str, data.columns))
    norm = [c.strip().lower() for c in cols]
    target = distance_col.strip().lower()
    matches = [i for i, n in enumerate(norm) if (n == target) or n.startswith(target) or ("Distance" in n)]
    if not matches:
        raise ValueError(f"No se encontró la columna '{distance_col}'.")
    dist_idx = matches[0]
    
    # All columns after the distance column are assumed to be time columns
    time_cols = cols[dist_idx + 1:]
    
    # Verify that they look like time columns (contain ":" or are numeric)
    valid_time_cols = []
    for col in time_cols:
        sample_vals = data[col].dropna().astype(str).head(5).tolist()
        if any(":" in str(val) for val in sample_vals) or all(re.match(r'^[\d\.]+$', str(val)) for val in sample_vals if str(val).strip()):
            valid_time_cols.append(col)
            
    return valid_time_cols


def parse_time_value(time_str: str) -> float:
    """
    Parse a time string into seconds, returning NaN for non-finite/negative values.
    """
    if pd.isna(time_str) or str(time_str).strip() == "":
        return np.nan
    # If already numeric, ensure it is finite and non-negative
    try:
        val = float(time_str)
        return float(val) if np.isfinite(val) and val >= 0 else np.nan
    except (ValueError, TypeError):
        pass
    # Try parsing as HH:MM:SS or MM:SS
    try:
        s = str(time_str)
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:  # HH:MM:SS
                h, m, s = map(float, parts)
                sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:  # MM:SS
                m, s = map(float, parts)
                sec = m * 60 + s
            else:
                sec = np.nan
            return float(sec) if np.isfinite(sec) and sec >= 0 else np.nan
    except (ValueError, TypeError):
        pass
    # Try parsing as timedelta
    try:
        td = pd.to_timedelta(time_str)
        sec = td.total_seconds()
        return float(sec) if np.isfinite(sec) and sec >= 0 else np.nan
    except (ValueError, TypeError):
        pass
    return np.nan


def get_edge_data(
    data: pd.DataFrame,
    source_col: str = "start_node",
    target_col: str = "end_node",
    distance_col: str = "Distance",
    time_cols: Optional[List[str]] = None,
    drop_zeros: bool = True,
    drop_na: bool = True,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Extract edge data from DataFrame for graph construction.
    
    Args:
        data: DataFrame with network data
        source_col: Column name for source nodes
        target_col: Column name for target nodes
        distance_col: Column name for distance data
        time_cols: List of column names containing time data (inferred if None)
        drop_zeros: Whether to drop zero time values
        drop_na: Whether to drop NA time values
        
    Returns:
        Dictionary mapping edge tuples to their attributes
    """
    if source_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Missing {source_col} or {target_col}")
    
    if time_cols is None:
        time_cols = infer_time_columns(data, distance_col=distance_col)
    
    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in data.iterrows():
        u = row[source_col]
        v = row[target_col]
        d_raw = pd.to_numeric(pd.Series([row.get(distance_col, np.nan)]), errors="coerce").iloc[0]
        d_val = None if pd.isna(d_raw) else float(d_raw)

        # Process time columns
        time_values = []
        for col in time_cols:
            val = row.get(col)
            if not pd.isna(val) and val != "":
                seconds = parse_time_value(val)
                if not pd.isna(seconds) and np.isfinite(seconds):
                    if not drop_zeros or seconds != 0:
                        time_values.append(float(seconds))

        key = (u, v)
        if key not in edge_data:
            edge_data[key] = {
                "Distance": d_val,
                "Time": [],
                "start_longitude": float(row.get("start_longitude", np.nan)),
                "start_latitude": float(row.get("start_latitude", np.nan)),
                "end_longitude": float(row.get("end_longitude", np.nan)),
                "end_latitude": float(row.get("end_latitude", np.nan))
            }
        
        if edge_data[key]["Distance"] is None and d_val is not None:
            edge_data[key]["Distance"] = float(d_val)
        
        if time_values:
            edge_data[key]["Time"].extend(time_values)

    return edge_data


def build_graph_from_edge_data(
    edge_data: Dict[Tuple[str, str], Dict[str, Any]],
    directed: bool = True,
) -> nx.Graph:
    """
    Build a NetworkX graph from edge data.
    
    Args:
        edge_data: Dictionary mapping edge tuples to their attributes
        directed: Whether to create a directed graph
        
    Returns:
        NetworkX graph
    """
    G = AccessDiGraph() if directed else AccessGraph()
    
    for (u, v), attrs in edge_data.items():
        G.add_edge(u, v, **attrs)
    
    return G


def plot_edge_distributions_from_graph(
    G: nx.Graph,
    num_edges: int = 10,
    min_count: int = 100,
    bins: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Plot histograms of travel times for selected edges.
    
    Args:
        G: NetworkX graph
        num_edges: Number of edges to plot
        min_count: Minimum number of time observations required
        bins: Number of histogram bins
        xlim: X-axis limits (min, max)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    sns.set(style="whitegrid")
    items = []
    for u, v, data in G.edges(data=True):
        ts = data.get("Time", [])
        if isinstance(ts, list):
            clean = [float(t) for t in ts if np.isfinite(t)]
            if len(clean) >= min_count:
                items.append(((u, v), clean))
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    items = items[:num_edges]
    
    if not items:
        print("No edges with sufficient data found.")
        return None

    cols = min(5, len(items))
    rows = int(np.ceil(len(items) / cols))
    
    if figsize is None:
        figsize = (5 * cols, 4 * rows)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, ((u, v), vals) in enumerate(items):
        ax = axes[i]
        vals_arr = np.asarray(vals, dtype=float)
        mean_val = float(np.mean(vals_arr)) if vals_arr.size else np.nan
        median_val = float(np.median(vals_arr)) if vals_arr.size else np.nan
        std_val = float(np.std(vals_arr)) if vals_arr.size else np.nan

        ax.hist(vals_arr, bins=bins, color="blue", alpha=0.7)

        if np.isfinite(mean_val):
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        if np.isfinite(median_val):
            ax.axvline(median_val, color='green', linestyle=':', alpha=0.7, label=f'Median: {median_val:.2f}')

        ax.set_title(f"Edge {u} \u2192 {v} (n={len(vals_arr)})")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)

    # Turn off any unused subplots
    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_graph(
    G: nx.Graph, 
    node_size: int = 8, 
    with_labels: bool = False,
    label_size: int = 6,
    figsize: Tuple[int, int] = (10, 8),
    edge_width_scale: float = 1.0,
    edge_width_attr: str = "Distance",
    node_color_attr: Optional[str] = None,
    title: Optional[str] = "Chicago Transportation Network",
    add_basemap: bool = True,
    basemap_zoom: int = 12,
    show_node_ids: bool = True,
    show_arrows: Optional[bool] = None,  # New parameter to control arrow display
    arrow_size: int = 15,                # Size of arrowheads
    bidirectional_color: str = "blue",   # Color for bidirectional edges
    one_way_color: str = "red"           # Color for one-way edges
) -> plt.Figure:
    """
    Plot the network graph with customizable styling and Chicago map background.
    
    Args:
        G: NetworkX graph
        node_size: Size of the nodes
        with_labels: Whether to display node labels
        label_size: Font size for node labels
        figsize: Figure size
        edge_width_scale: Scaling factor for edge widths
        edge_width_attr: Attribute to use for edge widths
        node_color_attr: Attribute to use for node colors
        title: Plot title
        add_basemap: Whether to add a background map
        basemap_zoom: Zoom level for the basemap (higher = more detail)
        show_node_ids: Whether to show node IDs with small font size
        show_arrows: Whether to show direction arrows (defaults to True for directed graphs)
        arrow_size: Size of arrowheads on directed edges
        bidirectional_color: Color for edges that exist in both directions
        one_way_color: Color for one-way edges
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine if the graph is directed
    is_directed = isinstance(G, nx.DiGraph)
    
    # If show_arrows is not explicitly set, default to the graph's directed property
    if show_arrows is None:
        show_arrows = is_directed
    
    # Use geographic positions if available, otherwise spring layout
    has_geo_positions = True
    if all(G.nodes[n].get('pos') is not None for n in G.nodes()):
        pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
    else:
        try:
            # Try to use geographic coordinates if available
            pos = {}
            for u, v, data in G.edges(data=True):
                if u not in pos and all(k in data for k in ['start_longitude', 'start_latitude']):
                    pos[u] = (data['start_longitude'], data['start_latitude'])
                if v not in pos and all(k in data for k in ['end_longitude', 'end_latitude']):
                    pos[v] = (data['end_longitude'], data['end_latitude'])
            
            # If not all nodes have positions, fall back to spring layout
            if len(pos) < len(G.nodes):
                pos = nx.spring_layout(G, seed=42)
                has_geo_positions = False
        except:
            pos = nx.spring_layout(G, seed=42)
            has_geo_positions = False
    
    # Set up edge widths based on attribute if available
    if edge_width_attr and any(edge_width_attr in d for _, _, d in G.edges(data=True)):
        edge_widths = []
        for _, _, d in G.edges(data=True):
            width = d.get(edge_width_attr, 1.0)
            if isinstance(width, list):
                width = np.mean(width) if width else 1.0
            edge_widths.append(float(width) * edge_width_scale)
    else:
        edge_widths = [1.0 * edge_width_scale] * len(G.edges)
    
    # Add Chicago map as background if geographic positions are available
    if add_basemap and has_geo_positions:
        # Convert geographic coordinates to web mercator for contextily
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        
        # Transform node positions to web mercator
        pos_web_mercator = {}
        for node, coords in pos.items():
            lon, lat = coords
            x, y = transformer.transform(lon, lat)
            pos_web_mercator[node] = (x, y)
        
        # Draw nodes and edges in web mercator projection
        if node_color_attr and node_color_attr in G.nodes[list(G.nodes)[0]]:
            node_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes]
            nx.draw_networkx_nodes(G, pos_web_mercator, node_size=node_size, node_color=node_colors, 
                                  cmap=plt.cm.viridis, ax=ax)
        else:
            nx.draw_networkx_nodes(G, pos_web_mercator, node_size=node_size, node_color="#1f77b4", 
                                  ax=ax, edgecolors='black')
        
        # For a directed graph, identify bidirectional edges
        if is_directed:
            # Find bidirectional edges (edges that exist in both directions)
            bidirectional_edges = []
            one_way_edges = []
            
            for u, v in G.edges():
                if G.has_edge(v, u):
                    # Edge exists in both directions
                    if (u, v) not in bidirectional_edges and (v, u) not in bidirectional_edges:
                        bidirectional_edges.append((u, v))
                else:
                    # One-way edge
                    one_way_edges.append((u, v))
            
            # Draw bidirectional edges
            if bidirectional_edges:
                bidirectional_widths = [edge_widths[list(G.edges()).index((u, v))] 
                                       for u, v in bidirectional_edges if (u, v) in G.edges()]
                nx.draw_networkx_edges(G, pos_web_mercator, edgelist=bidirectional_edges,
                                      width=bidirectional_widths, edge_color=bidirectional_color,
                                      alpha=0.7, ax=ax, arrows=show_arrows,
                                      arrowsize=arrow_size, arrowstyle='-|>')
            
            # Draw one-way edges
            if one_way_edges:
                one_way_widths = [edge_widths[list(G.edges()).index((u, v))] 
                                 for u, v in one_way_edges]
                nx.draw_networkx_edges(G, pos_web_mercator, edgelist=one_way_edges,
                                      width=one_way_widths, edge_color=one_way_color,
                                      alpha=0.7, ax=ax, arrows=show_arrows,
                                      arrowsize=arrow_size, arrowstyle='-|>')
        else:
            # For undirected graphs, just draw all edges normally
            nx.draw_networkx_edges(G, pos_web_mercator, width=edge_widths, 
                                  edge_color=bidirectional_color, alpha=0.7, ax=ax)
        
        # Add small node IDs if requested
        if show_node_ids or with_labels:
            font_size = label_size if show_node_ids else 8
            labels = nx.draw_networkx_labels(G, pos_web_mercator, font_size=font_size, 
                                          font_color="black", font_weight="bold", ax=ax)
            
            # Add path effects to make labels more visible
            for _, t in labels.items():
                t.set_path_effects([
                    PathEffects.withStroke(linewidth=2, foreground='white'),
                    PathEffects.withStroke(linewidth=1, foreground='gray')
                ])
                # Set bbox properties for a small white background
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1, boxstyle='round,pad=0.1'))
        
        # Get bounds of the graph for better centering
        x_vals = [x for _, (x, _) in pos_web_mercator.items()]
        y_vals = [y for _, (_, y) in pos_web_mercator.items()]
        
        if x_vals and y_vals:  # Ensure we have valid positions
            # Add the basemap
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=basemap_zoom)
            
            # Add a more elegant compass rose (North arrow) in the bottom left
            # Use a smaller size and more elegant design
            ax_coords = [0.02, 0.02, 0.05, 0.05]  # Position in figure coordinates (smaller)
            compass_ax = fig.add_axes(ax_coords)
            compass_ax.set_aspect('equal')
            compass_ax.set_xlim(-1, 1)
            compass_ax.set_ylim(-1, 1)
            
            # Draw a more elegant north arrow
            # Create a background for the arrow for better visibility
            circle = plt.Circle((0, 0), 0.8, fc='white', ec='black', alpha=0.7)
            compass_ax.add_patch(circle)
            
            # Draw the arrow with a thinner design
            compass_ax.arrow(0, -0.5, 0, 0.9, head_width=0.2, head_length=0.2, 
                           fc='black', ec='black', linewidth=1.5)
            compass_ax.text(0, 0.6, 'N', ha='center', va='bottom', fontsize=8, 
                          fontweight='bold')
            compass_ax.axis('off')
    else:
        # Draw nodes and edges with standard layout
        if node_color_attr and node_color_attr in G.nodes[list(G.nodes)[0]]:
            node_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                                  cmap=plt.cm.viridis, ax=ax)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#1f77b4", ax=ax)
        
        # Handle directed vs undirected edges
        if is_directed:
            # Find bidirectional edges (edges that exist in both directions)
            bidirectional_edges = []
            one_way_edges = []
            
            for u, v in G.edges():
                if G.has_edge(v, u):
                    # Edge exists in both directions
                    if (u, v) not in bidirectional_edges and (v, u) not in bidirectional_edges:
                        bidirectional_edges.append((u, v))
                else:
                    # One-way edge
                    one_way_edges.append((u, v))
            
            # Draw bidirectional edges
            if bidirectional_edges:
                bidirectional_widths = [edge_widths[list(G.edges()).index((u, v))] 
                                       for u, v in bidirectional_edges if (u, v) in G.edges()]
                nx.draw_networkx_edges(G, pos, edgelist=bidirectional_edges,
                                      width=bidirectional_widths, edge_color=bidirectional_color,
                                      alpha=0.7, ax=ax, arrows=show_arrows, 
                                      arrowsize=arrow_size, arrowstyle='-|>')
            
            # Draw one-way edges
            if one_way_edges:
                one_way_widths = [edge_widths[list(G.edges()).index((u, v))] 
                                 for u, v in one_way_edges]
                nx.draw_networkx_edges(G, pos, edgelist=one_way_edges,
                                      width=one_way_widths, edge_color=one_way_color,
                                      alpha=0.7, ax=ax, arrows=show_arrows,
                                      arrowsize=arrow_size, arrowstyle='-|>')
            
            # Add legend for directed graphs
            if show_arrows:
                bidir = mpatches.Patch(color=bidirectional_color, label='Bidirectional')
                one_way = mpatches.Patch(color=one_way_color, label='One-way')
                legend = ax.legend(handles=[bidir, one_way], loc='upper right', 
                                  frameon=True, framealpha=0.8)
                legend.get_frame().set_facecolor('white')
        else:
            # For undirected graphs, just draw all edges normally
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#999999", 
                                  alpha=0.8, ax=ax)
    
    # Add a descriptive title with proper positioning at the top
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add information about the graph
    info_text = f"Network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    at = AnchoredText(info_text, loc='lower right', frameon=True, prop=dict(size=10))
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.8)
    ax.add_artist(at)
    
    plt.axis("off")
    plt.tight_layout()
    return fig


def add_pos_to_graph(G: nx.Graph) -> nx.Graph:
    """
    Add position attributes to graph nodes based on edge geographic data.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Updated NetworkX graph
    """
    # Create position dictionary from edge geographic coordinates
    pos = {}
    for u, v, data in G.edges(data=True):
        if all(k in data for k in ['start_longitude', 'start_latitude']):
            if u not in pos:
                pos[u] = (data['start_longitude'], data['start_latitude'])
        if all(k in data for k in ['end_longitude', 'end_latitude']):
            if v not in pos:
                pos[v] = (data['end_longitude'], data['end_latitude'])
    
    # Add positions to node attributes
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
    
    return G


def calculate_edge_statistics(G: nx.Graph) -> nx.Graph:
    """
    Calculate statistics for time data on each edge and add to edge attributes.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Updated NetworkX graph
    """
    for u, v, data in G.edges(data=True):
        times = data.get('Time', [])
        if isinstance(times, list):
            clean = [float(t) for t in times if np.isfinite(t)]
            if clean:
                G[u][v]['mean_time'] = float(np.mean(clean))
                G[u][v]['median_time'] = float(np.median(clean))
                G[u][v]['std_time'] = float(np.std(clean))
                G[u][v]['min_time'] = float(np.min(clean))
                G[u][v]['max_time'] = float(np.max(clean))
                G[u][v]['count'] = len(clean)
    
    return G


def _sanitize_graph_for_io(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of G with attributes converted to GraphML/GML-friendly types.
    - Splits node 'pos' tuple into 'pos_x' and 'pos_y' (only if finite), removes 'pos'.
    - Drops edge 'Time' (list), which is too large and unsupported.
    - Converts numpy scalars to Python scalars.
    - Removes None values.
    - Converts remaining tuples/lists/dicts to strings as a fallback.
    """
    H = G.copy()

    # Sanitize node attributes
    for n, attrs in list(H.nodes(data=True)):
        if 'pos' in attrs and isinstance(attrs['pos'], tuple) and len(attrs['pos']) == 2:
            x, y = attrs['pos']
            # Only keep finite coordinates
            if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x)):
                H.nodes[n]['pos_x'] = float(x)
            if isinstance(y, (int, float, np.integer, np.floating)) and np.isfinite(float(y)):
                H.nodes[n]['pos_y'] = float(y)
            del H.nodes[n]['pos']

        for k, v in list(H.nodes[n].items()):
            if v is None:
                del H.nodes[n][k]
                continue
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                H.nodes[n][k] = v.item()
            elif isinstance(v, (tuple, list, dict)):
                H.nodes[n][k] = str(v)

    # Sanitize edge attributes
    for u, v, attrs in list(H.edges(data=True)):
        if 'Time' in attrs:
            del H[u][v]['Time']  # drop list
        for k, val in list(H[u][v].items()):
            if val is None:
                del H[u][v][k]
                continue
            if isinstance(val, (np.floating, np.integer, np.bool_)):
                H[u][v][k] = val.item()
            elif isinstance(val, (tuple, list, dict)):
                H[u][v][k] = str(val)

    return H


def save_graph_to_file(G: nx.Graph, filename: str) -> None:
    """
    Save graph to a file in multiple formats.
    """
    # Sanitize a copy for safe serialization
    H = _sanitize_graph_for_io(G)

    # Try GraphML (will fall back to pure-XML if lxml is missing in NetworkX)
    try:
        nx.write_graphml(H, f"{filename}.graphml")
    except Exception as e:
        print(f"Warning: Could not write GraphML: {e}. Skipping .graphml")

    # Try GML
    try:
        nx.write_gml(H, f"{filename}.gml")
    except Exception as e:
        print(f"Warning: Could not write GML: {e}. Skipping .gml")

    # Save as edge list with attributes (using sanitized graph as well)
    try:
        with open(f"{filename}.edgelist", 'w') as f:
            for u, v, data in H.edges(data=True):
                attrs = []
                for k, val in data.items():
                    if isinstance(val, list):
                        val_str = f"[{','.join(map(str, val[:5]))}...]" if len(val) > 5 else str(val)
                    else:
                        val_str = str(val)
                    attrs.append(f"{k}={val_str}")
                f.write(f"{u} {v} {' '.join(attrs)}\n")
        print(f"Graph saved to {filename}.[graphml|gml|edgelist]")
    except Exception as e:
        print(f"Warning: Could not write edge list: {e}")


def orchestrate(
    file_path: str,
    source_col: str = "start_node",
    target_col: str = "end_node",
    distance_col: str = "Distance",
    time_cols: Optional[List[str]] = None,
    directed: bool = True,
    num_edges: int = 10,
    min_count: int = 100,
    bins: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    save_output: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[nx.Graph]]:
    """
    Orchestrate the entire data processing and visualization pipeline.
    
    Args:
        file_path: Path to the CSV file
        source_col: Column name for source nodes
        target_col: Column name for target nodes
        distance_col: Column name for distance data
        time_cols: List of column names containing time data (inferred if None)
        directed: Whether to create a directed graph
        num_edges: Number of edges to plot in distributions
        min_count: Minimum number of time observations required
        bins: Number of histogram bins
        xlim: X-axis limits (min, max)
        save_output: Whether to save output figures and graph
        output_dir: Directory to save output files (created if doesn't exist)
        
    Returns:
        Tuple of (DataFrame, NetworkX graph)
    """
    print(f"Reading data from {file_path}...")
    df = read_data(file_path)
    if df is None:
        print("Failed to read data.")
        return None, None
    
    print(f"Found {len(df)} rows of data.")
    
    # Create output directory if needed
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Infer time columns if not provided
    if time_cols is None:
        time_cols = infer_time_columns(df, distance_col=distance_col)
        print(f"Inferred {len(time_cols)} time columns: {time_cols[:5]}...")
    
    print(f"Extracting edge data...")
    edge_data = get_edge_data(
        df,
        source_col=source_col,
        target_col=target_col,
        distance_col=distance_col,
        time_cols=time_cols,
        drop_zeros=True,
        drop_na=True,
    )
    
    print(f"Building graph with {len(edge_data)} edges...")
    G = build_graph_from_edge_data(edge_data, directed=directed)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Add position data to nodes
    G = add_pos_to_graph(G)
    
    # Calculate edge statistics
    G = calculate_edge_statistics(G)
    
    # Plot edge time distributions
    print(f"Plotting travel time distributions...")
    fig_dist = plot_edge_distributions_from_graph(G, num_edges=num_edges, min_count=min_count, bins=bins, xlim=xlim)
    if save_output and output_dir and fig_dist:
        fig_dist.savefig(os.path.join(output_dir, "time_distributions.png"), dpi=300, bbox_inches='tight')
    
    # Plot graph structure
    print(f"Plotting graph structure...")
    fig_graph = plot_graph(G, node_size=8, with_labels=False)
    if save_output and output_dir and fig_graph:
        fig_graph.savefig(os.path.join(output_dir, "graph_structure.png"), dpi=300, bbox_inches='tight')
    
    # Save graph to file
    if save_output and output_dir:
        graph_path = os.path.join(output_dir, "transportation_graph")
        save_graph_to_file(G, graph_path)
    
    plt.show()
    return df, G


def get_parameters_for_model(
    G: nx.Graph,
    filter_outliers: bool = True,
    outlier_threshold: float = 3.0,
    save_output: bool = False,
    output_path: Optional[str] = None
) -> Dict[Tuple[Any, Any], Dict[str, float]]:
    """
    Extract parameters for each edge that can be used for modeling.
    
    Args:
        G: NetworkX graph
        filter_outliers: Whether to filter time outliers
        outlier_threshold: Number of standard deviations to use for outlier detection
        save_output: Whether to save parameters to file
        output_path: Path to save parameters
        
    Returns:
        Dictionary mapping edges to their parameters
    """
    parameters = {}
    
    for u, v, data in G.edges(data=True):
        times = data.get('Time', [])
        if not times:
            continue
        times = np.asarray([float(t) for t in times if np.isfinite(t)], dtype=float)
        if times.size == 0:
            continue
        # Filter outliers if requested
        if filter_outliers and times.size > 5:
            mean = float(np.mean(times))
            std = float(np.std(times))
            if std > 0:
                z = np.abs((times - mean) / std)
                times = times[z < outlier_threshold]
        if times.size < 5:
            continue
        # Calculate parameters
        mean = float(np.mean(times))
        std = float(np.std(times))
        median = float(np.median(times))
        min_val = float(np.min(times))
        max_val = float(np.max(times))
        parameters[(u, v)] = {
            'mean': mean,
            'std': std,
            'median': median,
            'min': min_val,
            'max': max_val,
            'count': int(times.size),
            'distance': data.get('Distance', 1.0)
        }

    if save_output and output_path:
        with open(output_path, 'w') as f:
            f.write("source,target,mean,std,median,min,max,count,distance\n")
            for (u, v), params in parameters.items():
                f.write(f"{u},{v},{params['mean']},{params['std']},{params['median']},{params['min']},{params['max']},{params['count']},{params['distance']}\n")
    
    # Calculate the overall mean and average standard deviation
    all_means = []
    all_stds = []
    
    for (u, v), params in parameters.items():
        all_means.append(params['mean'])
        all_stds.append(params['std'])
    
    overall_mean = np.mean(all_means) if all_means else 0
    avg_std = np.mean(all_stds) if all_stds else 0
    
    params_summary = {
        "overall_mean": float(overall_mean),
        "avg_std": float(avg_std)
    }
    
    return parameters, params_summary


if __name__ == "__main__":
    # Define file paths
    path = "./Data/Chicago_1_filtered.csv"
    output_dir = "./Output"
    
    # Run the full pipeline
    df, G = orchestrate(
        file_path=path,
        source_col="start_node",
        target_col="end_node",
        distance_col="Distance",
        time_cols=None,  # Auto-infer time columns
        directed=True,
        num_edges=10,
        min_count=100,
        bins=30,
        xlim=(0, 5),
        save_output=True,
        output_dir=output_dir
    )
    
    if G is not None:

        # # Plot an enhanced version of the graph with a Chicago basemap
        # # showing directionality of the transportation network
        # fig_graph_enhanced = plot_graph(
        #     G, 
        #     node_size=10, 
        #     with_labels=False,
        #     label_size=6,
        #     figsize=(12, 10),
        #     title="Chicago Transportation Network",
        #     add_basemap=True,
        #     basemap_zoom=13,
        #     show_node_ids=True,
        #     show_arrows=True,  # Explicitly show arrows for directed edges
        #     arrow_size=10,     # Smaller arrows to avoid clutter
        #     bidirectional_color="blue",
        #     one_way_color="red"
        # )
        # if fig_graph_enhanced and output_dir:
        #     fig_graph_enhanced.savefig(os.path.join(output_dir, "chicago_network_with_map.png"), 
        #                               dpi=300, bbox_inches='tight')
        
        # # Also create a simplified directed graph view that highlights one-way streets
        # fig_directed_graph = plot_graph(
        #     G,
        #     node_size=8,
        #     with_labels=False,
        #     label_size=5,  # Even smaller labels for clarity
        #     figsize=(12, 10),
        #     title="Chicago One-Way Street Network",
        #     add_basemap=True,
        #     basemap_zoom=13,
        #     show_arrows=True,
        #     arrow_size=12,
        #     bidirectional_color="#cccccc",  # Light gray for bidirectional
        #     one_way_color="red"             # Highlight one-way streets
        # )
        # if fig_directed_graph and output_dir:
        #     fig_directed_graph.savefig(os.path.join(output_dir, "chicago_one_way_streets.png"),
        #                              dpi=300, bbox_inches='tight')
        
        # Get parameters for modeling
        parameters, params_summary = get_parameters_for_model(
            G, 
            filter_outliers=True,
            save_output=True,
            output_path=os.path.join(output_dir, "edge_parameters.csv")
        )
        print(params_summary)
        
        # # Create output directory if it doesn't exist
        # os.makedirs(output_dir, exist_ok=True)
        
        # Run simulation and training with proper error handling
        try:
            # Generate synthetic data
            simulation = random_data.data_simulation(
                n=1000, 
                input_params=params_summary, 
                dist=('gamma', 'gamma'), 
                corr=False
            )
            
            # Train the model
            model, history = train.train_model(
                simulation, 
                dist=('gamma', 'gamma'), 
                corr=False
            )

            #arcs = [G[(239, 217)], G[(217, 426)]]
            arcs = [G[(516, 22)], G[(22, 364)]]
            
            for i in arcs:
                print(i["mean_time"], i["std_time"], (i["start_latitude"], i["start_longitude"]), (i["end_latitude"], i["end_longitude"]))
            # Pasar 'gamma' como tipo de distribución para ajustar
            dict_edges, dict_sums = path_sum.sum_edges(arcs, model, dist_type='gamma')
            path_sum.plot_resulting_norm(dict_sums, arcs)
            Orchestator.test_orchestador(arcs)


        #     Plot training history
            train.plot_training(history, model_name="Gamma Distribution Model")
            
        #     # # Save the model
        #     # model.save(os.path.join(output_dir, "transport_model.h5"))
        #     # print(f"Model saved to {os.path.join(output_dir, 'transport_model.h5')}")
            
        #     # # Create and save visualization of model performance
        #     fig, axs, stats = random_data.create_boxplot(G, compare=True)
        #     #if fig:
        #     #     fig.savefig(os.path.join(output_dir, "time_boxplots.png"), dpi=300, bbox_inches='tight')
                
        except Exception as e:
            print(f"Error in simulation or training: {e}")