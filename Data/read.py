from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import networkx as nx
from Training import random_data


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
    try:
        df = pd.read_csv(file_path, comment="/")
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception:
        return None


def infer_time_columns(data: pd.DataFrame, distance_col: str = "Distance") -> List[str]:
    cols = list(map(str, data.columns))
    norm = [c.strip().lower() for c in cols]
    target = distance_col.strip().lower()
    matches = [i for i, n in enumerate(norm) if (n == target) or n.startswith(target) or ("Distance" in n)]
    if not matches:
        raise ValueError("No se encontró la columna 'Distance'.")
    dist_idx = matches[0]
    return cols[dist_idx + 1 :]


def get_edge_data(
    data: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    distance_col: str = "Distance",
    time_cols: Optional[List[str]] = None,
    drop_zeros: bool = True,
    drop_na: bool = True,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    if source_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Missing {source_col} or {target_col}")
    if time_cols is None:
        time_cols = infer_time_columns(data, distance_col=distance_col)

    edge_data: Dict[Tuple[str, str], Dict[str, object]] = {}

    for _, row in data.iterrows():
        u = row[source_col]
        v = row[target_col]
        d_raw = pd.to_numeric(pd.Series([row.get(distance_col, np.nan)]), errors="coerce").iloc[0]
        d_val = None if pd.isna(d_raw) else float(d_raw)

        series = row[time_cols]
        series = series.replace([np.inf, -np.inf], np.nan)
        numeric = pd.to_numeric(series, errors="coerce")
        mask = numeric.isna()
        td_seconds = pd.Series(index=series.index, dtype="float64")
        if mask.any():
            td = pd.to_timedelta(series[mask], errors="coerce")
            td_seconds.loc[mask] = td.dt.total_seconds().astype("float64")
        seconds = numeric.astype("float64")
        seconds.loc[mask] = td_seconds.loc[mask]
        vals = seconds.to_numpy(dtype=float)

        if drop_na:
            vals = vals[~np.isnan(vals)]
        if drop_zeros:
            vals = vals[vals != 0.0]

        key = (u, v)
        # Only keep Distance and Time on the edge; no Source/Target
        if key not in edge_data:
            edge_data[key] = {"Distance": d_val, "Time": []}
        if edge_data[key]["Distance"] is None and d_val is not None:
            edge_data[key]["Distance"] = float(d_val)  # ensure Python float
        if vals.size:
            # Ensure Python floats (avoid np.float64)
            edge_data[key]["Time"].extend([float(x) for x in vals.tolist()])

    return edge_data


def build_graph_from_edge_data(
    edge_data: Dict[Tuple[str, str], Dict[str, object]],
    directed: bool = True,
) -> nx.Graph:
    # Use custom graphs that support G[(u, v)]
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
) -> None:
    sns.set(style="whitegrid")
    items = [((u, v), data.get("Time", [])) for u, v, data in G.edges(data=True)]
    items = [iv for iv in items if isinstance(iv[1], list) and len(iv[1]) >= min_count]
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    items = items[:num_edges]
    if not items:
        return

    cols = min(5, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, ((u, v), vals) in zip(axes, items):
        ax.hist(vals, bins=bins, color="blue", alpha=0.7)
        ax.set_title(f"{u} -> {v} (n={len(vals)})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(*xlim)

    for ax in axes[len(items) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_graph(G: nx.Graph) -> None:
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="#1f77b4")
    nx.draw_networkx_edges(G, pos, edge_color="#999999", width=0.8, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6, alpha=0.8)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def orchestrate(
    file_path: str,
    source_col: str = "source",
    target_col: str = "target",
    time_cols: Optional[List[str]] = None,
    directed: bool = True,
    num_edges: int = 10,
    min_count: int = 100,
    bins: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
):
    df = read_data(file_path)
    if df is None:
        return None, None
    edge_data = get_edge_data(
        df,
        source_col=source_col,
        target_col=target_col,
        distance_col="Distance",
        time_cols=time_cols,
        drop_zeros=True,
        drop_na=True,
    )
    G = build_graph_from_edge_data(edge_data, directed=directed)
    plot_edge_distributions_from_graph(G, num_edges=num_edges, min_count=min_count, bins=bins, xlim=xlim)
    plot_graph(G)
    return df, G


if __name__ == "__main__":
    path = "./Data/Chicago_1_filtered.csv"
    df, G = orchestrate(
        file_path=path,
        source_col="start_node",
        target_col="end_node",
        time_cols=None,
        directed=True,
        num_edges=10,
        min_count=100,
        bins=30,
        xlim=(0, 5),
    )

    # Access edge attributes directly with G[(u, v)]
    print((G[(43, 42)]))
    parameters = random_data.get_parameters(G)
    # Default behavior now removes time outliers
    

    # To see the comparison between original and filtered time data
    fig, axs, stats = random_data.create_boxplot(G, compare=True)

    # Get parameters with time outliers filtered
    params = random_data.get_parameters(G, filter_outliers_data=True)
    print(parameters)