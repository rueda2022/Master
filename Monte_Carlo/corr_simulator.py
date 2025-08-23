"""
Simulador de correlaciones mediante cópulas (Gaussiana o t-Student).

- Define n variables con marginales específicas (normal o gamma).
- Muestra N observaciones independientes.
- Impone correlación entre vecinos (v_i con v_{i-1} y v_{i+1}) con ρ dado,
  construyendo una cópula en el espacio latente (gaussiano o t).
- Transforma de regreso a las marginales originales.
- Muestra la matriz de correlación final y compara histogramas antes/después.

Requisitos: numpy, scipy.stats, matplotlib, seaborn
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def build_distributions(specs):
    """
    Crea objetos de distribución de scipy.stats a partir de especificaciones.
    Cada especificación puede ser:
      - ("normal", {"loc": 0, "scale": 1})
      - ("gamma", {"k": 2, "theta": 2})  # k=shape, theta=scale
    """
    dists = []
    for kind, params in specs:
        if kind.lower() in ("normal", "norm"):
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
            dists.append(stats.norm(loc=loc, scale=scale))
        elif kind.lower() in ("gamma",):
            # scipy.stats.gamma usa 'a' como shape y 'scale' como theta
            k = params.get("k", params.get("a", 2.0))
            theta = params.get("theta", params.get("scale", 1.0))
            dists.append(stats.gamma(a=k, scale=theta))
        else:
            raise ValueError(f"Distribución no soportada: {kind}")
    return dists


def sample_independent(dists, N, rng):
    """
    Muestra N valores independientes por cada distribución.
    Retorna array (N, n).
    """
    samples = [dist.rvs(size=N, random_state=rng) for dist in dists]
    return np.column_stack(samples)


def to_uniforms(X, dists, eps=1e-12):
    """
    Transforma columnas de X a U[0,1] usando CDFs marginales.
    Hace clipping para evitar 0 y 1 exactos.
    """
    U = np.zeros_like(X, dtype=float)
    for j, dist in enumerate(dists):
        u = dist.cdf(X[:, j])
        u = np.clip(u, eps, 1 - eps)
        U[:, j] = u
    return U


def chain_correlation_matrix(n, rho):
    """
    Matriz de correlación tridiagonal con 1 en diagonal y rho en vecinos.
    """
    R = np.eye(n)
    off = np.full(n - 1, rho, dtype=float)
    R[np.arange(n - 1), np.arange(1, n)] = off
    R[np.arange(1, n), np.arange(n - 1)] = off
    return R


def make_spd(R, max_tries=10, jitter0=1e-10):
    """
    Asegura que R sea definida positiva agregando jitter en la diagonal si es necesario.
    """
    jitter = jitter0
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(R)
            return R
        except np.linalg.LinAlgError:
            R = R.copy()
            R[np.diag_indices_from(R)] += jitter
            jitter *= 10
    raise np.linalg.LinAlgError("No se pudo hacer SPD la matriz de correlación.")


def gaussian_copula_correlate(U, R, rng):
    """
    Aplica dependencia con cópula Gaussiana:
      Z = Phi^-1(U), Z_corr = Z * L^T, U' = Phi(Z_corr)
    """
    # normal latente
    Z = stats.norm.ppf(U)
    # Cholesky de la correlación objetivo
    L = np.linalg.cholesky(R)
    Z_corr = Z @ L.T
    U_corr = stats.norm.cdf(Z_corr)
    # Clipping defensivo
    eps = 1e-12
    U_corr = np.clip(U_corr, eps, 1 - eps)
    return U_corr


def t_copula_correlate(U, R, nu, rng):
    """
    Aplica dependencia con cópula t-Student con ν grados de libertad:
      Z = Phi^-1(U), Z_corr = Z * L^T
      T = Z_corr / sqrt(S/nu), con S ~ Chi2(nu) iid por muestra
      U' = CDF_tν(T)
    """
    Z = stats.norm.ppf(U)
    L = np.linalg.cholesky(R)
    Z_corr = Z @ L.T
    # Escalamiento t: una escala por fila (muestra)
    S = stats.chi2.rvs(df=nu, size=Z_corr.shape[0], random_state=rng)
    scale = np.sqrt(S / nu)  # shape (N,)
    T = Z_corr / scale[:, None]
    U_corr = stats.t.cdf(T, df=nu)
    eps = 1e-12
    U_corr = np.clip(U_corr, eps, 1 - eps)
    return U_corr


def from_uniforms(U, dists):
    """
    Transforma uniformes U a las marginales objetivo usando PPFs.
    """
    Y = np.zeros_like(U, dtype=float)
    for j, dist in enumerate(dists):
        Y[:, j] = dist.ppf(U[:, j])
    return Y


def plot_results(X, Y, corr_after, titles=None):
    """
    - Histogramas superpuestos antes (X) y después (Y) por variable.
    - Heatmap de correlaciones finales.
    """
    n = X.shape[1]
    sns.set(style="whitegrid", context="talk")

    # Histogramas por variable
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.ravel()
    for j in range(n):
        ax = axes[j]
        ax.hist(X[:, j], bins=40, density=True, alpha=0.5, label="Antes", color="#1f77b4")
        ax.hist(Y[:, j], bins=40, density=True, alpha=0.5, label="Después", color="#ff7f0e")
        name = f"v{j+1}" if titles is None else titles[j]
        ax.set_title(f"Histograma {name}")
        ax.legend()
    # Ocultar axes vacíos
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Comparación de marginales antes vs después", y=1.02)
    fig.tight_layout()

    # Heatmap de correlación final
    plt.figure(figsize=(6 + n * 0.2, 5 + n * 0.2))
    sns.heatmap(corr_after, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Matriz de correlación (después)")
    plt.tight_layout()
    plt.show()


def _convert_distributions(distributions):
    """
    Convierte lista de dicts estilo ind_simulator a specs usadas en este módulo.
    [{'dist_name': 'normal', 'params': {'mu': 0, 'std': 1}},
     {'dist_name': 'gamma', 'params': {'shape': 2, 'scale': 1}}]
      -> [("normal", {"loc": 0, "scale": 1}), ("gamma", {"k": 2, "theta": 1})]
    """
    specs = []
    for d in distributions:
        kind = d.get('dist_name', '').lower()
        params = d.get('params', {})
        if kind in ('normal', 'norm'):
            mu = params.get('mu', params.get('loc', 0.0))
            std = params.get('std', params.get('scale', 1.0))
            specs.append(("normal", {"loc": mu, "scale": std}))
        elif kind == 'gamma':
            k = params.get('shape', params.get('k', 2.0))
            theta = params.get('scale', params.get('theta', 1.0))
            specs.append(("gamma", {"k": k, "theta": theta}))
        else:
            raise ValueError(f"Distribución no soportada: {kind}")
    return specs


def _get_summary_statistics(data, quantiles=(0.25, 0.5, 0.75), value=None):
    import numpy as _np
    arr = _np.array(data)
    summary = {
        'mean': float(_np.mean(arr)),
        'std': float(_np.std(arr)),
        'quantiles': _np.quantile(arr, quantiles).tolist()
    }
    if value is not None:
        summary['probability_less_than'] = float(_np.mean(arr < value))
    return summary


def _plot_sum_distribution(data, title='Resultados (suma de variables)', statistics=None):
    import numpy as _np
    import seaborn as _sns
    import matplotlib.pyplot as _plt
    arr = _np.array(data).ravel()
    _sns.set_theme(style="whitegrid")
    _plt.figure(figsize=(12, 6))
    _sns.histplot(arr, bins=50, kde=True, color='royalblue', alpha=0.7, edgecolor='none')
    if statistics:
        mean_val = statistics['mean']
        med = _np.median(arr)
        std_val = statistics['std']
        q1, q2, q3 = statistics['quantiles']
        _plt.axvline(mean_val, color='crimson', linestyle='--', linewidth=2, label=f"Media: {mean_val:.2f}")
        _plt.axvline(med, color='forestgreen', linestyle='-', linewidth=2, label=f"Mediana: {med:.2f}")
        txt = (f"Media: {mean_val:.2f}\nDesv. Est.: {std_val:.2f}\nQ1: {q1:.2f}\nQ2: {q2:.2f}\nQ3: {q3:.2f}")
        if 'probability_less_than' in statistics:
            txt += f"\nP(Suma < T): {statistics['probability_less_than']:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        _plt.text(0.02, 0.98, txt, transform=_plt.gca().transAxes, va='top', bbox=props)
    _plt.title(title)
    _plt.xlabel('Valor')
    _plt.ylabel('Frecuencia')
    _plt.legend(loc='upper right')
    _plt.tight_layout()
    _plt.show()


def Orchestrator(distributions=[], num_samples=10000, iterations=10000, T=None,
                 rho=0.3, copula='gaussian', nu=5, seed=12345, plot=True):
    """
    Orquestador de simulación correlacionada con cópulas, usando el mismo estilo de
    entrada que ind_simulator.Orchestrator.

    - distributions: lista de dicts {'dist_name': 'normal'|'gamma', 'params': {...}}
      normal: {'mu', 'std}; gamma: {'shape', 'scale'}.
    - num_samples: tamaño de la muestra base (N) para construir la cópula.
    - iterations: número de iteraciones a muestrear de la muestra correlacionada para
      construir la distribución de la suma (similar a ind_simulator).
    - T: umbral para calcular P(Suma < T).
    - rho: correlación entre vecinos en la matriz tridiagonal.
    - copula: 'gaussian' o 't'. Si 't', usar parámetro nu.
    - seed: semilla aleatoria.
    - plot: si True, grafica histogramas de variables y de la suma.

    Retorna un dict con estadísticas de la suma. Además, si plot=True, muestra
    gráficos de marginales y heatmap de correlaciones.
    """
    rng = np.random.default_rng(seed)

    # 1) Preparar marginales
    specs = _convert_distributions(distributions)
    dists = build_distributions(specs)
    n = len(dists)

    # 2) Muestras independientes y paso a uniformes
    X = sample_independent(dists, num_samples, rng=rng)
    U = to_uniforms(X, dists)

    # 3) Matriz de correlación y cópula
    R = chain_correlation_matrix(n, rho)
    R = make_spd(R)
    if copula.lower() in ("gaussian", "normal"):
        U_corr = gaussian_copula_correlate(U, R, rng=rng)
    elif copula.lower() in ("t", "student", "t-student"):
        U_corr = t_copula_correlate(U, R, nu=nu, rng=rng)
    else:
        raise ValueError(f"Cópula no soportada: {copula}")

    # 4) Volver a marginales
    Y = from_uniforms(U_corr, dists)

    # 5) Estadísticas y suma para 'iterations' iteraciones
    # Seleccionar 'iterations' filas aleatorias y sumar por fila
    idx = rng.integers(low=0, high=num_samples, size=iterations)
    sums = np.sum(Y[idx, :], axis=1)

    # 6) Estadísticas de salida
    stats_sum = _get_summary_statistics(sums, value=T)

    # 7) Plots
    if plot:
        corr_after = np.corrcoef(Y, rowvar=False)
        titles = [f"v{i+1}" for i in range(n)]
        plot_results(X, Y, corr_after, titles=titles)
        _plot_sum_distribution(sums, title='Distribución de la suma (cópula) ', statistics=stats_sum)

    return stats_sum


def main():
    rng = np.random.default_rng(12345)  # Semilla para reproducibilidad

    # Parámetros
    n = 5
    N = 10_000
    rho = 0.6  # correlación entre vecinos
    copula = "gaussian"  # "gaussian" o "t"
    nu = 5  # grados de libertad si copula == "t"

    # Definición de marginales (ejemplo: alternando normal y gamma)
    # v1 ~ normal(0,1), v2 ~ gamma(k=2, θ=2), etc.
    specs = [
        ("normal", {"loc": 0, "scale": 1}),
        ("gamma", {"k": 2, "theta": 2}),
        ("normal", {"loc": 1, "scale": 2}),
        ("gamma", {"k": 3, "theta": 1}),
        ("normal", {"loc": -1, "scale": 1.5}),
    ]
    # Si n != len(specs), extender/cortar de forma cíclica para el demo
    if len(specs) != n:
        base = specs.copy()
        specs = [base[i % len(base)] for i in range(n)]

    dists = build_distributions(specs)

    # 1) Muestreo independiente
    X = sample_independent(dists, N, rng=rng)

    # 2) Transformar a uniformes U (copula trick)
    U = to_uniforms(X, dists)

    # 3) Construir matriz de correlación tridiagonal
    R = chain_correlation_matrix(n, rho)
    R = make_spd(R)  # asegurar SPD para Cholesky

    # 4) Aplicar dependencia en el espacio latente
    if copula.lower() in ("gaussian", "normal"):
        U_corr = gaussian_copula_correlate(U, R, rng=rng)
    elif copula.lower() in ("t", "student", "t-student"):
        U_corr = t_copula_correlate(U, R, nu=nu, rng=rng)
    else:
        raise ValueError(f"Cópula no soportada: {copula}")

    # 5) Transformar de regreso a marginales originales
    Y = from_uniforms(U_corr, dists)

    # 6) Correlaciones antes y después (Pearson)
    corr_before = np.corrcoef(X, rowvar=False)
    corr_after = np.corrcoef(Y, rowvar=False)

    # Mostrar en consola
    np.set_printoptions(precision=3, suppress=True)
    print("Matriz de correlación (antes, independiente):")
    print(corr_before)
    print("\nMatriz de correlación (después de cópula):")
    print(corr_after)
    print("\nNota: la correlación empírica y la de Pearson tras transformar "
          "marginales puede diferir de ρ y de R (espacio latente), especialmente con N finito y no-normalidad.")

    # 7) Visualizaciones
    titles = [f"v{i+1}" for i in range(n)]
    plot_results(X, Y, corr_after, titles=titles)


if __name__ == "__main__":
    main()
