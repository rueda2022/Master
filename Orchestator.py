from Monte_Carlo import ind_simulator, corr_simulator


def test_orchestador(arcs):
    # # Definición de distribuciones (mismo formato para ambos orquestadores)
    distributions = []
    for arc in arcs:
        # Convertir de parámetros normales a gamma (forma alpha y escala beta)
        mean = arc['mean_time']
        std = arc['std_time']
        var = std**2
        alpha = (mean**2) / var if var > 0 else 1.0  # Parámetro de forma
        beta = mean / var if var > 0 else 1.0        # Parámetro de escala
        distributions.append({'dist_name': 'gamma', 'params': {'alpha': alpha, 'beta': beta}})
    num_samples = 10000
    iterations = 10000
    T = 10

    # Ejemplo 1: simulación independiente (como existía)
    stats_ind = ind_simulator.Orchestrator(
        distributions=distributions,
        num_samples=num_samples,
        iterations=iterations,
        T=T,
    )
    print("Estadísticas (suma) - Independiente:", stats_ind)

    # # Ejemplo 2: simulación correlacionada con cópula (misma firma de llamada)
    # stats_corr = corr_simulator.Orchestrator(
    #     distributions=distributions,
    #     num_samples=num_samples,
    #     iterations=iterations,
    #     T=T,
    #     rho=0.4,           # correlación entre vecinos
    #     copula='gaussian', # o 't'
    #     nu=5,               # usado solo si copula='t'
    #     seed=12345,
    #     plot=True,
    # )
    # print("Estadísticas (suma) - Correlacionada:", stats_corr)


# if __name__ == "__main__":
#     test_orchestador()
