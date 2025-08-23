from Monte_Carlo import ind_simulator, corr_simulator


def main():
    # # Definición de distribuciones (mismo formato para ambos orquestadores)
    # distributions = [
    #     {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    #     # {'dist_name': 'normal', 'params': {'mu': 5, 'std': 1}},
    # ]

    # num_samples = 100000
    # iterations = 100000
    # T = 10

    # # Ejemplo 1: simulación independiente (como existía)
    # stats_ind = ind_simulator.Orchestrator(
    #     distributions=distributions,
    #     num_samples=num_samples,
    #     iterations=iterations,
    #     T=T,
    # )
    # print("Estadísticas (suma) - Independiente:", stats_ind)

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
 main()
