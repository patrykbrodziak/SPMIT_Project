from . genetic import VRPGeneticOptimizer


def set_vrp_hyper_parameters(n_points, n_agents):
    """Return heuristically set hyper parameter for VRP genetic optimizer"""
    population_size = max(n_agents * n_points // 10, 10 * n_points)
    optimizer = VRPGeneticOptimizer(
        population_size=population_size,
        constraint="sum",
        n_agents=n_agents,
        crossover_rate=1.0,
        mutation_rate=0.8,
        elitism_rate=0.05,
        extra_initialization_rate=1.0,
        crossover="cx",
        mutation="inv",
        crossover_schedule_type="decreasing_root",
        mutation_schedule_type="increasing_root",
    )

    return optimizer
