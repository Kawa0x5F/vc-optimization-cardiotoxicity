import math
import random

import numpy as np

import ga_genetic_algorithm_results as genetic_algorithm_results
import mod_protocols as protocols


DIMENSIONS_PER_STEP = 4


def run_bwo(vco_config):
    """Run Beluga Whale Optimization for a voltage clamp protocol."""
    if vco_config.population_size < 2:
        raise ValueError('BWO requires at least two whales in the population.')

    current = vco_config.target_current
    dimension = vco_config.steps_in_protocol * DIMENSIONS_PER_STEP
    lower_bound, upper_bound = get_bounds(vco_config)

    population = np.asarray([
        init_whale(vco_config)
        for _ in range(vco_config.population_size)
    ])
    fitness = np.asarray([
        evaluate_whale(position, vco_config)
        for position in population
    ])

    population, fitness = sort_population(population, fitness)
    best_position = population[0].copy()
    best_fitness = fitness[0]

    final_population = [
        make_generation_records(population, fitness, vco_config)
    ]

    for generation in range(1, vco_config.max_generations):
        print(f'\tBWO generation {generation} for {current}')

        new_population = population.copy()
        whale_fall_probability = (
            0.1 - 0.05 * (generation / vco_config.max_generations)
        )
        balance_factors = np.asarray([
            random.random()
            * (1 - 0.5 * generation / vco_config.max_generations)
            for _ in range(vco_config.population_size)
        ])

        for i in range(vco_config.population_size):
            random_whale_index = get_random_whale_index(
                i, vco_config.population_size)

            if balance_factors[i] > 0.5:
                new_population[i] = explore(
                    population=population,
                    whale_index=i,
                    random_whale_index=random_whale_index,
                    dimension=dimension)
            else:
                new_population[i] = exploit(
                    population=population,
                    best_position=best_position,
                    whale_index=i,
                    random_whale_index=random_whale_index,
                    dimension=dimension,
                    generation=generation,
                    max_generations=vco_config.max_generations)

            new_population[i] = apply_bounds(
                new_population[i], lower_bound, upper_bound)
            new_fitness = evaluate_whale(new_population[i], vco_config)

            if new_fitness < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness

        for i in range(vco_config.population_size):
            if balance_factors[i] <= whale_fall_probability:
                random_whale_index = get_random_whale_index(
                    i, vco_config.population_size)
                fallen_position = whale_fall(
                    population=population,
                    whale_index=i,
                    random_whale_index=random_whale_index,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    whale_fall_probability=whale_fall_probability,
                    generation=generation,
                    max_generations=vco_config.max_generations)

                fallen_position = apply_bounds(
                    fallen_position, lower_bound, upper_bound)
                fallen_fitness = evaluate_whale(fallen_position, vco_config)

                if fallen_fitness < fitness[i]:
                    population[i] = fallen_position
                    fitness[i] = fallen_fitness

        population, fitness = sort_population(population, fitness)

        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_position = population[0].copy()

        final_population.append(
            make_generation_records(population, fitness, vco_config))
        generate_statistics(fitness)

    return genetic_algorithm_results.GAResultVoltageClampOptimization(
        vco_config,
        current=current,
        generations=final_population)


def init_whale(config):
    """Initialize a whale position as a voltage-clamp protocol vector."""
    position = np.zeros(config.steps_in_protocol * DIMENSIONS_PER_STEP)
    voltage_min, voltage_max = config.step_voltage_bounds
    duration_min, duration_max = config.step_duration_bounds

    for i in range(config.steps_in_protocol):
        offset = i * DIMENSIONS_PER_STEP
        position[offset] = random.choice([0.0, 1.0])
        position[offset + 1] = np.random.uniform(voltage_min, voltage_max)
        position[offset + 2] = np.random.uniform(voltage_min, voltage_max)
        position[offset + 3] = np.random.uniform(duration_min, duration_max)

    return position


def whale_to_protocol(position, config):
    """Convert a BWO position vector into a VoltageClampProtocol."""
    steps = []

    for i in range(config.steps_in_protocol):
        offset = i * DIMENSIONS_PER_STEP
        step_type = position[offset]
        voltage_start = position[offset + 1]
        voltage_end = position[offset + 2]
        duration = position[offset + 3]

        if step_type < 0.5:
            steps.append(protocols.VoltageClampStep(
                voltage=voltage_start,
                duration=duration))
        else:
            steps.append(protocols.VoltageClampRamp(
                voltage_start=voltage_start,
                voltage_end=voltage_end,
                duration=duration))

    return protocols.VoltageClampProtocol(steps=steps)


def evaluate_whale(position, config):
    """Evaluate a whale position. BWO minimizes, so contribution is negated."""
    try:
        protocol = whale_to_protocol(position, config)
        individual = genetic_algorithm_results.VCOptimizationIndividual(
            protocol=protocol,
            fitness=0)
        max_contributions = individual.evaluate(config=config, prestep=5000)
        contribution = max_contributions.loc[
            max_contributions['Current'] == config.target_current
        ]['Contribution'].values[0]
    except Exception:
        return 0.0

    return -contribution


def get_bounds(config):
    """Create lower and upper bounds for every BWO vector dimension."""
    lower_bound = []
    upper_bound = []
    voltage_min, voltage_max = config.step_voltage_bounds
    duration_min, duration_max = config.step_duration_bounds

    for _ in range(config.steps_in_protocol):
        lower_bound.extend([0.0, voltage_min, voltage_min, duration_min])
        upper_bound.extend([1.0, voltage_max, voltage_max, duration_max])

    return np.asarray(lower_bound), np.asarray(upper_bound)


def apply_bounds(position, lower_bound, upper_bound):
    return np.clip(position, lower_bound, upper_bound)


def levy_flight(dimension):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = 0.01 * np.random.randn(dimension) * sigma
    v = np.random.randn(dimension)

    return u / np.power(np.abs(v), 1 / beta)


def explore(population, whale_index, random_whale_index, dimension):
    new_position = population[whale_index].copy()
    random_r1 = random.random()
    random_r2 = random.random()

    indices = np.arange(dimension)
    np.random.shuffle(indices)

    if dimension <= population.shape[0] / 5:
        if dimension >= 2:
            first_dimension = indices[0]
            second_dimension = indices[1]
            movement = (
                population[random_whale_index, first_dimension]
                - population[whale_index, second_dimension]
            ) * (random_r1 + 1)

            new_position[first_dimension] = (
                population[whale_index, first_dimension]
                + movement * math.sin(random_r2 * 2 * math.pi)
            )
            new_position[second_dimension] = (
                population[whale_index, second_dimension]
                + movement * math.cos(random_r2 * 2 * math.pi)
            )
    else:
        for j in range(dimension // 2):
            first_dimension = indices[2 * j]
            second_dimension = indices[2 * j + 1]

            new_position[first_dimension] = (
                population[whale_index, first_dimension]
                + (
                    population[random_whale_index, first_dimension]
                    - population[whale_index, first_dimension]
                )
                * (random_r1 + 1)
                * math.sin(random_r2 * 2 * math.pi)
            )
            new_position[second_dimension] = (
                population[whale_index, second_dimension]
                + (
                    population[random_whale_index, first_dimension]
                    - population[whale_index, second_dimension]
                )
                * (random_r1 + 1)
                * math.cos(random_r2 * 2 * math.pi)
            )

    return new_position


def exploit(population, best_position, whale_index, random_whale_index,
            dimension, generation, max_generations):
    random_r3 = random.random()
    random_r4 = random.random()
    c1 = 2 * random_r4 * (1 - generation / max_generations)

    return (
        random_r3 * best_position
        - random_r4 * population[whale_index]
        + c1
        * levy_flight(dimension)
        * (population[random_whale_index] - population[whale_index])
    )


def whale_fall(population, whale_index, random_whale_index, lower_bound,
               upper_bound, whale_fall_probability, generation,
               max_generations):
    random_r5 = random.random()
    random_r6 = random.random()
    random_r7 = random.random()
    c2 = 2 * population.shape[0] * whale_fall_probability
    step_size = (
        random_r7
        * (upper_bound - lower_bound)
        * math.exp(-c2 * generation / max_generations)
    )

    return (
        random_r5
        * (population[whale_index] - random_r6 * population[random_whale_index])
        + step_size
    )


def get_random_whale_index(whale_index, population_size):
    random_whale_index = random.randrange(population_size)
    while random_whale_index == whale_index:
        random_whale_index = random.randrange(population_size)

    return random_whale_index


def sort_population(population, fitness):
    sorted_indices = np.argsort(fitness)
    return population[sorted_indices], fitness[sorted_indices]


def make_generation_records(population, fitness, config):
    records = []

    for position, bwo_fitness in zip(population, fitness):
        records.append(genetic_algorithm_results.VCOptimizationIndividual(
            protocol=whale_to_protocol(position, config),
            fitness=-bwo_fitness))

    return records


def generate_statistics(fitness):
    contribution_fitness = -fitness

    print('\t\tMin fitness: {}'.format(np.min(contribution_fitness)))
    print('\t\tMax fitness: {}'.format(np.max(contribution_fitness)))
    print('\t\tAverage fitness: {}'.format(np.mean(contribution_fitness)))
    print('\t\tStandard deviation: {}'.format(np.std(contribution_fitness)))


def start_bwo(vco_config):
    return run_bwo(vco_config)
