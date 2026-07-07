"""Main driver for Beluga Whale Optimization voltage clamp experiments."""

import pickle
from pathlib import Path

import bwo_vc_optimization
import ga_configs


WITH_ARTEFACT = True

VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
    window=2,
    step_size=2,
    steps_in_protocol=4,
    step_duration_bounds=(5, 1000),
    step_voltage_bounds=(-120, 60),
    target_current='',
    population_size=200,
    max_generations=51,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.1,
    tournament_size=2,
    step_types=['step', 'ramp'],
    with_artefact=WITH_ARTEFACT,
    model_name='Kernik')

LIST_OF_CURRENTS = ['I_Na', 'I_Kr', 'I_Ks', 'I_To', 'I_F', 'I_CaL', 'K1']


def main():
    """Run voltage clamp protocol optimization with BWO."""
    vco_dir_name = (
        f'bwo_trial_steps_ramps_{VCO_CONFIG.model_name}_'
        f'{VCO_CONFIG.population_size}_{VCO_CONFIG.max_generations}_'
        f'{VCO_CONFIG.steps_in_protocol}_{VCO_CONFIG.step_voltage_bounds[0]}_'
        f'{VCO_CONFIG.step_voltage_bounds[1]}'
    )

    output_dir = Path(__file__).resolve().parent / 'bwo_results' / vco_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for current in LIST_OF_CURRENTS:
        output_path = output_dir / (
            f'bwo_results_{current}_artefact_{WITH_ARTEFACT}')
        print(
            f'Finding best BWO protocol for {current}. '
            f'Writing protocol to: {output_path}')
        VCO_CONFIG.target_current = current
        result = bwo_vc_optimization.start_bwo(VCO_CONFIG)

        with output_path.open('wb') as output_file:
            pickle.dump(result, output_file)


if __name__ == '__main__':
    main()
