import numpy as np

## calculate emissions from forecasted abatement and CO2 baseline
def abatement_to_emissions(forecasted_abatement, CO2_baseline):
    CO2_emissions = CO2_baseline * (1 - forecasted_abatement)

    return CO2_emissions

def calculate_cumulative_emissions(emissions_ts):
    cumulative_emissions_array = np.append(np.zeros(1), np.cumsum(emissions_ts)[:-1])
    return cumulative_emissions_array