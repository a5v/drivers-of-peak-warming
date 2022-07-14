import pandas as pd
from SCC_functions import *


def peak_warming_calculator(consumption_discount=0.035, consumption_growth=0.02,
                            gamma=2, D0=0.00236,
                            P_50=120, s=0.05, r=0.04, P_100=300,
                            start_year=2020, end_year=3000, last_SCC_year=2500,
                            T_TCRE_1=0.00045, k_s=0.12,
                            T_0=1.2, delta_T=3, alpha=0.02,
                            size_of_perturbation=1,
                            CO2_baseline=40,
                            return_all_output=False):

    years = create_years_array(start_year, end_year)
    years_of_perturbation = create_years_array(start_year, last_SCC_year)

    geometric_T = create_geometric_T(years, T_0=T_0, alpha=alpha, delta_T=delta_T)

    num_of_iterations = 100
    T_ts = geometric_T
    for iteration in range(num_of_iterations):

        SCC_ts_to_P_100 = []
        for SCC_year in range(len(years_of_perturbation)):
            W_ts = create_total_consumption(years, consumption_growth=consumption_growth)
            discount_function_ts = create_discount_function(years, SCC_year=years_of_perturbation[SCC_year],
                                                         consumption_discount=consumption_discount)
            T_perturbed_ts = create_geometric_T_perturbed(years, T=T_ts, SCC_year=years_of_perturbation[SCC_year],
                                                          T_TCRE=T_TCRE_1*size_of_perturbation, k_s=k_s)

            time_series_data = {'years': years, 'W': W_ts, 'discount function': discount_function_ts,
                                'T': T_ts, 'T perturbed': T_perturbed_ts}
            time_series_df = pd.DataFrame(data=time_series_data).set_index('years')

            SCC = SCC_calculator(time_series_df, size_of_perturbation=size_of_perturbation, gamma=gamma, D0=D0)

            if SCC < P_100:
                SCC_ts_to_P_100.append(SCC)
            else:
                SCC_ts_to_P_100.append(P_100)
                break
        check_SCC(P_100, SCC_ts_to_P_100)
        SCC_ts, P0 = forecast_SCC(SCC_ts_to_P_100, years)

        abatement_ts = abatement(P=SCC_ts, P0=P0, P_50=P_50, s=s, P_100=P_100, r=r)
        emissions_ts = abatement_to_emissions(abatement_ts, CO2_baseline)
        cumulative_emissions_array = calculate_cumulative_emissions(emissions_ts)
        temperature_change = T_TCRE_1 * cumulative_emissions_array
        temperature_change_plateau = temp_change_plateau(temperature_change)

        T_ts = T_0 + temperature_change_plateau

        if iteration == 0 or iteration == 1:
            peak_T = max(T_ts)
        else:
            previous_peak_T = peak_T
            peak_T = max(T_ts)
            if abs(peak_T - previous_peak_T) < 0.005:
                break

        if iteration == num_of_iterations - 1:
            print_convergence_error(P_100, P_50, consumption_discount, consumption_growth, r, s)

    if return_all_output:
        output_data = {'years': years, 'SCC': SCC_ts, 'abatement': abatement_ts,
                            'T': T_ts, 'emissions': emissions_ts}
        output_df = pd.DataFrame(data=output_data).set_index('years')
        return peak_T, output_df
        # return peak_T, SCC_ts, abatement_ts, emissions_ts, T_ts
    else:
        return peak_T


def print_convergence_error(P_100, P_50, consumption_discount, consumption_growth, r, s):
    print("convergence condition not achieved")
    print(f"{consumption_discount=}")
    print(f"{consumption_growth=}")
    print(f"{P_50=}")
    print(f"{s=}")
    print(f"{r=}")
    print(f"{P_100=}")


def check_SCC(P_100, SCC_ts_to_P_100):
    if SCC_ts_to_P_100[-1] < P_100:
        print("P_100 not achieved by achieved by final perturbed year")


def temp_change_plateau(temperature_change):
    temperature_change_plateau = np.array(temperature_change, copy=True)
    for i in range(len(temperature_change_plateau)):
        if i > np.argmax(temperature_change_plateau):
            temperature_change_plateau[i] = max(temperature_change_plateau)
    return temperature_change_plateau


def calculate_cumulative_emissions(emissions_ts):
    cumulative_emissions_array = np.append(np.zeros(1), np.cumsum(emissions_ts)[:-1])
    return cumulative_emissions_array


def forecast_SCC(SCC_ts_to_P_100, years_forecast):

    SCC_ts = []

    for i in range(len(years_forecast)):
        if i < len(SCC_ts_to_P_100):
            SCC_ts.append(SCC_ts_to_P_100[i])
        else:
            SCC_ts.append(SCC_ts_to_P_100[-1])

    SCC_ts = np.array(SCC_ts)

    P0 = SCC_ts[0]
    return SCC_ts, P0


def abatement(P, P0, P_50, r, s, P_100):
    if P0 >= P_50:
        print("P0 is greater than P_50")
    elif ((P_100 - P0) / (P_50 - P0)) ** (s / r) <= 2:
        print("MAC curve condition not satisfied")

    P_h = P0 + ((P_50 - P0) ** (-s / r) - 2 * (P_100 - P0) ** (-s / r)) ** (-r / s)

    Am = 1 + ((P_100 - P0) / (P_h - P0)) ** (-s / r)
    A = Am / (1 + ((P - P0) / (P_h - P0)) ** (-s / r))
    return A


def abatement_to_emissions(abatement_ts, CO2_baseline):
    CO2_emissions = CO2_baseline * (1 - abatement_ts)

    return CO2_emissions
