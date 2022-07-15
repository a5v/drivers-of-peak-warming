import pandas as pd
from SCC_functions import *
from peak_warming_calculator import peak_warming_calculator


def peak_warming_calculator_Pretis(r_minus_g=0.015, g_0=0.02, beta=0.001,
                                   gamma=2, D0=0.00236,
                                   P_50=120, s=0.05, r=0.03, P_100=300,
                                   start_year=2020, end_year=3000, last_SCC_year=2500,
                                   T_TCRE_1=0.00045, k_s=0.12,
                                   T_0=1.2, delta_T=3, alpha=0.02,
                                   size_of_perturbation=1,
                                   CO2_baseline=40,
                                   return_all_output=False):
    W_0 = 80

    peak_T_Nordhaus, output_df_Nordhaus = peak_warming_calculator(consumption_discount=g_0+r_minus_g,
                                                                  consumption_growth=g_0,
                                                                  gamma=gamma, D0=D0,
                                                                  P_50=P_50, s=s, r=r, P_100=P_100,
                                                                  start_year=start_year, end_year=end_year,
                                                                  last_SCC_year=last_SCC_year,
                                                                  T_TCRE_1=T_TCRE_1, k_s=k_s,
                                                                  T_0=T_0, delta_T=delta_T, alpha=alpha,
                                                                  size_of_perturbation=size_of_perturbation,
                                                                  CO2_baseline=CO2_baseline,
                                                                  return_all_output=True)
    P0_Nordhaus = output_df_Nordhaus['SCC'].loc[2020]
    print(f"P0_Nordhaus = {P0_Nordhaus}")

    years = create_years_array(start_year, end_year)
    years_of_perturbation = create_years_array(start_year, last_SCC_year)

    geometric_T = create_geometric_T(years, T_0=T_0, alpha=alpha, delta_T=delta_T)

    num_of_iterations = 100
    T_ts = geometric_T
    for iteration in range(num_of_iterations):

        SCC_ts_to_P_100 = []
        for SCC_year in range(len(years_of_perturbation)):
            T_perturbed_ts = create_geometric_T_perturbed(years, T=T_ts, SCC_year=years_of_perturbation[SCC_year],
                                                          T_TCRE=T_TCRE_1 * size_of_perturbation, k_s=k_s)
            g = g_0 - beta * (T_ts[:-1] ** 2 - T_0 ** 2)
            g_perturbed = g_0 - beta * (T_perturbed_ts[:-1] ** 2 - T_0 ** 2)

            consumption_discount = g[SCC_year] + r_minus_g
            discount_function = create_discount_function(years, years_of_perturbation[SCC_year],
                                                         consumption_discount=consumption_discount)

            W = create_total_consumption_Pretis(years, g, W_start_year=W_0)
            W_perturbed = create_total_consumption_Pretis(years, g_perturbed, W_start_year=W_0)

            time_series_data_Pretis = {'years': years, 'W': W, 'W perturbed': W_perturbed,
                                       'discount function': discount_function}
            time_series_df_Pretis = pd.DataFrame(data=time_series_data_Pretis).set_index('years')

            SCC = SCC_calculator_Pretis(time_series_df_Pretis, size_of_perturbation=size_of_perturbation)

            if SCC_year == 0:
                P0 = SCC
                # print(SCC)
            # SCC_list_actual.append(SCC)
            SCC_adjusted = SCC - (P0 - P0_Nordhaus)
            if SCC_adjusted < P_100:
                SCC_ts_to_P_100.append(SCC_adjusted)
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
            print_convergence_error(P_100, P_50, consumption_discount, g_0, r, s)

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
