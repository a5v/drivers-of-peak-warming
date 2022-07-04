import pandas as pd
import numpy as np

from SCC_functions import *


def peak_warming_calculator(consumption_discount=0.035, consumption_growth=0.02,
                            gamma=2, D0=0.00267,
                            P_50=100, s=0.05, r=0.04, P_100=300,
                            start_year=2020, end_year=3000, last_SCC_year=2500,
                            T_TCRE_1=0.00054, k_s=0.12,
                            T_0=1.2, delta_T=3, alpha=0.02,
                            size_of_perturbation=1,
                            CO2_baseline  = 40.339401131162724,
                            return_all_output=False):
    # last_historical_year = 2019
    # first_SCC_year = last_historical_year

    years = create_years_array(start_year, end_year)
    years_of_perturbation = create_years_array(start_year, last_SCC_year)
    # years_forecast = create_years_array(first_SCC_year, end_year)
    # years_forecast_length = len(years_forecast)

    # T_0, T_historical = read_historical_T()

    # T_complete_initial, T_forecast_initial = create_T_initial(T_0, T_historical, alpha, delta_T,
    #                                                             years_forecast_length)

    geometric_T = create_geometric_T(years, T_0=T_0, alpha=alpha, delta_T=delta_T)

    num_of_iterations = 100
    # T_forecast_iteration = T_forecast_initial
    T_iteration = geometric_T
    for iteration in range(num_of_iterations):

        SCC_list = []
        for SCC_year in range(len(years_of_perturbation)):
            ## define perturbation
            # T_perturbed = create_T_perturbed(T_TCRE, T_forecast_iteration, years_forecast_length, years_forecast,
            #                                  T_historical, k_s, SCC_year, years_of_perturbation)
            W = create_total_consumption(years)
            ## define discount function
            discount_function = create_discount_function(years, SCC_year=years_of_perturbation[SCC_year],
                                                         consumption_discount=consumption_discount)

            # cost = cost_of_perturbation(T_iteration, T_perturbed, W, discount_function, gamma, D0)
            # SCC = cost / (10 ** 9)

            T_perturbed = create_geometric_T_perturbed(years, T=T_iteration, SCC_year=years_of_perturbation[SCC_year],
                                                       T_TCRE=T_TCRE_1 * size_of_perturbation, k_s=k_s)

            time_series_data = {'years': years, 'W': W, 'discount function': discount_function,
                                'T': T_iteration, 'T perturbed': T_perturbed}

            time_series_df = pd.DataFrame(data=time_series_data).set_index('years')

            SCC = SCC_calculator(time_series_df, size_of_perturbation=size_of_perturbation, gamma=gamma, D0=D0)

            if SCC < P_100:
                SCC_list.append(SCC)
            else:
                SCC_list.append(P_100)
                break
        SCC_calculated = np.asarray(SCC_list)
        # print(SCC_calculated)

        check_SCC_calculated(P_100, SCC_calculated)

        SCC_forecasted, P0 = forecast_SCC(SCC_calculated, years)

        forecasted_abatement = abatement(P=SCC_forecasted, P0=P0, P_50=P_50, s=s, P_100=P_100, r=r)
        forecasted_emissions = abatement_to_emissions(forecasted_abatement, CO2_baseline)
        cumulative_emissions_array = calculate_cumulative_emissions(forecasted_emissions)
        temperature_change = T_TCRE_1 * cumulative_emissions_array
        temperature_change_plateau = temp_change_plateau(temperature_change)

        # define T_iteration for next loop
        T_forecast_iteration = T_0 + temperature_change_plateau
        # T_iteration = np.concatenate([T_historical, T_forecast_iteration[1:]])
        T_iteration = T_forecast_iteration

        if iteration == 0 or iteration == 1:
            peak_T = max(T_iteration)
        else:
            previous_peak_T = peak_T
            peak_T = max(T_iteration)
            if abs(peak_T - previous_peak_T) < 0.005:
                break

        if iteration == num_of_iterations - 1:
            print_convergence_error(P_100, P_50, consumption_discount, consumption_growth, r, s)

    T_complete = T_iteration

    if return_all_output:
        return peak_T, SCC_forecasted, forecasted_abatement, forecasted_emissions, T_complete
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


def check_SCC_calculated(P_100, SCC_calculated):
    if SCC_calculated[-1] < P_100:
        print("P_100 not achieved by achieved by final perturbed year")


def temp_change_plateau(temperature_change):
    temperature_change_plateau = np.array(temperature_change, copy=True)
    for i in range(len(temperature_change_plateau)):
        if i > np.argmax(temperature_change_plateau):
            temperature_change_plateau[i] = max(temperature_change_plateau)
    return temperature_change_plateau


# def create_T_initial(T_0, T_historical, alpha, delta_T, years_forecast_length):
#     T_forecast_initial = T_0 + (delta_T-T_0) * (1 - np.exp(-alpha * np.arange(years_forecast_length)))
#     T_complete_initial = np.concatenate([T_historical, T_forecast_initial[1:]])
#     return T_complete_initial, T_forecast_initial
#
#
# def read_historical_T():
#     T_gas_df = pd.read_csv("T_gas.csv", index_col=0)
#     T_historical = T_gas_df['CO2_best']
#     T_0 = T_historical[2019]
#     return T_0, T_historical


def calculate_cumulative_emissions(forecasted_emissions):
    cumulative_emissions_array = np.append(np.zeros(1), np.cumsum(forecasted_emissions)[:-1])
    return cumulative_emissions_array


def forecast_SCC(SCC_calculated, years_forecast):
    # log_SCC = np.log(SCC_calculated)
    # ## add linear fit
    # X = sm.add_constant(years_of_perturbation)  # add a constant to fit
    # results = sm.OLS(log_SCC, X).fit()  # save results of fit
    # SCC_forecasted = np.exp(results.params[0] + results.params[1] * years_forecast)

    SCC_forecasted = []

    for i in range(len(years_forecast)):
        if i < len(SCC_calculated):
            SCC_forecasted.append(SCC_calculated[i])
        else:
            SCC_forecasted.append(SCC_calculated[-1])

    SCC_forecasted = np.array(SCC_forecasted)

    P0 = SCC_forecasted[0]
    return SCC_forecasted, P0


# calculate_SCC_for_SCC_years(T_TCRE, T_forecast_iteration, years_forecast_length, years_forecast,
#                                                       T_historical, T_iteration, W, consumption_discount, k_s, years,
#                                                       years_of_perturbation, gamma, D0, P_100)

# def calculate_SCC_for_SCC_years(T_TCRE, T_forecast_iteration, years_forecast_length, years_forecast, T_historical,
#                                       T_iteration, W, consumption_discount, k_s, years, years_of_perturbation,
#                                       gamma, D0, P_100):
#     SCC_list = []
#     for SCC_year in range(len(years_of_perturbation)):
#         ## define perturbation
#         T_perturbed = create_T_perturbed(T_TCRE, T_forecast_iteration, years_forecast_length, years_forecast,
#                                          T_historical, k_s, SCC_year, years_of_perturbation)
#
#         ## define discount function
#         discount_function = create_discount_function(consumption_discount, SCC_year, years,
#                                                      years_of_perturbation)
#         cost = cost_of_perturbation(T_iteration, T_perturbed, W, discount_function, gamma, D0)
#         SCC = cost / (10 ** 9)
#         if SCC < P_100:
#             SCC_list.append(SCC)
#         else:
#             SCC_list.append(P_100)
#             break
#     SCC_calculated = np.asarray(SCC_list)
#     return SCC_calculated


# def create_T_perturbed(T_TCRE, T_forecast_iteration, years_forecast_length, years_forecast, T_historical, k_s, SCC_year,
#                        years_of_perturbation):
#     T_p = np.zeros(years_forecast_length)
#     for forecasted_year in range(years_forecast_length):
#         if years_of_perturbation[SCC_year] <= years_forecast[forecasted_year]:
#             T_p[forecasted_year] = T_TCRE * (
#                         1 - np.exp(-k_s * (years_forecast[forecasted_year] - years_of_perturbation[SCC_year])))
#     T_forecast_perturbed = T_forecast_iteration + T_p
#     T_perturbed = np.concatenate([T_historical, T_forecast_perturbed[1:]])
#     return T_perturbed
#
#
# def create_discount_function(consumption_discount, SCC_year, years, years_of_perturbation):
#     num_of_years = len(years)
#     discount_function = np.zeros(num_of_years)
#     for forecasted_year in range(num_of_years):
#         if years[forecasted_year] >= years_of_perturbation[SCC_year]:
#             discount_function[forecasted_year] = np.exp(
#                 -consumption_discount * (years[forecasted_year] - years_of_perturbation[SCC_year]))
#     return discount_function


# def get_CO2_baseline():
#     # ssp = 'ssp245'
#     # # get emissions data using imported scripts + convert into FaIRv2.0.0-alpha multiindex format
#     # ssp_emms = pd.concat([get_ssp_emissions(ssp)], axis=1, keys=[ssp])
#     # ssp_df = pd.read_csv("ssp245.csv", index_col=0)
#     # CtoCO2_conversion = 44 / 12
#     # # ssp245_CO2_past = ssp_emms["ssp245"]["carbon_dioxide"] * CtoCO2_conversion
#     # ssp245_CO2_past = ssp_df["carbon_dioxide"] * CtoCO2_conversion
#     # CO2_baseline = ssp245_CO2_past[2019]
#     CO2_baseline  = 40.339401131162724
#     return CO2_baseline


# def create_years_array(first_year, end_year):
#     years = np.arange(first_year, end_year + 1)
#     return years
#
#
# def create_total_consumption(first_year, last_historical_year, years, consumption_growth):
#     W_2019 = 80
#     W = []
#     for i in range(len(years)):
#         W.append(W_2019 * np.exp(consumption_growth * (i - (last_historical_year - first_year))))
#     W = np.asarray(W)
#     return W


def cost_of_perturbation(T, T_perturb, W, discount_function, gamma=2, D0=0.00267):
    S_Wt = D0 * T ** gamma
    S_Wt_perturb = D0 * T_perturb ** gamma
    consumption_loss_fraction = S_Wt_perturb - S_Wt
    absolute_consumption_loss = consumption_loss_fraction * W
    discounted_consumption_loss = absolute_consumption_loss * discount_function
    area = sum(discounted_consumption_loss)
    cost = area * 10 ** 12  # convert to dollar amount

    return cost


def abatement(P, P0, P_50, r, s, P_100):
    if P0 >= P_50:
        print("P0 is greater than P_50")
    elif ((P_100 - P0) / (P_50 - P0)) ** (s / r) <= 2:
        print("MAC curve condition not satisfied")

    P_h = P0 + ((P_50 - P0) ** (-s / r) - 2 * (P_100 - P0) ** (-s / r)) ** (-r / s)

    Am = 1 + ((P_100 - P0) / (P_h - P0)) ** (-s / r)
    A = Am / (1 + ((P - P0) / (P_h - P0)) ** (-s / r))
    return A


# def get_ssp_emissions(ssp, end_year=2019):
#     emms = RCMIP_to_FaIR_input_emms(ssp).interpolate().loc[1750:end_year]
#
#     ## rebase emission-driven forcings & species with natural emissions included in RCMIP to zero @ 1750
#     rebase_species = ['so2', 'nox', 'co', 'nmvoc', 'bc', 'nh3', 'oc', 'nox_avi', 'methyl_bromide', 'methyl_chloride',
#                       'chcl3', 'ch2cl2']
#     emms.loc[:, rebase_species] -= emms.loc[1750, rebase_species]
#
#     return emms


def abatement_to_emissions(forecasted_abatement, CO2_baseline):
    CO2_emissions = CO2_baseline * (1 - forecasted_abatement)

    return CO2_emissions
