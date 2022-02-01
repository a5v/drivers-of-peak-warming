import pandas as pd
import numpy as np
from scipy.integrate import simps
import statsmodels.api as sm
from fair.scripts.data_retrieval import RCMIP_to_FaIR_input_emms


def peak_temperature_calculator(consumption_discount=0.035, gamma=2, D0=0.00267, P_h=44, r=20, s=18, Am=1.1,
                                end_year=2500, last_perturbed_year=2100, return_all_output=False):

    start_year = 1750
    last_historical_year = 2019
    first_perturbed_year = last_historical_year

    years_complete = create_years_array(start_year, end_year)
    years_of_perturbation = create_years_array(first_perturbed_year, last_perturbed_year)
    years_forecasted = create_years_array(last_historical_year, end_year)
    years_forecasted_length = len(years_forecasted)

    T_TCRE = 0.00054  # need to check this! Note this is 1 GtCO2
    k_s = 0.12

    delta_T = 2
    alpha = 0.02

    T_2019, T_historical = read_historical_T()

    T_complete_initial, T_forecasted_initial = create_T_initial(T_2019, T_historical, alpha, delta_T,
                                                                years_forecasted_length)

    W = create_total_consumption(start_year, last_historical_year, years_complete)

    CO2_baseline = get_CO2_baseline()

    num_of_iterations = 6
    T_forecasted_iteration = T_forecasted_initial
    T_complete_iteration = T_complete_initial
    for iteration in range(num_of_iterations):

        SCC_array = calculate_SCC_for_perturbed_years(T_TCRE, T_forecasted_iteration, years_forecasted_length, years_forecasted,
                                                      T_historical, T_complete_iteration, W, consumption_discount, k_s, years_complete,
                                                      years_of_perturbation, gamma, D0)

        SCC_forecasted, P0 = forecast_SCC(SCC_array, years_forecasted, years_of_perturbation)

        forecasted_abatement = abatement(SCC_forecasted, P0, P_h, r, s, Am)
        forecasted_emissions = abatement_to_emissions(forecasted_abatement, CO2_baseline)
        cumulative_emissions_array = calculate_cumulative_emissions(years_forecasted, forecasted_emissions)
        temperature_change = T_TCRE * cumulative_emissions_array

        # define T_iteration for next loop
        T_forecasted_iteration = T_2019 + temperature_change
        T_complete_iteration = np.concatenate([T_historical, T_forecasted_iteration[1:]])

    peak_T = max(T_complete_iteration)
    T_complete = T_complete_iteration

    if return_all_output:
        return peak_T, SCC_forecasted, forecasted_abatement, forecasted_emissions, T_complete
    else:
        return peak_T


def create_T_initial(T_2019, T_historical, alpha, delta_T, years_forecasted_length):
    T_forecasted_initial = T_2019 + delta_T * (1 - np.exp(-alpha * np.arange(years_forecasted_length)))
    T_complete_initial = np.concatenate([T_historical, T_forecasted_initial[1:]])
    return T_complete_initial, T_forecasted_initial


def read_historical_T():
    T_gas_df = pd.read_csv("T_gas.csv", index_col=0)
    T_historical = T_gas_df['CO2_best']
    T_2019 = T_historical[2019]
    return T_2019, T_historical


def calculate_cumulative_emissions(T_forecast_years, forecasted_emissions):
    cumulative_emissions = []
    for forecasted_year in range(len(T_forecast_years)):
        area = simps(forecasted_emissions[:forecasted_year + 1], dx=1)
        cumulative_emissions.append(area)
    cumulative_emissions_array = np.asarray(cumulative_emissions)
    return cumulative_emissions_array


def forecast_SCC(SCC_array, T_forecast_years, years_of_perturbation):
    log_SCC = np.log(SCC_array)
    ## add linear fit
    X = sm.add_constant(years_of_perturbation)  # add a constant to fit
    results = sm.OLS(log_SCC, X).fit()  # save results of fit
    SCC_forecasted = np.exp(results.params[0] + results.params[1] * T_forecast_years)
    P0 = SCC_forecasted[0]
    return SCC_forecasted, P0


def calculate_SCC_for_perturbed_years(T_TCRE, T_forecast_iteration, T_forecast_length, T_forecast_years, T_gas_df,
                                      T_total_iteration, W, consumption_discount, k_s, years, years_of_perturbation,
                                      gamma, D0):
    SCC_list = []
    for perturbed_year in range(len(years_of_perturbation)):
        ## define perturbation

        T_perturbed = create_T_perturbed(T_TCRE, T_forecast_iteration, T_forecast_length, T_forecast_years,
                                         T_gas_df, k_s, perturbed_year, years_of_perturbation)

        ## define discount function
        discount_function = create_discount_function(consumption_discount, perturbed_year, years,
                                                     years_of_perturbation)
        cost = cost_of_perturbation(T_total_iteration, T_perturbed, W, discount_function, gamma, D0)
        SCC = cost / (10 ** 9)
        SCC_list.append(SCC)
    SCC_array = np.asarray(SCC_list)
    return SCC_array


def create_T_perturbed(T_TCRE, T_forecast_iteration, T_forecast_length, T_forecast_years, T_historical, k_s, perturbed_year,
                       years_of_perturbation):
    T_p = np.zeros(T_forecast_length)
    for forecasted_year in range(T_forecast_length):
        if years_of_perturbation[perturbed_year] <= T_forecast_years[forecasted_year]:
            T_p[forecasted_year] = T_TCRE * (
                        1 - np.exp(-k_s * (T_forecast_years[forecasted_year] - years_of_perturbation[perturbed_year])))
    T_forecast_perturbed = T_forecast_iteration + T_p
    T_perturbed = np.concatenate([T_historical, T_forecast_perturbed[1:]])
    return T_perturbed


def create_discount_function(consumption_discount, perturbed_year, years, years_of_perturbation):
    num_of_years = len(years)
    discount_function = np.zeros(num_of_years)
    for forecasted_year in range(num_of_years):
        if years[forecasted_year] >= years_of_perturbation[perturbed_year]:
            discount_function[forecasted_year] = np.exp(
                -consumption_discount * (years[forecasted_year] - years_of_perturbation[perturbed_year]))
    return discount_function


def get_CO2_baseline():
    ssp = 'ssp245'
    # get emissions data using imported scripts + convert into FaIRv2.0.0-alpha multiindex format
    ssp_emms = pd.concat([get_ssp_emissions(ssp)], axis=1, keys=[ssp])
    CtoCO2_conversion = 44 / 12
    ssp245_CO2_past = ssp_emms["ssp245"]["carbon_dioxide"] * CtoCO2_conversion
    CO2_baseline = ssp245_CO2_past[2019]
    return CO2_baseline


def create_years_array(first_year, end_year):
    years = np.arange(first_year, end_year + 1)
    return years


def create_total_consumption(first_year, last_historical_year, years):
    consumption_growth = 1.02
    W_2019 = 80
    W = []
    for i in range(len(years)):
        W.append(W_2019 * consumption_growth ** (i - (last_historical_year - first_year)))
    W = np.asarray(W)
    return W


def cost_of_perturbation(T, T_perturb, W, discount_function, gamma=2, D0=0.00267):
    S_Wt = D0 * T ** gamma
    S_Wt_perturb = D0 * T_perturb ** gamma
    consumption_loss_fraction = S_Wt_perturb - S_Wt
    absolute_consumption_loss = consumption_loss_fraction * W
    discounted_consumption_loss = absolute_consumption_loss * discount_function
    area = simps(discounted_consumption_loss, dx=1)
    cost = area * 10 ** 12  # convert to dollar amount

    return cost


def abatement(P, P0, P_h, r, s, Am):
    A = Am / (1 + ((P - P0) / P_h) ** (-r / s))

    return A


def get_ssp_emissions(ssp, end_year=2019):
    emms = RCMIP_to_FaIR_input_emms(ssp).interpolate().loc[1750:end_year]

    ## rebase emission-driven forcings & species with natural emissions included in RCMIP to zero @ 1750
    rebase_species = ['so2', 'nox', 'co', 'nmvoc', 'bc', 'nh3', 'oc', 'nox_avi', 'methyl_bromide', 'methyl_chloride',
                      'chcl3', 'ch2cl2']
    emms.loc[:, rebase_species] -= emms.loc[1750, rebase_species]

    return emms


def abatement_to_emissions(forecasted_abatement, CO2_baseline):
    CO2_emissions = CO2_baseline * (1 - forecasted_abatement)

    return CO2_emissions