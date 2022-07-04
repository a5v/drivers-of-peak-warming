import numpy as np

def create_years_array(first_year, end_year):
    years = np.arange(first_year, end_year + 1)
    return years


def create_total_consumption(years, W_fixed_year=2020, W_fixed=80, consumption_growth=0.02):
    start_year = years[0]

    W = []
    for i in range(len(years)):
        W.append(W_fixed * np.exp(consumption_growth * (i - (W_fixed_year - start_year))))
    W = np.asarray(W)
    return W

def create_discount_function(years, SCC_year, consumption_discount=0.035):
    num_of_years = len(years)
    discount_function = np.zeros(num_of_years)
    for year in range(num_of_years):
        if years[year] >= SCC_year:
            discount_function[year] = np.exp(-consumption_discount * (years[year] - SCC_year))
    return discount_function

def create_geometric_T(years, T_0=1.2, alpha=0.02, delta_T=3):
    num_of_years = len(years)
    geometric_T = T_0 + (delta_T-T_0) * (1 - np.exp(-alpha * np.arange(num_of_years)))
    return geometric_T

def create_geometric_T_perturbed(years, T, SCC_year, T_TCRE, k_s):
    num_of_years = len(years)
    T_p = np.zeros(num_of_years)
    for year in range(num_of_years):
        if years[year] >= SCC_year:
            T_p[year] = T_TCRE * (1 - np.exp(-k_s * (years[year] - SCC_year)))
    geometric_T_perturbed = T + T_p
    return geometric_T_perturbed


def SCC_calculator(time_series_df, size_of_perturbation, gamma=2, D0=0.00267):  # conventional damage function SCC
    T = time_series_df['T'].to_numpy()
    T_perturbed = time_series_df['T perturbed'].to_numpy()
    W = time_series_df['W'].to_numpy()
    discount_function = time_series_df['discount function'].to_numpy()

    S_Wt = D0 * T ** gamma
    S_Wt_perturb = D0 * T_perturbed ** gamma
    consumption_loss_fraction = S_Wt_perturb - S_Wt
    absolute_consumption_loss = consumption_loss_fraction * W
    discounted_consumption_loss = absolute_consumption_loss * discount_function
    area = sum(discounted_consumption_loss)
    SCC = area * 10 ** 12 / (size_of_perturbation * 10 ** 9)  # convert to dollar amount and normalise for 1 tCO2
    return SCC

