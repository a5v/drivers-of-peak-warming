def abatement(P, P0, P_50, r, s, P_100):  # 0.05
    if P0 >= P_50:
        print("P0 is greater than P_50")
    elif ((P_100 - P0) / (P_50 - P0)) ** (s / r) <= 2:
        print("MAC curve condition not satisfied")

    P_h = P0 + ((P_50 - P0) ** (-s / r) - 2 * (P_100 - P0) ** (-s / r)) ** (-r / s)
    Am = 1 + ((P_100 - P0) / (P_h - P0)) ** (-s / r)

    A = Am / (1 + ((P - P0) / (P_h - P0)) ** (-s / r))

    return A


