import numpy as np

def g(x):
    return 1.0 / (1.0 + x)

def log_binary_search(Es_num, Es_den, prob_func_curr, prob_func_next, 
                      tolerance=1e-9, max_iters=100) -> float:
    '''
        Performs a binary search to find a constant satisfying Eq. (4) in
        https://arxiv.org/pdf/2509.13678

        prob_func_curr := p_i(E)
        prob_func_next := p_i+1(E)
    '''
    # precompute probability ratios

    # Numerator expected value has term: π_i(E) / π_i+1(E)
    probs_num = [prob_func_curr(E) / prob_func_next(E) for E in Es_num]

    # Denominator expected value has term: π_i+1(E) / π_i(E)
    probs_den = [prob_func_next(E) / prob_func_curr(E) for E in Es_den] 

    log_C_min = np.log(1E-10)
    log_C_max = np.log(1E10) 
    for i in range(max_iters):
        # Guess C at midpoint of current search interval
        log_C = (log_C_min + log_C_max) / 2.0
        C = np.exp(log_C)

        # Compute (1 / N) * sum[ g( C * (π_i(Ej) / π_i+1(Ej)) )]
        expected_num = np.mean([g(C * probs_num[j]) for j in range(len(probs_num))])

        # Compute: (1 / N) * sum[ g( C^-1 * (π_i+1(Ej) / π_i(Ej)) )]
        expected_den = np.mean([g((1/C) * probs_den[j]) for j in range(len(probs_den))])

        diff = expected_num - expected_den

        if abs(diff) < tolerance:
            return C
        # Fraction less than 1, need to decrease C to increase the fraction
        elif diff < 0:
            log_C_max = log_C
        # Fraction greater than 1, need to increase C to decrease the fraction
        else:
            log_C_min = log_C
    
    # Search didn't converge
    return C
