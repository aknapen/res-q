import numpy as np

def g(x):
    return 1.0 / (1.0 + x)

def IE_function(C, pi_j, pi_j_plus_1):
   
    r = C * pi_j / pi_j_plus_1
    g_vals = g(r)
    return np.mean(g_vals) #1/N E, j=1..N g(C^{-1} pi_{j+1}/pi_j)

def log_binary_search(samples_j, samples_j_plus_1,  pi_j_func, pi_j_plus_1_func,  tolerance=1e-9, max_iter=100):

    pi_j_top = np.array([pi_j_func(E) for E in samples_j])
    pi_j_plus_1_top = np.array([pi_j_plus_1_func(E) for E in samples_j])
    
    pi_j_bottom = np.array([pi_j_func(E) for E in samples_j_plus_1])
    pi_j_plus_1_bottom = np.array([pi_j_plus_1_func(E) for E in samples_j_plus_1])
    ############# This is counter intuitive because  normally pi_j_top === pi_j_bottom , but not here.
    
    

    # log_C_min = 1  
    # log_C_max = 100.0   
    log_C_min = np.log(0.01)   # or whatever your actual C_min should be
    log_C_max = np.log(10.0) 
  
    C_min = 0
    C_max = 10.0
    
    for i in range(max_iter):
        log_C = (log_C_min + log_C_max) / 2.0
        C = np.exp(log_C)
        
        # C = (C_min + C_max) / 2.0
        top_exp = IE_function(C,  pi_j_top, pi_j_plus_1_top)
        bottom_exp = IE_function((1.0 / C), pi_j_bottom, pi_j_plus_1_bottom)
        
        difference = top_exp - bottom_exp
  
   
        # Check convergence
        if abs(difference) < tolerance:
            print(f"\ndifference! d = {difference:.6f}")
            print(f"\nConverged! C = {C:.6f}")
            return C
        if log_C_max <= log_C_min:
            # print(f"\nBounds may not have converged! C = {C:.6f}")
            return C
        
        # Update bounds based on the sign of difference
        # If top > bottom, we need to decrease C , I am not sure about this because when C increases the top one decrease because of 1/(1+Cx)
        if difference < 0:
            log_C_max = log_C
            # C_max = C
        else:
            # C_min = C
            log_C_min = log_C
        
    print(f"\nBounds may not have converged! C = {C:.6f}")
    top_exp = IE_function(C,  pi_j_top, pi_j_plus_1_top)
    bottom_exp = IE_function((1.0 / C), pi_j_bottom, pi_j_plus_1_bottom)
    print(f"Final difference: {top_exp - bottom_exp}")    
    return C

