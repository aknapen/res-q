from matplotlib.pylab import ceil
from res_q import RareEventSimulator
from qldpc.objects import Pauli
import random

def test():
    d = 5
    sim = RareEventSimulator(
        distance=d,
        physical_p=1e-3,
        target_p=1e-6,
        shots_per_chain=100,
        rng_seed=42,
        basis=Pauli.X
    )
    
    print(f"Number of gates : {len(sim.gate_list)}")
    print(f"Number of (gate, fault) pairs: {len(sim.gate_fault_list)}")
    
    result = sim.is_malicious(set())
    print(f"Empty set causes logical error: {result}")
    
    
    #  Try to find a malicious error set of weight ceil(d/2) = 2
    random.seed(52)
    found_malicious = False
    
    for i in range(200):
        error_set = set(random.sample(sim.gate_fault_list, int(ceil(d/2))))
        if sim.is_malicious(error_set):
            found_malicious = True
            print(f"Found malicious set on attempt {i + 1}")
            break
    
    if not found_malicious:
        print("Did not find malicious set in 200 attempts")
    
    
    random.seed(52)
    test_set = frozenset(random.sample(sim.gate_fault_list, int(ceil(d/2))))
    
    p_high = 1e-3
    p_low = 1e-4
    
    prob_high = sim._approx_prob_of_set(test_set, p_high)
    prob_low = sim._approx_prob_of_set(test_set, p_low)
    
    print(f"Probability at p={p_high}: {prob_high:.6e}")
    print(f"Probability at p={p_low}: {prob_low:.6e}")
    print(f"Ratio: {prob_high / prob_low:.2f}")
    
if __name__ == '__main__':
    test()


