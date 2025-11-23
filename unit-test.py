from math import ceil
from typing import Set
import random
import unittest

import numpy as np
import numpy.testing as npt
import stim
from qldpc.objects import Pauli

from res_q import RareEventSimulator, GateFault

class TestRareEventSimulator(unittest.TestCase):
    def test_inject_events_empty(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        
        actual_ckt = sim.inject_events(event_set)

        expected_ckt = sim.circuit.without_noise().flattened()
        
        self.assertTrue(expected_ckt.approx_equals(actual_ckt, atol=0.00001))

    def test_inject_events_sorted(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        event_set.add(
            ((17, (13,), 'Z'), ('Z',))
        )
        event_set.add(
            ((22, (9, 1), 'DEPOLARIZE2'), ('X','Z'))
        )
        event_set.add(
            ((55, (0,), 'MEAS_RESET_IDLE'), ('Y',))
        )
        event_set.add(
            ((68, (12, 4), 'DEPOLARIZE2'), ("Z", "I"))
        )
        actual_ckt = sim.inject_events(event_set)

        expected_ckt = stim.Circuit('''
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(0, 1) 1
            QUBIT_COORDS(0, 2) 2
            QUBIT_COORDS(0, 3) 3
            QUBIT_COORDS(0, 4) 4
            QUBIT_COORDS(0, 5) 5
            QUBIT_COORDS(0, 6) 6
            QUBIT_COORDS(0, 7) 7
            QUBIT_COORDS(0, 8) 8
            QUBIT_COORDS(1, 0) 9
            QUBIT_COORDS(1, 1) 10
            QUBIT_COORDS(1, 2) 11
            QUBIT_COORDS(1, 3) 12
            QUBIT_COORDS(1, 4) 13
            QUBIT_COORDS(1, 5) 14
            QUBIT_COORDS(1, 6) 15
            QUBIT_COORDS(1, 7) 16
            RX 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            Z_ERROR(1.0) 13
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            X_ERROR(1.0) 9
            Z_ERROR(1.0) 1
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(0, 0, 0) rec[-8]
            DETECTOR(0, 0, 1) rec[-7]
            DETECTOR(0, 0, 2) rec[-6]
            DETECTOR(0, 0, 3) rec[-5]
            TICK
            RX 9 10 11 12 13 14 15 16
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(1, 0, 0) rec[-8] rec[-16]
            DETECTOR(1, 0, 1) rec[-7] rec[-15]
            DETECTOR(1, 0, 2) rec[-6] rec[-14]
            DETECTOR(1, 0, 3) rec[-5] rec[-13]
            Y_ERROR(1.0) 0
            TICK
            RX 9 10 11 12 13 14 15 16
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            Z_ERROR(1.0) 12
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(2, 0, 0) rec[-8] rec[-16]
            DETECTOR(2, 0, 1) rec[-7] rec[-15]
            DETECTOR(2, 0, 2) rec[-6] rec[-14]
            DETECTOR(2, 0, 3) rec[-5] rec[-13]
            TICK
            MX 0 1 2 3 4 5 6 7 8
            DETECTOR(5, 0, 0) rec[-9] rec[-8] rec[-6] rec[-5] rec[-17]
            DETECTOR(5, 0, 1) rec[-7] rec[-4] rec[-16]
            DETECTOR(5, 0, 2) rec[-6] rec[-3] rec[-15]
            DETECTOR(5, 0, 3) rec[-5] rec[-4] rec[-2] rec[-1] rec[-14]
            OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
        ''')
        
        self.assertTrue(expected_ckt.approx_equals(actual_ckt, atol=0.001))

    def test_inject_events_unsorted(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        event_set.add(
            ((55, (0,), 'MEAS_RESET_IDLE'), ('Y',))
        )
        event_set.add(
            ((22, (9, 1), 'DEPOLARIZE2'), ('X','Z'))
        )
        event_set.add(
            ((17, (13,), 'Z'), ('Z',))
        )
        event_set.add(
            ((68, (12, 4), 'DEPOLARIZE2'), ("Z", "I"))
        )
        actual_ckt = sim.inject_events(event_set)

        expected_ckt = stim.Circuit('''
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(0, 1) 1
            QUBIT_COORDS(0, 2) 2
            QUBIT_COORDS(0, 3) 3
            QUBIT_COORDS(0, 4) 4
            QUBIT_COORDS(0, 5) 5
            QUBIT_COORDS(0, 6) 6
            QUBIT_COORDS(0, 7) 7
            QUBIT_COORDS(0, 8) 8
            QUBIT_COORDS(1, 0) 9
            QUBIT_COORDS(1, 1) 10
            QUBIT_COORDS(1, 2) 11
            QUBIT_COORDS(1, 3) 12
            QUBIT_COORDS(1, 4) 13
            QUBIT_COORDS(1, 5) 14
            QUBIT_COORDS(1, 6) 15
            QUBIT_COORDS(1, 7) 16
            RX 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            Z_ERROR(1.0) 13
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            X_ERROR(1.0) 9
            Z_ERROR(1.0) 1
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(0, 0, 0) rec[-8]
            DETECTOR(0, 0, 1) rec[-7]
            DETECTOR(0, 0, 2) rec[-6]
            DETECTOR(0, 0, 3) rec[-5]
            TICK
            RX 9 10 11 12 13 14 15 16
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(1, 0, 0) rec[-8] rec[-16]
            DETECTOR(1, 0, 1) rec[-7] rec[-15]
            DETECTOR(1, 0, 2) rec[-6] rec[-14]
            DETECTOR(1, 0, 3) rec[-5] rec[-13]
            Y_ERROR(1.0) 0
            TICK
            RX 9 10 11 12 13 14 15 16
            TICK
            CX 9 4 11 6 12 8
            CZ 13 1 14 5 15 7
            TICK
            CX 9 1 11 3 12 5
            CZ 13 0 14 4 15 6
            TICK
            CX 9 3 10 5 12 7
            CZ 14 2 15 4 16 8
            TICK
            CX 9 0 10 2 12 4
            Z_ERROR(1.0) 12
            CZ 14 1 15 3 16 7
            TICK
            MX 9 10 11 12 13 14 15 16
            DETECTOR(2, 0, 0) rec[-8] rec[-16]
            DETECTOR(2, 0, 1) rec[-7] rec[-15]
            DETECTOR(2, 0, 2) rec[-6] rec[-14]
            DETECTOR(2, 0, 3) rec[-5] rec[-13]
            TICK
            MX 0 1 2 3 4 5 6 7 8
            DETECTOR(5, 0, 0) rec[-9] rec[-8] rec[-6] rec[-5] rec[-17]
            DETECTOR(5, 0, 1) rec[-7] rec[-4] rec[-16]
            DETECTOR(5, 0, 2) rec[-6] rec[-3] rec[-15]
            DETECTOR(5, 0, 3) rec[-5] rec[-4] rec[-2] rec[-1] rec[-14]
            OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
        ''')
        
        # print(f"Expected:\n{expected_ckt}")
        # print("======================")
        # print(f"Actual:\n{actual_ckt}")

        self.assertTrue(expected_ckt.approx_equals(actual_ckt, atol=0.001))

    def test_samples_from_injected_circuit_empty(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        noisy_ckt = sim.inject_events(event_set)

        sampler = noisy_ckt.compile_detector_sampler()
        dets, _ = sampler.sample(shots=1, separate_observables=True)
        detectors = np.flatnonzero(dets[0])

        npt.assert_equal(detectors, np.array([]))
        
    def test_samples_from_injected_circuit(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        event_set.add(
            ((17, (13,), 'Z'), ('Z',))
        )
        event_set.add(
            ((22, (9, 1), 'DEPOLARIZE2'), ('X','Z'))
        )
        event_set.add(
            ((55, (0,), 'MEAS_RESET_IDLE'), ('Y',))
        )
        event_set.add(
            ((68, (12, 4), 'DEPOLARIZE2'), ("Z", "I"))
        )
        noisy_ckt = sim.inject_events(event_set)

        sampler = noisy_ckt.compile_detector_sampler()
        dets, _ = sampler.sample(shots=1, separate_observables=True)
        detectors = np.flatnonzero(dets[0])

        npt.assert_equal(detectors, np.array([4, 8, 11, 15]))

    def test_is_malicious_empty(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()

        logical_error = sim.is_malicious(event_set)

        self.assertFalse(logical_error)
        
    def test_is_malicious_true(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        event_set.add(
            ((17, (0,), 'Z'), ('Z',))
        )
        event_set.add(
            ((17, (3,), 'Z'), ('Z',))
        )

        logical_error = sim.is_malicious(event_set)

        self.assertTrue(logical_error)

    def test_is_malicious_false(self):
        d = 3
        sim = RareEventSimulator(
            distance=d,
            target_p=1E-4,
            basis=Pauli.X
        )

        # Create a known set of failure events
        event_set: Set[GateFault] = set()
        event_set.add(
            ((17, (0,), 'Z'), ('Z',))
        )

        logical_error = sim.is_malicious(event_set)

        self.assertFalse(logical_error)

    # def test(self):
    #     d = 5
    #     sim = RareEventSimulator(
    #         distance=d,
    #         physical_p=1e-3,
    #         target_p=1e-6,
    #         shots_per_chain=100,
    #         rng_seed=42,
    #         basis=Pauli.X
    #     )
        
    #     print(f"Number of gates : {len(sim.gate_list)}")
    #     print(f"Number of (gate, fault) pairs: {len(sim.gate_fault_list)}")
        
    #     result = sim.is_malicious(set())
    #     print(f"Empty set causes logical error: {result}")
        
        
    #     #  Try to find a malicious error set of weight ceil(d/2) = 2
    #     random.seed(52)
    #     found_malicious = False
        
    #     for i in range(200):
    #         error_set = set(random.sample(sim.gate_fault_list, int(ceil(d/2))))
    #         if sim.is_malicious(error_set):
    #             found_malicious = True
    #             print(f"Found malicious set on attempt {i + 1}")
    #             break
        
    #     if not found_malicious:
    #         print("Did not find malicious set in 200 attempts")
        
        
    #     random.seed(52)
    #     test_set = frozenset(random.sample(sim.gate_fault_list, int(ceil(d/2))))
        
    #     p_high = 1e-3
    #     p_low = 1e-4
        
    #     prob_high = sim._approx_prob_of_set(test_set, p_high)
    #     prob_low = sim._approx_prob_of_set(test_set, p_low)
        
    #     print(f"Probability at p={p_high}: {prob_high:.6e}")
    #     print(f"Probability at p={p_low}: {prob_low:.6e}")
    #     print(f"Ratio: {prob_high / prob_low:.2f}")
    
if __name__ == '__main__':
    unittest.main()


