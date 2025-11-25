import stim
import numpy as np
from scipy.stats import norm

from util import log_binary_search

from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple, Set, Dict, Optional, Any

import stim
import pymatching

from qldpc.circuits.noise_model import SI1000NoiseModel, NoiseRule
from qldpc.objects import Pauli, PauliXZ
from qldpc.circuits import get_memory_experiment
from qldpc.codes.quantum import CSSCode, SurfaceCode
from qldpc.circuits.bookkeeping import QubitIDs

"""
Rare-event splitting simulator for Stim circuits

This is a runnable Python implementation (template + working code) of the
rare-event splitting / MCMC technique described in "Rare Event Simulation of
Quantum Error-Correcting Circuits" (Mayer et al., 2025). It is written to be
used with Stim-generated circuits that include DETECTOR/OBSERVABLE
annotations (the usual Stim surface-code circuits do).

How to use
-----------
1. Install Stim (locally):
   pip install stim

2. Create or export a Stim circuit for the rotated surface code in the
   X-basis that contains DETECTOR and OBSERVABLE lines. You can use the
   Stim example circuits, or export one from your repository. Save it to
   disk, e.g. `rotated_surface_code_X.stim`.

   If you already have a Stim circuit object in Python, pass it directly to
   the functions below.

3. Run the script as a module or import functions from it. Example (CLI-like):

   from rare_event_stim import RareEventSimulator
   sim = RareEventSimulator(circuit_path='rotated_surface_code_X.stim',
                            physical_p=1e-3,
                            target_p=1e-6,
                            shots_per_chain=1000)
   result = sim.run()
   print(result)

What this implementation does
-----------------------------
- Implements a splitting schedule heuristic (as in the paper) to produce a
  sequence of physical p values decreasing from an initial p0 to target p_t.
- Implements a Metropolis-Hastings Markov chain on the space of failing
  events (gate, fault) pairs, using acceptance probabilities described in
  the paper (Section II/III).
- Uses Stim (when available) to: parse the circuit, build a detector-error
  model (DEM), and deterministically inject Pauli errors corresponding to
  a chosen (gate,fault) set to test whether that set is malignant (i.e., it
  causes a logical failure). For this, we use the DEM to map faulty events
  to Pauli products that flip detectors/observables.

Limitations / Notes
-------------------
- This implementation assumes Stim is installed locally. The runtime environment
  provided with this chat does not have Stim installed, so I cannot run it
  here.
- The mapping from (gate,fault) to Pauli product is performed approximately
  by using Stim's `detector_error_model(decompose_operations=True)` and
  parsing lines. For complicated custom error models the mapping may require
  customization.
- Decoder calls: this implementation determines logical failure by running a
  deterministic (noiseless) simulation of the circuit with the Pauli errors
  injected and then checking whether any OBSERVABLE flips occurred. If your
  circuit uses a decoder externally, you can adapt `is_malicious` to call it.

References
----------
Mayer et al., "Rare Event Simulation of Quantum Error-Correcting Circuits"
(2025). The algorithm implemented follows Sections II and III.

"""

# Type aliases
Gate = Tuple[int, Tuple[int], str] # (gate_index, (gate_targets), noise_channel)
Fault = Tuple[str] # tuple of Pauli strings
GateFault = Tuple[Gate, Fault]
EventSet = frozenset

# Static fault lookup table
channel_faults = {
    "Z": [("Z")],
    "DEPOLARIZE1": [("Z"), ("X"), ("Y")],
    "IDLE": [("Z"), ("X"), ("Y")],
    "MEAS_RESET_IDLE": [("Z"), ("X"), ("Y")],
    "DEPOLARIZE2": [("I","X"), ("I","Y"), ("I","Z"), 
                    ("X","X"), ("X","Y"), ("X","Z"), 
                    ("Y","X"), ("Y","Y"), ("Y","Z"), 
                    ("Z","X"), ("Z","Y"), ("Z","Z"), 
                    ("X","I"), ("Y","I"), ("Z","I")],
}

@dataclass
class RareEventSimulator:
    distance: int = 3
    p0: float = 1e-3  # starting p for MC setup
    target_p: float = 1e-9
    shots_per_chain: int = 200  # N in the paper for expectation estimates
    rng_seed: Optional[int] = None
    num_warmup_steps: int = 1000
    basis: PauliXZ = Pauli.X

    # internal fields
    circuit: stim.Circuit = field(default=None, init=False)
    dem: stim.DetectorErrorModel = field(default=None, init=False)
    gate_list: List[Gate] = field(default_factory=list, init=False)
    gate_fault_list: List[GateFault] = field(default_factory=list, init=False)
    gate_failure_prob: Dict[Gate, float] = field(default_factory=dict, init=False)
    gate_fault_prob: Dict[GateFault, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.rng_seed is not None:
            random.seed(self.rng_seed)

        self.code: CSSCode = SurfaceCode(rows=self.distance, cols=self.distance, rotated=True)
        self.noise_model = SI1000NoiseModel(self.p0)
        
        self.circuit: stim.Circuit = get_memory_experiment(
            self.code,
            basis=self.basis,
            num_rounds=self.distance,
            noise_model=self.noise_model
        ).flattened()

        self.decoder = pymatching.Matching(
            self.circuit.detector_error_model(decompose_errors=True)
        )

        self._catalog_gate_faults()
        
        # Create all possible (gate, fault) combinations
        self.gate_fault_list = []
        for gate in self.gate_list:
            (_, _, channel) = gate
            for fault in channel_faults[channel]:
                self.gate_fault_list.append((gate, fault))
                
        # Create conditional fault probabilities: P(fault | gate failed)
        self.gate_fault_prob = {}
        for gate_fault in self.gate_fault_list:
            gate, fault = gate_fault
            (_, _, channel) = gate
            # Uniform distribution over possible faults for this channel
            prob_fault_given_failure = 1.0 / len(channel_faults[channel])
            self.gate_fault_prob[gate_fault] = prob_fault_given_failure

    def _catalog_gate_faults(self):
        """Catalog all possible failing gates in a Stim circuit.

        Gates are cataloged uniquely based on their location within the Stim
        circuit (instruction index) as well as the qubits they operate on (to
        differentiate the same instruction action on multiple (sets) of qubits).
        """
        active_qubits: set[int] = set()
        measure_or_reset_in_moment = False

        instr: stim.CircuitInstruction
        for (i, instr) in enumerate(self.circuit.without_noise().flattened()):
            # A TICK instruction indicates a new "moment" (timeslice) in the quantum circuit
            # so we need to re-evaluate the sets of active and idle qubits
            if instr.name == "TICK":
                # Calculate idle qubits
                qubit_ids = QubitIDs.from_code(self.code)
                all_qubits = set(qubit_ids.data + qubit_ids.check)
                
                idle_qubits = all_qubits - active_qubits

                for qubit in idle_qubits:
                    gate_channel = "IDLE"
                    # i-1 here so that the noise channel gets added
                    # at the end of the CURRENT moment, right before 
                    # the TICK marking the start of the NEXT moment
                    # gate: Gate = (i-1, tuple(qubit), gate_channel)
                    gate: Gate = (i-1, (qubit,), gate_channel)

                    self.gate_list.append(gate)
                    self.gate_failure_prob[gate] = self.noise_model.idle_error

                    if measure_or_reset_in_moment:
                        gate_channel = "MEAS_RESET_IDLE"
                        # i-1 here so that the noise channel gets added
                        # at the end of the CURRENT moment, right before 
                        # the TICK marking the start of the NEXT moment
                        
                        # gate: Gate = (i-1, tuple(qubit), gate_channel)
                        gate: Gate = (i-1, (qubit,), gate_channel)
                        
                        self.gate_list.append(gate)
                        self.gate_failure_prob[gate] = self.noise_model.additional_error_waiting_for_m_or_r


                # Reset state for the next circuit moment
                active_qubits.clear()
                measure_or_reset_in_moment = False
            
            noise_rule: NoiseRule = self.noise_model.get_noise_rule(instr)

            # The instruction is some type of noisy gate
            if noise_rule is not None:
                target_group: List[stim.GateTarget]
                for target_group in instr.target_groups():
                    # We define a gate uniquely by its position within the circuit (instruction #)
                    # and the qubit(s) it acts on (instruction targets)
                    gate_targets = tuple([target.qubit_value for target in target_group])
                    
                    # Update the set of qubits involved in operations during this moment
                    active_qubits = active_qubits | set(gate_targets)

                    gate_prob = 0
                    gate_channel = ""
                    if noise_rule.reset_error != 0:
                        gate_prob = noise_rule.reset_error
                        gate_channel = "Z" if self.basis == Pauli.X else "X"
                        measure_or_reset_in_moment = True
                    elif noise_rule.readout_error != 0:
                        gate_prob = noise_rule.readout_error
                        gate_channel = "Z" if self.basis == Pauli.X else "X"
                        measure_or_reset_in_moment = True
                    else:
                        for channel, prob in noise_rule.after.items():
                            # NOTE: this is an assumption that only
                            # one probability is associated with this 
                            # noise rule, which may not be the case for
                            # MPP-type instructions
                            gate_prob = prob[0]
                            gate_channel = channel
                    
                    gate: Gate = (i, gate_targets, gate_channel)
                    self.gate_list.append(gate)
                    self.gate_failure_prob[gate] = gate_prob

    # ---------- Core algorithm pieces ----------
    def splitting_schedule(self, p0: float, pt: float) -> List[float]:
        """Generate decreasing sequence of p values from p0 to pt using heuristic.

        For simplicity we use the heuristic in the paper: p_{i+1} = p_i * 2^{-1/sqrt(wi)}
        with wi = max(d/2, p_i * G).
        """
        ps = [p0]
        G = len(self.gate_list)
        d = self.distance
        while ps[-1] > pt:
            pi = ps[-1]
            wi = max(d/2, pi * G)
            factor = 2 ** (-1.0 / math.sqrt(wi))
            pn = pi * factor
            ps.append(pn)

        return ps

    def inject_events(self, event_set: Set[GateFault]) -> stim.Circuit:
        '''
            Given a set of error events, constructs the corresponding Stim
            circuit in which those events occur.
        '''
        target_circuit: stim.Circuit = self.circuit.without_noise().flattened()
        # Sort events by gate index
        event_list = sorted(event_set, key=lambda x: x[0][0])

        # Each time we insert a new gate error, we need to shift placement
        # of the next gate error by 1 relative to its specified gate index
        index_offset = 0

        for event in event_list:
            (gate, fault) = event
            (gate_index, qubits, _) = gate
            # for i, f in enumerate(f):
            for i, f in enumerate(fault):
                if f != "I":
                    target_circuit.insert(
                        gate_index+1+index_offset, 
                        # stim.CircuitInstruction(fault+"_ERROR", [qubits[i]], [1.0])
                        stim.CircuitInstruction(f+"_ERROR", [qubits[i]], [1.0])
                    )
                    index_offset += 1
        
        return target_circuit

    def is_malicious(self, event_set: Set[GateFault]) -> bool:
        """Test whether a given set of gate-faults causes a logical failure.

        Implementation strategy:
        - Create a copy of the original circuit.
        - For each (gate_index, label) in the event_set, deterministically inject
          corresponding Pauli operations at the appropriate location in the circuit.
        - Generate a sample from the circuit, and compare the flipped observable with
          the prediction outputted by the decoder.
        """
        circuit: stim.Circuit = self.inject_events(event_set)
        sampler = circuit.compile_detector_sampler()
        dets, obs = sampler.sample(shots=1, separate_observables=True)

        pred = self.decoder.decode(dets[0])

        return pred != obs[0]

    def metropolis_step(self, current: Set[GateFault], p: float) -> Set[GateFault]:
        """Perform one Metropolis step modifying a single gate's fault as in paper.

        Returns the new set (may be the same as current if rejected).
        """
        # pick a gate uniformly among catalog and a fault uniformly at random
        # for that gate
        # gate: Gate = random.choice(self.gate_fault_list)

        gate_fault: GateFault = random.choice(self.gate_fault_list)
        gate, _ = gate_fault

        gate: Gate = random.choice(self.gate_list)
        (_, _, channel) = gate
        fault: Fault = random.choice(channel_faults[channel])

        # Probability that the gate would have failed
        prob_g = self.gate_failure_prob[gate] * (p / self.p0)
        # Probability that, given the gate has failed, 
        # the failure would have been the one chosen
        prob_g_f = 1 / len(channel_faults[channel])
        
        event: GateFault = (gate, fault)

        current_gates = set([gf[0] for gf in current])
        if gate in current_gates: # selected gate is already in error set
            # find the (gate, fault) currently in the error set
            current_event: GateFault = next((e for e in current if e[0] == gate), None)
            new = (current | {event}) - {current_event}

            (_, current_fault) = current_event
            if fault == current_fault:
                accept = 1
            else:
                accept = np.random.random() <= prob_g_f
        else: # selected gate is not already in the error set
            new = current | {event}
            accept = np.random.random() <= ((prob_g / (1 - prob_g)) * prob_g_f)

        # Only consider acceptance if new causes a logical failure
        if self.is_malicious(new) and accept:
            return new
        else:
            return current

    def sample_failures(self, p: float, 
                        init_sample: Set[GateFault] = set()) -> List[Set[GateFault]]:
        """Produce samples from pi|F via MCMC (Metropolis) as in the paper.

        This returns a list of distinct failing events discovered by the chain.
        """
        # seed initial failing events via direct Monte Carlo (or heuristic)
        chain_state = init_sample
        steps = 0

        # Warm up chain to converge to stable distribution
        while steps < self.num_warmup_steps:
            new_state = self.metropolis_step(chain_state, p)
            chain_state = new_state
            steps += 1

        steps = 0
        discovered = []
        while steps < self.shots_per_chain:
            new_state = self.metropolis_step(chain_state, p)
            discovered.append(frozenset(new_state))
            chain_state = new_state
            steps += 1

        # while jumps < num_jumps and steps < self.max_steps_per_chain:
        #     new_state = self.metropolis_step(chain_state, p)
        #     # if changed, it's a jump
        #     if new_state != chain_state:
        #         jumps += 1
        #         discovered.append(frozenset(new_state))
        #         chain_state = new_state
        #     steps += 1
        return discovered

    def errors_to_events(self, circuit: stim.Circuit, 
                         dem: stim.DetectorErrorModel, errors: np.ndarray):
        event_set: Set[GateFault] = set()
        flat_errors = [
            instr for instr in dem.flattened() if instr.type == "error"
        ]

        dem_filter = stim.DetectorErrorModel()
        for error in errors:
            dem_filter.clear()
            dem_filter.append(flat_errors[error])
            expl = circuit.explain_detector_error_model_errors(
                dem_filter=dem_filter, 
                reduce_to_one_representative_error=True
            )[0]

            index = expl.circuit_error_locations[0].stack_frames[0].instruction_offset
            qubits = tuple([target.gate_target.value 
                            for target in expl.circuit_error_locations[0].flipped_pauli_product])

            channel = circuit[expl.circuit_error_locations[0].stack_frames[0].instruction_offset].name

            gate: Gate = (index, qubits, channel)
            fault: Fault = tuple([target.gate_target.pauli_type 
                                  for target in expl.circuit_error_locations[0].flipped_pauli_product])

            event_set.add((gate, fault))
        
        return event_set


    def naive_monte_carlo(self, p: float, epsilon: float = 0.01, 
                          alpha: float = 0.05, min_shots: int = 100000, 
                          max_shots: int = 10_000_000):
        def normal_ci(k, n, alpha=0.05):
            """
                Computes confidence interval (1 - alpha) for samples
                generated via naive Monte Carlo sampling.
            """
            p = k / n # mean
            z = norm.ppf(1 - alpha/2) # z-score
            se = np.sqrt(p * (1 - p) / n) # standard error?
            return np.clip(p - z * se, 0, 1), np.clip(p + z * se, 0, 1)
        
        logical_errors = 0
        num_shots = 0

        shots_per_sample = min_shots

        noise_model = SI1000NoiseModel(p)
        circuit: stim.Circuit = get_memory_experiment(
            code=self.code, 
            basis=self.basis, 
            num_rounds=self.distance, 
            noise_model=noise_model).flattened()
        
        dem = circuit.detector_error_model()
        sampler = dem.compile_sampler()
        
        decoder = pymatching.Matching(circuit.detector_error_model())

        failure_sample = None
        while True:
            dets, obs, errors = sampler.sample(shots_per_sample, return_errors=True)

            preds = decoder.decode_batch(dets)
            for shot in range(shots_per_sample):
                if preds[shot] != obs[shot]:
                    if failure_sample is None:
                        failure_sample = self.errors_to_events(circuit, dem, np.flatnonzero(errors[shot]))
                    logical_errors += 1
            num_shots += shots_per_sample

            # Compute confidence interval so far
            low, high = normal_ci(logical_errors, num_shots)
            halfwidth = (high-low) / 2

            # If the statistical error is below a threshold,
            # terminate the MC sampling
            if halfwidth <= epsilon or num_shots >= max_shots:
                return logical_errors, num_shots, failure_sample

    # ---------- High-level run ----------
    def run(self):
        """Run rare-event splitting end-to-end (high-level).

        Returns a dict of results including estimated logical failure rate at
        target_p and the intermediate ratios.
        """
        p0 = self.p0
        pt = self.target_p
        ps = self.splitting_schedule(p0, pt)
        logical_error_rates = []

        # Naive Monte Carlo sampling at the highest physical
        # error rate
        logical_errors, num_shots, failure_sample = self.naive_monte_carlo(p0)
        logical_error_rates.append(logical_errors / num_shots)

        # For the remaining physical error rates, use the splitting
        # technique
        p_curr = p0
        Es_num = self.sample_failures(p_curr,
                                      init_sample=failure_sample)
        for i in range(1, len(ps)):
            p_next = ps[i]
            Es_den = self.sample_failures(p_next,
                                          init_sample=Es_num[-1])

            # Estimate of logical error rate ratio
            C = log_binary_search(Es_num, Es_den, 
                                  lambda E: self._approx_prob_of_set(E, p_curr),
                                  lambda E: self._approx_prob_of_set(E, p_next),
                                  max_iters=10000)

            logical_error_rates.append(C * logical_error_rates[-1])

            p_curr = p_next
            Es_num = Es_den


        return logical_error_rates
        '''
        ratios = []
        for i in range(len(ps) - 1):
            pi = ps[i]
            pin = ps[i + 1]
            # produce samples from pi|F via MCMC
            E_num = self.sample_failures(pi, num_jumps=self.shots_per_chain)
            
            E_den = self.sample_failures(pin, num_jumps=self.shots_per_chain)
            
            def pi_j_func(E):
                return self._approx_prob_of_set(E, pi)
            
            def pi_j_plus_1_func(E):
                return self._approx_prob_of_set(E, pin)
            
            C = log_binary_search(
                E_num,  # samples from π_i|F
                E_den,  # samples from π_{i+1|F}
                lambda E: self._approx_prob_of_set(E, pi),     # π_i(E)
                lambda E: self._approx_prob_of_set(E, pin),    # π_{i+1}(E)
                tolerance=1e-9,
                max_iter=100
            )
            ratio = C  # Use C as the ratio estimate
            ratios.append(ratio)
            
            
            
            # # Search for a value C which expectation at i == expectation at i+1
            # C = log_binary_search(
            #     E_num, 
            #     E_den,
            #     pi_j_func, 
            #     pi_j_plus_1_func,
            #     tolerance=1e-9,
            #     max_iter=100
            # )
        
        
            # # estimate ratio using Bennett-type estimator (g(x) = 1/(1+x))
            # # compute weights w_j = g(C*pi(E)/pi+1(E)) and choose C satisfying eq (4)
            # # For simplicity we search for C by binary search on log-space
            # def estimate_ratio(C: float) -> float:
            #     vals_i = []
            #     vals_in = []
            #     for s in samples:
            #         # compute pi(E) and pi+1(E) approximately via prob_of_set
            #         # reuse approx prob_of_set defined locally
            #         piE = self._approx_prob_of_set(s, pi)
            #         pinE = self._approx_prob_of_set(s, pin)
            #         if piE == 0 or pinE == 0:
            #             continue
            #         x = C * (piE / pinE)
            #         vals_i.append(1.0 / (1.0 + x))
            #     # Similarly we would need samples from pi+1|F; to avoid a nested MCMC
            #     # we use the approximation that samples are similar and estimate the ratio
            #     # using the average of the above as a heuristic. A fully correct
            #     # implementation requires generating samples at pi+1 as well.
            #     if not vals_i:
            #         return 1.0
            #     return sum(vals_i) / len(vals_i)

            # C_est = 1.0
            # ratio = pin / pi  # placeholder; full method requires solving eq (4)
            # ratios.append(ratio)
        # multiply ratios to get final
        overall_ratio = 1.0
        for r in ratios:
            overall_ratio *= r
        # initial logical rate estimate at p0 via naive Monte Carlo
        # For simplicity return placeholder estimates
        return {
            'ps': ps,
            'ratios': ratios,
            'overall_ratio': overall_ratio,
            'estimated_pt': overall_ratio * p0,
        }
        '''

    def _approx_prob_of_set(self, S: Set[GateFault], p: float) -> float:
        # reuse inner logic from metropolis; factorized as separate function
        prod = 1.0

        failing_gates = set()
        for event in S:
            gate, fault = event
            _, _, channel = gate

            failing_gates.add(gate)

            p_g = self.gate_failure_prob[gate] * (p / self.p0)
            p_g_f = 1 / len(channel_faults[channel])

            prod *= (p_g * p_g_f)
        
        nonfailing_gates = set(self.gate_list) - failing_gates
        for g in nonfailing_gates:
            p_g = self.gate_failure_prob[gate] * (p / self.p0)
            prod *= (1 - p_g)

        return prod

# If run as a script, provide a minimal CLI
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--distance", type=int, default=3, help="Distance of the error correction code to simulate")
    parser.add_argument('--p0', type=float, default=1e-3, help="Initial physical error rate for performing Monte Carlo sampling")
    parser.add_argument('--pt', type=float, default=1e-9, help="Lowest physical error rate at which to run rare event simulation")
    parser.add_argument('--shots', type=int, default=200, help="Number of Markov chain samples to generate")
    parser.add_argument('--seed', type=int, default=0, help="Seed for random number generator")

    args = parser.parse_args()

    sim = RareEventSimulator(distance=args.distance, p0=args.p0, target_p=args.pt, 
                             shots_per_chain=args.shots, rng_seed=args.seed)
    # lers = sim.run()
    # print(lers)
    # import json
    # print(json.dumps(out, indent=2))
