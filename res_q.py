import numpy as np
from scipy.stats import norm
from time import time 
import json

from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple, Set, Dict, Optional

import stim
import pymatching

from qldpc.circuits.noise_model import SI1000NoiseModel, NoiseRule
from qldpc.objects import Pauli, PauliXZ
from qldpc.circuits import get_memory_experiment
from qldpc.codes.quantum import CSSCode, SurfaceCode
from qldpc.circuits.bookkeeping import QubitIDs

from util import log_binary_search

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
    "Z": [("Z",)],
    "MX": [("Z",)],
    "DEPOLARIZE1": [("Z",), ("X",), ("Y",)],
    "IDLE": [("Z",), ("X",), ("Y",)],
    "MEAS_RESET_IDLE": [("Z",), ("X",), ("Y",)],
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
    jumps_per_chain: int = 200  # N in the paper for expectation estimates
    num_warmup_steps: int = 10000
    basis: PauliXZ = Pauli.X

    # internal fields
    circuit: stim.Circuit = field(default=None, init=False)
    dem: stim.DetectorErrorModel = field(default=None, init=False)
    # List of all unique gates in the circuit
    gate_list: List[Gate] = field(default_factory=list, init=False)
    # Mapping from a group of qubits to all the gates that act on that group of qubits
    targets_to_gates: Dict[tuple, list] = field(default_factory=dict, init=False)
    # Mapping from a gate to the probability that it can fail for any reason
    gate_failure_prob: Dict[Gate, float] = field(default_factory=dict, init=False)
    # Mapping from a (gate, fault) pair to the conditional probability that, given
    # the gate failed, it failed with that particular fault
    gate_fault_prob: Dict[GateFault, float] = field(default_factory=dict, init=False)
    # Mapping from a S, a set of (gate, fault) pairs, and two physical error rates, p1 and p2,
    # to the ratio Pr(S @ p1) / Pr(S @ p2)
    event_set_prob: Dict[Tuple[Set[GateFault], float, float], float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.code: CSSCode = SurfaceCode(rows=self.distance, cols=self.distance, rotated=True)
        self.noise_model = SI1000NoiseModel(self.p0)
        
        self.circuit: stim.Circuit = self.noise_model.noisy_circuit(
            get_memory_experiment(
                self.code,
                basis=self.basis,
                num_rounds=self.distance,
                noise_model=self.noise_model
            ).without_noise().flattened()
        )

        self.decoder = pymatching.Matching(
            self.circuit.detector_error_model()
        )

        self._catalog_gate_faults()
        
        # Create conditional fault probabilities: P(fault | gate failed)
        self.gate_fault_prob = {}
        for gate in self.gate_list:
            (_, _, channel) = gate
            prob_fault_given_failure = 1.0 / len(channel_faults[channel])

            # For each possible fault for that gate, store the conditional
            # probability
            for fault in channel_faults:
                self.gate_fault_prob[(gate, fault)] = prob_fault_given_failure
            
        # For use in memoization for _event_set_prob_ratio()
        self.event_set_prob = {}
        
    def _catalog_gate_faults(self):
        """Catalog all possible failing gates in a Stim circuit.

        Gates are cataloged uniquely based on their location within the Stim
        circuit (instruction index) as well as the qubits they operate on (to
        differentiate the same instruction action on multiple (sets) of qubits).
        """
        active_qubits: set[int] = set()
        measure_or_reset_in_moment = False
        tick_offset = 0

        instr: stim.CircuitInstruction
        for (i, instr) in enumerate(self.circuit.without_noise().flattened()):
            # A TICK instruction indicates a new "moment" (timeslice) in the quantum circuit
            # so we need to re-evaluate the sets of active and idle qubits
            index = i
            if instr.name == "TICK":
                tick_offset += 1
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
                        index = i - 1 # noise for measurements goes before the measurement
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
                    

                    gate: Gate = (index, gate_targets, gate_channel)
                    self.gate_list.append(gate)
                    # Keeps track of different offsets/gate indices at which gate_targets can be found
                    self.targets_to_gates.setdefault(gate_targets, []).append((tick_offset, index))
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

        # Avoid undershooting target physical error rate
        if ps[-1] < pt:
            ps[-1] = pt
        
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

            for i, f in enumerate(fault):
                if f != "I":
                    target_circuit.insert(
                        gate_index+1+index_offset, 
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

    def metropolis_step(self, current: Set[GateFault], p: float) -> Set[GateFault] | None:
        """Perform one Metropolis step modifying a single gate's fault as in paper.

        Returns the new set (or None if rejected).
        """
        # pick a gate uniformly among catalog and a fault uniformly at random
        # for that gate
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
            
            # Add in the new gate, fault pair and remove the already existing 
            # pair with the same gate (may remove the gate, fault pair entirely
            # if the already existing pair has the same fault as the one being added)
            new = (current | {event}) - {current_event}
            
            (_, current_fault) = current_event

            # If the gate, fault pair was entirely removed, always accept the change
            if fault == current_fault:
                acceptance_probability = 1
                accept = 1
            else:
                acceptance_probability = prob_g_f
                accept = np.random.random() <= acceptance_probability

        else: # selected gate is not already in the error set
            new = current | {event}

            acceptance_probability = ((prob_g / (1 - prob_g)) * prob_g_f)

        accept = np.random.random() <= acceptance_probability

        # Only consider acceptance if new causes a logical failure
        if accept and self.is_malicious(new):    
            return new
        else:
            return None

    def sample_failures(self, p: float, 
                        init_sample: Set[GateFault] = set()) -> List[Set[GateFault]]:
        """Produce samples from pi|F via MCMC (Metropolis) as in the paper.

        This returns a list of distinct failing events discovered by the chain.
        """
        # seed initial failing events via direct Monte Carlo (or heuristic)
        chain_state = init_sample
        
        num_steps = 0
        # Warm up chain to converge to stable distribution
        while num_steps < self.num_warmup_steps:
            new_state = self.metropolis_step(chain_state, p)
            if new_state is not None:
                chain_state = new_state
            num_steps += 1

        # Keep extending the chain until a threshold number
        # of state changes occurs
        num_jumps = 0
        discovered = []
        while num_jumps < self.jumps_per_chain:
            new_state = self.metropolis_step(chain_state, p)
            if new_state is not None:
                chain_state = new_state
                num_jumps += 1
            discovered.append(frozenset(chain_state))

        return discovered

    def errors_to_events(self, circuit: stim.Circuit, 
                         dem: stim.DetectorErrorModel,
                         errors: np.ndarray) -> Set[GateFault]:
        ''' Translates an array of Stim errors into a set of (gate, fault) pairs.'''

        event_set: Set[GateFault] = set()
        flat_errors = [
            instr for instr in dem.flattened() if instr.type == "error"
        ]

        # Loop through all triggered errors in the DEM and parse corresponding
        # circuit-level information (e.g., flipped qubits, time of the error)
        dem_filter = stim.DetectorErrorModel()
        for e in errors:
            error = flat_errors[e]
            dem_filter.clear()
            dem_filter.append(error)
            expl = circuit.explain_detector_error_model_errors(
                dem_filter=dem_filter, 
                reduce_to_one_representative_error=True
            )[0]

            loc: stim.CircuitErrorLocation = expl.circuit_error_locations[0]
            frame: stim.CircuitErrorLocationStackFrame = loc.stack_frames[0]
            
            qubits = [target.gate_target.value for target in 
                            loc.instruction_targets.targets_in_range]

            
            # Determine which qubits experienced Pauli errors and what types
            # of Pauli errors
            flipped_qubits = []
            flips = []
            # Flip of a qubit
            if len(loc.flipped_pauli_product) != 0:
                flipped_qubits = [target.gate_target.value 
                                  for target in loc.flipped_pauli_product]
                flips = [target.gate_target.pauli_type for target in
                                    loc.flipped_pauli_product]
            # Flip of a measurement value
            else:
                flipped_qubits = [target.gate_target.value 
                                  for target in loc.flipped_measurement.observable]
                
                for target in loc.flipped_measurement.observable:
                    # Pauli flip is the opposite of the measurement basis
                    if target.gate_target.pauli_type in ["Z", "Y"]:
                        flips.append("X")
                    else:
                        flips.append("Z")
            
            faults = []
            for qubit in qubits:
                if qubit in flipped_qubits:
                    faults.append(
                        flips[flipped_qubits.index(qubit)]
                    )
                else:
                    faults.append("I")
            
            channel = circuit[frame.instruction_offset].name.strip("_ERROR")
            # Measurement error channel Pauli type is opposite of measurement basis
            if circuit[frame.instruction_offset].name == "MX":
                channel = "Z"
            elif circuit[frame.instruction_offset].name == "M":
                channel = "X"
            
            tick_offsets, gate_indices = zip(*self.targets_to_gates[tuple(qubits)])

            # A bit hacky, but since instruction indices get offset when adding
            # in noise channels, we need a way to go from the noise channel indices
            # in the Stim circuit to the gate indices in our RES. To do so, we've kept
            # a record of all tick offsets and gate indices at which a particular set of 
            # gate targets can be found. Then, using the noise channel tick offset, we 
            # choose the gate index for the given set of gate targets whose tick offset 
            # matches that of the noise channel tick offset
            for i, tick_offset in enumerate(tick_offsets):
                if tick_offset == loc.tick_offset:
                    index = gate_indices[i]
                    break
            
            gate: Gate = (index, tuple(qubits), channel)
            event_set.add((gate, tuple(faults)))
        
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
        circuit: stim.Circuit = noise_model.noisy_circuit(
            get_memory_experiment(
                self.code,
                basis=self.basis,
                num_rounds=self.distance,
                noise_model=noise_model
            ).without_noise().flattened()
        )
        
        dem = circuit.detector_error_model()
        sampler = dem.compile_sampler()
        
        decoder = pymatching.Matching(circuit.detector_error_model())

        failure_sample = None
        while True:
            dets, obs, errors = sampler.sample(shots_per_sample, return_errors=True)

            preds = decoder.decode_batch(dets)
            for shot in range(shots_per_sample):
                if preds[shot] != obs[shot]:
                    logical_errors += 1

                    # Record a failure sample to initialize the first
                    # Markov chain
                    if failure_sample is None:
                        failure_sample = self.errors_to_events(circuit, 
                                                               dem, 
                                                               np.flatnonzero(errors[shot]))
                    
            num_shots += shots_per_sample

            # If the statistical error is below a threshold,
            # terminate the MC sampling
            if logical_errors > 100 or num_shots >= max_shots:
                return logical_errors, num_shots, failure_sample

    # ---------- High-level run ----------
    def run(self):
        """Run rare-event splitting end-to-end (high-level).

        Returns a dict of results including estimated logical failure rate at
        target_p and the intermediate ratios.
        """
        results = {}
        results["timing"] = {}

        total_start = time()
        p0 = self.p0
        pt = self.target_p
        ps = self.splitting_schedule(p0, pt)
        logical_error_rates = []

        # Naive Monte Carlo sampling at the highest physical
        # error rate
        start = time()
        num_logical_errors, num_shots, failure_sample = self.naive_monte_carlo(p0)
        end = time()

        results["timing"]["monte-carlo"] = end - start

        logical_error_rates.append(num_logical_errors / num_shots)

        # For the remaining physical error rates, use the splitting
        # technique
        p_curr = p0
        start = time()
        Es_num = self.sample_failures(p_curr,
                                      init_sample=failure_sample)
        end = time()

        results["timing"]["markov-chain"] = [end - start]
        results["timing"]["binary-search"] = []
        results["C"] = []

        for i in range(1, len(ps)):
            p_next = ps[i]

            start = time()
            Es_den = self.sample_failures(p_next,
                                          init_sample=Es_num[-1])
            end = time()

            results["timing"]["markov-chain"].append(end-start)

            start = time()
            # Estimate of logical error rate ratio
            C = log_binary_search(Es_num, Es_den, 
                                  lambda E: self._event_set_prob_ratio(E, p_curr, p_next),
                                  lambda E: self._event_set_prob_ratio(E, p_next, p_curr),
                                  max_iters=10000)
            end = time()
            
            results["timing"]["binary-search"].append(end-start)
            results["C"].append(C)

            logical_error_rates.append(C * logical_error_rates[-1])

            p_curr = p_next
            Es_num = Es_den

        results["physical-error-rates"] = ps
        results["logical-error-rates"] = logical_error_rates

        total_end = time()

        results["timing"]["total"] = total_end - total_start

        return results

    def _event_set_prob_ratio(self, S: Set[GateFault], p1: float, p2: float) -> float:
        '''
            Computes the ratio of the occurrence probability for a set S of (gate, fault) pairs
            at two different physical error rates, p1 and p2. The occurrence probability for the
            set S is calculated as the product of (1) the probability that the gates in S failed
            and (2) the probability that the gates not in S didn't fail.

            This function uses two tricks to speed up execution. First, since we're computing ratios,
            the ratio of the probabilities of each gate in S failing reduces to (p1 / p2)^N where N
            is the number of failing gates. Second, and much more significantly, since many steps in
            the Markov chain don't actually alter the chain state, there will be many repeated sets 
            of (gate, fault) pairs. Hence, we use memoization to store results for a given set S and
            physical error rates p1 and p2 and attempt to look up the result before trying to do the
            full computation.
        '''
        
        prob = self.event_set_prob.get((S, p1, p2))
        if prob is not None:
            return prob

        prod = 1.0
        r1 = p1 / self.p0
        r2 = p2 / self.p0

        # (1) Accumulate the probabilties that the failing gates would
        # have failed
        prod *= (r1 / r2) ** len(S)
        
        # (2) Accumulate the probabilities that the other gates in the
        # circuit didn't fail
        nonfailing_gates = set(self.gate_list) - set(gate for gate, _ in S)
        for gate in nonfailing_gates:
            pg = self.gate_failure_prob[gate]
            prod *= (1 - (pg * r1)) / (1 - (pg * r2))

        # Store the computed result for lookup in subsequent calls
        self.event_set_prob[(S, p1, p2)] = prod

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
                             jumps_per_chain=args.shots)
    
    results = sim.run()

    with open(f"results/mcmc/d={args.distance}.json", "w") as f:
        json.dump(results, f, indent=4)
