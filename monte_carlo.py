from res_q import RareEventSimulator
import json

def main():
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
    
    ps = sim.splitting_schedule(sim.p0, sim.target_p)
    lers = []

    print(f"d = {args.distance}")
    for p in ps:
        errors, num_shots, _ = sim.naive_monte_carlo(p)
        lers.append(errors/num_shots)

        print(f"\tp = {p}: LER = {errors / num_shots}, shots = {num_shots}")

    results = {
        "physical-error-rates": ps,
        "logical-error-rates": lers
    }


    with open(f"results/mc/test/d={args.distance}.json", "w") as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    main()
