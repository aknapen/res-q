import argparse
import json
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex'])
color_cycle = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f032e6']
marker_cycle = [".", "x", "^"]
linestyle_cycle = ["-", "--"]
plt.rcParams["font.family"] = "sans-serif"

def parse_results(distances, setups):
    
    results = {}

    for setup in setups:
        ler_list = []
        per_list = []
        for d in distances:
            lers = []
            pers = []
            fname = f"{setup}/d={d}.json"
            with open(fname, "r") as f:
                run_result = json.load(f)
                pers = []
                lers = []
                for p, l in zip(run_result["physical-error-rates"], run_result["logical-error-rates"]):
                    if l > 1E-7:
                        pers.append(p)
                        lers.append(l)
        
                ler_list.append(lers)
                per_list.append(pers)

        results[setup] = {
            "physical-error-rates": per_list,
            "logical-error-rates": ler_list
        }
    
    return results

def graph_logical_error_rate(new, baseline, distances, outfile, ylabel, plot_labels, series_labels):
    fig, ax = plt.subplots(figsize=(7,4.5))

    for i,d in enumerate(distances):
        pers = new["physical-error-rates"][i]
        lers = new["logical-error-rates"][i]
        ax.plot(pers, lers, marker=".", color=color_cycle[i % len(color_cycle)],
                 linestyle="-", linewidth=2, markersize=12, label=plot_labels[i])
        
        if baseline != None:
            pers = baseline["physical-error-rates"][i]
            lers = baseline["logical-error-rates"][i]
            ax.plot(pers, lers, marker="v", color=color_cycle[i % len(color_cycle)],
                    linestyle="--", linewidth=2, markersize=12)
   
    if series_labels != None:
        # Legend 1: Line styles for series
        series_legend = []
        for i, label in enumerate(series_labels):
            series_legend.append(Line2D([0], [0], color='black', marker=("v" if i == 1 else "."), markersize=12, linestyle=linestyle_cycle[i], linewidth=2, label=label),)
        
        if len(series_labels) > 1:
            # TO SHOW SERIES LABELS OUTSIDE PLOT AREA
            fig.legend(handles=series_legend, fontsize=13, loc='upper center', bbox_to_anchor=(0.6, 1.07), ncol=2, frameon=True)

    ax.grid("both")
    ax.legend(loc="lower right", fontsize=14, frameon=True, fancybox=True)
    ax.tick_params(axis="y", labelsize=16)

    fig.supxlabel("Physical Error Rate", fontsize=18)
    fig.supylabel(ylabel, fontsize=18, x=0.04)

    ax.set_ylim(bottom=1E-9, top=1E-2)
    ax.set_xlim(left=9E-5, right=1.1E-3)
    
    plt.xscale("log")
    plt.yscale("log")

    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)

def main():
    parser = argparse.ArgumentParser()

    # Parse command-line options
    parser.add_argument("-d", "--distances", help="Code distances to be plotted", nargs="*")
    args = parser.parse_args()
    
    # Defaults
    distances = [3,5,7]

    if args.distances:
        distances = [int(d) for d in args.distances]

    for d in distances:
        if d < 0:
            print("[ERROR] Negative code distance specified.")
            exit(1)

    output_dir = f"plots/"
    new = f"results/mcmc/"
    baseline = f"results/mc/"
    ylabel = "Logical Error Rate"
    plot_labels = [f"d={d}" for d in distances]
    series_labels = ["RES", "Monte Carlo"]
    # series_labels = ["RES"]
    outfile = output_dir + f"logical_error_rate_comp.png"
    
    results = parse_results(distances, [new, baseline])

    makedirs(output_dir, exist_ok=True)

    graph_logical_error_rate(results[new],
                            results[baseline],
                            distances, outfile, ylabel, plot_labels, series_labels)
    

if __name__ == "__main__":
    main()