import argparse
import json
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science', 'no-latex'])
color_cycle = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f032e6']
marker_cycle = [".", "x", "^"]
linestyle_cycle = ["-", "--"]
plt.rcParams["font.family"] = "sans-serif"

def parse_results(distances, dir):
    mc_list = []
    markov_list = []
    search_list = []
    total_list = []
    
    for d in distances:
        fname = f"{dir}/d={d}.json"
        with open(fname, "r") as f:
            run_result = json.load(f)
            timing = run_result["timing"]
            
            mc_list.append(timing["monte-carlo"])
            markov_list.append(sum(timing["markov-chain"]))
            search_list.append(sum(timing["binary-search"]))
            total_list.append(timing["total"])

    results = {
        "monte-carlo": mc_list,
        "markov-chain": markov_list,
        "binary-search": search_list,
        "total": total_list
    }
    
    return results

def graph_timing(timing, distances, outfile, ylabel, plot_labels, series_labels):
    fig, ax = plt.subplots(figsize=(5,4.5))

    
    mc = timing["monte-carlo"]
    markov = timing["markov-chain"]
    search = timing["binary-search"]
    total = timing["total"]

    
    ax.plot(distances, mc, marker=".", color=color_cycle[0],
                linestyle="-", linewidth=2, markersize=12, label=plot_labels[0])
    ax.plot(distances, markov, marker=".", color=color_cycle[1],
                linestyle="-", linewidth=2, markersize=12, label=plot_labels[1])
    ax.plot(distances, search, marker=".", color=color_cycle[2],
                linestyle="-", linewidth=2, markersize=12, label=plot_labels[2])
    ax.plot(distances, total, marker=".", color=color_cycle[3],
                linestyle="-", linewidth=2, markersize=12, label=plot_labels[3])
   
    if series_labels != None:
        # Legend 1: Line styles for series
        series_legend = []
        for i, label in enumerate(series_labels):
            series_legend.append(Line2D([0], [0], color='black', linestyle=linestyle_cycle[i], linewidth=2, label=label),)
        
        if len(series_labels) > 1:
            # TO SHOW SERIES LABELS OUTSIDE PLOT AREA
            fig.legend(handles=series_legend, fontsize=13, loc='upper center', bbox_to_anchor=(0.6, 1.07), ncol=2, frameon=True)

    ax.grid("both")
    ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True)
    ax.tick_params(axis="y", labelsize=16)

    fig.supxlabel("Code Distance", fontsize=18)
    fig.supylabel(ylabel, fontsize=18, x=0.04)
    
    # plt.xscale("log")
    # plt.yscale("log")

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
    ylabel = "Runtime (s)"
    plot_labels = ["Monte Carlo", "MCMC", "Binary Search", "Total"]
    outfile = output_dir + f"timing.png"
    
    timing = parse_results(distances, new)

    makedirs(output_dir, exist_ok=True)

    graph_timing(timing, distances, outfile, ylabel, plot_labels, None)
    

if __name__ == "__main__":
    main()