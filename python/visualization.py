import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

root = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/"
fpath = root + "data/"
imgpath = root + "plots/"
min_data = 3

def read_sim(fpath):
    rows = []
    max_len = 0
    with open(fpath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split(',')]
            rows.append(row)
            max_len = max(max_len, len(row))
    
    padded = [row + [0]*(max_len - len(row)) for row in rows]
    return np.clip(np.array(padded), 0, None)

def plot(max_idx):
    start_month = pd.Timestamp("2025-01-01")
    month_labels = pd.date_range(start_month, periods=60, freq='MS')  # enough months

    # --- Read Gibbs data ---
    gibbs = []
    for i in range(min_data, max_idx):
        f = fpath + f"gibbs_results_{i}.csv"
        df = pd.read_csv(f)
        df['sn'] = i
        gibbs.append(df)
    g = pd.concat(gibbs)

    # --- Create figure and GridSpec ---
    fig = plt.figure(figsize=(16,10))
    gs = GridSpec(2,2, width_ratios=[1,1], height_ratios=[1,1], wspace=0.3, hspace=0.3)

    # --- Top-left: Spending sims ---
    ax1 = fig.add_subplot(gs[0,:])
    for idx in range(3,max_idx):
        tmp = read_sim(fpath + f"sim_results_{idx}.csv")
        n_sims = min(150, tmp.shape[0])
        indices = np.random.choice(tmp.shape[0], size=n_sims, replace=False)
        for i in indices:
            ax1.plot(range(idx, idx+tmp.shape[1]), tmp[i],
                    alpha=0.1, color='tab:blue', lw=0.5)
        
        mean_path = tmp.mean(axis=0)
        color = 'dimgray' if idx<max_idx-1 else 'black'
        alpha = 0.6 if idx<max_idx-1 else 1.0
        ax1.plot(range(idx, idx+tmp.shape[1]), mean_path, color=color, lw=2, alpha=alpha)
        ax1.scatter(idx, mean_path[0], color=color, s=30, zorder=5)
        
        if idx==max_idx-1:
            p5 = np.percentile(tmp, 5, axis=0)
            p95 = np.percentile(tmp, 95, axis=0)
            ax1.plot(range(idx, idx+tmp.shape[1]), p5, color='black', linestyle=':', lw=1.5)
            ax1.plot(range(idx, idx+tmp.shape[1]), p95, color='black', linestyle=':', lw=1.5)

    ax1.set_ylabel("Spending ($)")
    ax1.set_title("Spending Simulations")

    # --- Bottom-left: Survivor curve ---
    ax2 = fig.add_subplot(gs[1,0])
    for idx in range(3,max_idx):
        tmp = read_sim(fpath + f"sim_results_{idx}.csv")
        p_have_money = 1 - (tmp==0).mean(axis=0)

        color = 'tab:blue' if idx<max_idx-1 else 'black'
        alpha = idx/(max_idx-1)
        ax2.plot(range(idx, idx+tmp.shape[1]), p_have_money, lw=2, color=color, alpha=alpha)

    flag = np.where(p_have_money < 0.5)[0][0]
    ax2.vlines([flag+max_idx-1], ymin=0,ymax=1,color='tab:red',alpha=0.5,ls='--')
    ax2.set_ylabel("P[$ > 0]")
    ax2.set_ylim(0,1)
    ax2.set_title("Survivor / Burnout Plot")

    # --- Right: Gibbs histogram ---
    ax3 = fig.add_subplot(gs[1,1])
    bins = np.linspace(0, 10000, 51)
    for i, x in enumerate(sorted(g.sn.unique())):
        z = g[g.sn == x]
        color = 'tab:blue' if i<g.sn.unique().shape[0]-1 else 'black'
        alpha = 0.3 if i<max_idx-g.sn.unique().shape[0]-1 else 0.5
        ax3.hist(z.ypred, bins=bins, alpha=alpha, color=color)
        ax3.axvline(z.ypred.mean(), color=color, linestyle='--', linewidth=1, alpha=0.3)

    ax3.set_xlabel("Predicted Spending")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Gibbs Simulation Distribution")
    ax3.set_xlim(0,10000)

    # --- X-axis labels for left plots ---
    all_months = month_labels[:max([read_sim(fpath + f"sim_results_{i}.csv").shape[1] + i for i in range(3,8)])]
    ax1.set_xticks(range(len(all_months)))
    ax1.set_xticklabels([m.strftime("%b %Y") for m in all_months], rotation=45, ha='right')

    ax2.set_xticks(range(len(all_months)))
    ax2.set_xticklabels([m.strftime("%b %Y") for m in all_months], rotation=45, ha='right')

    first_non1 = np.argmax(p_have_money < 1)
    start = max(first_non1 - 1, 0)
    ax2.set_xlim(start+min_data+max_idx-1, None)

    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(imgpath + f"forecast_{max_idx}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesian Gibbs Sampler Burn Rate Projections')
    parser.add_argument('--max_idx', type=int, help='Number of months simulated')
    args = parser.parse_args()
    plot(args.max_idx)