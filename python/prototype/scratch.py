#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 18:29:20 2025

@author: rowanlavelle
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
%matplotlib auto

#%%
fpath = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/"
gibbs = []
for i in range(3,8):
    f = fpath + f"gibbs_results_{i}.csv"
    df = pd.read_csv(f)
    df['sn'] = i
    gibbs.append(df)

g = pd.concat(gibbs)

#%%
plt.clf()
bins = np.linspace(0, 10000, 51)
colors = plt.cm.tab10.colors

for i, x in enumerate(g.sn.unique()):
    z = g[g.sn == x]
    color = colors[i % len(colors)]
    
    plt.hist(z.ypred, bins=bins, alpha=0.3, label=x, color=color)
    
    mean_val = z.ypred.mean()
    plt.axvline(mean_val, color=color, linestyle='--', linewidth=1.5)

plt.legend()
plt.xlim(0, 10000)
plt.show()

#%%
fpath = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/"

def read_sim(fpath):
    rows = []
    max_len = 0
    with open(fpath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split(',')]
            rows.append(row)
            max_len = max(max_len, len(row))
    
    # pad with NaN
    padded = [row + [0]*(max_len - len(row)) for row in rows]
    return np.clip(np.array(padded), 0, None)

#%%
tmp = read_sim(fpath+f"sim_results_{7}.csv")

#%%
plt.clf()
indices = np.random.choice(tmp.shape[0], size=2000, replace=False)
for i in indices:
    plt.plot(tmp[i],alpha=0.1,color='tab:blue',lw=0.5)
plt.plot(tmp.mean(axis=0),alpha=1,color='black',lw=2)
plt.show()

#%%
fpath = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/"

def read_sim(fpath):
    rows = []
    max_len = 0
    with open(fpath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split(',')]
            rows.append(row)
            max_len = max(max_len, len(row))
    
    # pad with 0s and clip negatives
    padded = [row + [0]*(max_len - len(row)) for row in rows]
    return np.clip(np.array(padded), 0, None)

# ---
plt.clf()
fig, ax = plt.subplots(figsize=(10,6))

for idx in range(3, 8):  # sim files 3–7
    tmp = read_sim(fpath + f"sim_results_{idx}.csv")
    
    # random sample of sims (150)
    n_sims = min(150, tmp.shape[0])
    indices = np.random.choice(tmp.shape[0], size=n_sims, replace=False)
    
    for i in indices:
        ax.plot(range(idx, idx+tmp.shape[1]), tmp[i], 
                alpha=0.1, color='tab:blue', lw=0.5)
    
    # mean path
    mean_path = tmp.mean(axis=0)
    
    # different color and alpha based on idx
    if idx < 7:
        color = 'dimgray'
        alpha = 0.6
    else:  # idx == 7
        color = 'black'
        alpha = 1.0
    
    ax.plot(range(idx, idx+tmp.shape[1]), mean_path, 
            color=color, lw=2, alpha=alpha, label=f"Mean {idx}")
    
    # mark the start point with a dot
    ax.scatter(idx, mean_path[0], color=color, s=30, zorder=5)
    
    # percentile lines for sim 7
    if idx == 7:
        p5 = np.percentile(tmp, 5, axis=0)
        p95 = np.percentile(tmp, 95, axis=0)
        ax.plot(range(idx, idx+tmp.shape[1]), p5, 
                color='black', linestyle=':', lw=1.5, label="5th percentile")
        ax.plot(range(idx, idx+tmp.shape[1]), p95, 
                color='black', linestyle=':', lw=1.5, label="95th percentile")

ax.set_xlabel("Month")
ax.set_ylabel("Spending")
ax.set_title("Spending Simulations")
plt.show()

#%%
plt.clf()
fig, ax = plt.subplots(figsize=(10,6))

for idx in range(3, 8):  # sim files 3–7
    tmp = read_sim(fpath + f"sim_results_{idx}.csv")
    
    # compute probability of running out of money per month
    p_zero = (tmp == 0).mean(axis=0)
    
    # x-axis shifted by start month
    ax.plot(range(idx, idx+tmp.shape[1]), p_zero, lw=2, label=f"Start {idx}")

ax.set_xlabel("Month")
ax.set_ylabel("Probability of running out of money")
ax.set_title("Survivor / Burnout Plot")
ax.set_ylim(0,1)
ax.legend()
plt.show()

#%%
# --- define months starting from April 2025
start_month = pd.Timestamp("2025-01-01")
month_labels = pd.date_range(start_month, periods=60, freq='MS')  # enough months for longest sims

plt.clf()
fig, axes = plt.subplots(2,1, figsize=(12,10), sharex=True)
ax1, ax2 = axes

# --- Top subplot: simulation paths ---
for idx in range(3, 8):
    tmp = read_sim(fpath + f"sim_results_{idx}.csv")
    
    # random sample of sims (150)
    n_sims = min(150, tmp.shape[0])
    indices = np.random.choice(tmp.shape[0], size=n_sims, replace=False)
    
    for i in indices:
        ax1.plot(range(idx, idx+tmp.shape[1]), tmp[i],
                 alpha=0.1, color='tab:blue', lw=0.5)
    
    mean_path = tmp.mean(axis=0)
    if idx < 7:
        color = 'dimgray'
        alpha = 0.6
    else:
        color = 'black'
        alpha = 1.0
    
    ax1.plot(range(idx, idx+tmp.shape[1]), mean_path,
             color=color, lw=2, alpha=alpha, label=f"Mean start {idx}")
    
    ax1.scatter(idx, mean_path[0], color=color, s=30, zorder=5)
    
    if idx == 7:
        p5 = np.percentile(tmp, 5, axis=0)
        p95 = np.percentile(tmp, 95, axis=0)
        ax1.plot(range(idx, idx+tmp.shape[1]), p5,
                 color='black', linestyle=':', lw=1.5, label="5th percentile")
        ax1.plot(range(idx, idx+tmp.shape[1]), p95,
                 color='black', linestyle=':', lw=1.5, label="95th percentile")

ax1.set_ylabel("Spending ($)")
ax1.set_title("Spending Simulations")
ax1.legend()

# --- Bottom subplot: survivor / probability of having money ---
for idx in range(3, 8):
    tmp = read_sim(fpath + f"sim_results_{idx}.csv")
    
    p_have_money = 1 - (tmp == 0).mean(axis=0)
    ax2.plot(range(idx, idx+tmp.shape[1]), p_have_money, lw=2, label=f"Start {idx}")

ax2.set_ylabel("Probability of having money")
ax2.set_ylim(0,1)
ax2.set_title("Survivor / Burnout Plot")
ax2.legend()

# --- X-axis formatting ---
all_months = month_labels[:max([read_sim(fpath + f"sim_results_{i}.csv").shape[1] + i for i in range(3,8)])]
ax2.set_xticks(range(len(all_months)))
ax2.set_xticklabels([m.strftime("%b %Y") for m in all_months], rotation=45, ha='right')

plt.tight_layout()
plt.show()

#%%
np.log(2500)

#%%
mu = np.log(2500)
var = 0.01
s = np.random.normal(mu, np.sqrt(var), 10000)

#%%
plt.clf()
plt.hist(np.exp(s), bins=150)
plt.show()

#%%
alpha = 2
beta = 0.5
s = np.random.gamma(alpha, beta, 10000)

plt.clf()
plt.hist(s, bins=150)
plt.show()

#%%
theta = np.random.normal(mu, np.sqrt(var))
sigma = np.random.gamma(alpha, beta)
s = np.random.lognormal(theta, sigma, 100)
plt.clf()
plt.hist(s, bins=10)
plt.show()
theta, sigma

#%%
fpath = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/results.csv";
df = pd.read_csv(fpath)

#%%
plt.clf()
plt.hist(df.ypred, bins=150)
plt.show()

#%%
df.head()

#%%
Q = 2
# rent switch in theta func
def sigmoid(x, x0=0, k=1, m=0, L=1.5):
    sig = L/(1+np.exp(-1*k*(x-x0))) - m
    rel = 0.02
    return np.where(x < Q, rel, sig)

#%%
x = np.arange(0,50,1)
v = sigmoid(x,x0=Q, k=1, m=0.5)
plt.clf()
plt.plot(x,v)
plt.show()

#%%
N = 10000
paths = []
for i in range(N):
    tot = 130700#62260
    path = [tot]
    j = 0
    started_rent = False
    while tot > 0:
        if np.random.random() < v[j] or started_rent:
            started_rent = True
            s = np.random.choice(df.ypred.values) + 2300
        else:
            s = np.random.choice(df.ypred.values)
        
        tot -= s
        path.append(tot)
        j += 1
    paths.append(path)
        
#%%
mx = 0
for path in paths:
    if len(path) > mx:
        mx = len(path)

mat = np.zeros((N,mx))

for i,path in enumerate(paths):
    for j in range(len(path)):
        mat[i][j] = path[j]

msk = (mat <= 0).astype(int)

p = msk.sum(axis=0) / N
mat[mat < 0] = 0

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta

# Generate example data if you don't have it
M = mx  # 1000 sims, 100 time steps

# Set up the figure
plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[2, 1])  # Top plot taller

# --- Subplot 1: Cash Trajectories ---
ax1 = plt.subplot(gs[0])
# Plot 50 random trajectories (for clarity)
for i in np.random.choice(N, 500, replace=False):
    ax1.plot(mat[i], color='tab:blue', alpha=0.05, lw=0.5)
    
# Highlight median trajectory
median_traj = np.median(mat, axis=0)
ax1.plot(median_traj, color='black', lw=2, label='Median Cash')
ax1.plot(np.percentile(mat, 5, axis=0), color='black', lw=1, ls='--', alpha=0.5)
ax1.plot(np.percentile(mat, 95, axis=0), color='black', lw=1, ls='--', alpha=0.5)
ax1.axhline(0, color='tab:red', linestyle='--', alpha=0.5, label='Ruin ($0)')
ax1.axhline(15000, color='tab:orange', linestyle='--', alpha=0.5, label='Gaol ($15k)')
ax1.set_ylabel('Cash Balance')
ax1.legend(loc='upper right')
ax1.set_title('Monte Carlo Cash Trajectories')

# --- Subplot 2: Ruin Probability ---
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(p, color='tab:blue', lw=2, label='P(Ruin)', alpha=0.8)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('P(Ruin)')
ax2.legend(loc='upper left')
ax2.set_ylim(0, 1.1)  # Probability bounds

# Style tweaks
plt.suptitle('Burn Rate Analysis: Cash Trajectories & Ruin Probability', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.lines import Line2D

# Configuration
base_path = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/results_{}.csv"
N_SIMULATIONS = 10000
INITIAL_CASH = 130700
RENT_COST = 2300
COLORS = plt.cm.viridis(np.linspace(0, 1, 5))  # Color gradient for each run

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

# Process each results file
for run_idx in range(2, 7):
    # Load data
    df = pd.read_csv(base_path.format(run_idx))
    
    t = 16 - run_idx 
    # rent switch in theta func
    def sigmoid(x, x0=0, k=1, m=0, L=1.5):
        sig = L/(1+np.exp(-1*k*(x-x0))) - m
        rel = np.concatenate([np.arange(0,0.05,(0.05/t)), np.ones(x.shape[0]-t)])
        return np.where(x < t, rel, sig)

    x = np.arange(0,50,1)
    v = sigmoid(x,x0=t, k=1, m=0.5)
    
    # Generate paths
    paths = []
    for _ in range(N_SIMULATIONS):
        cash = INITIAL_CASH
        path = [cash]
        month = 0
        paying_rent = False
        
        while cash > 0:
            # Check rent trigger
            if not paying_rent and np.random.random() < v[month]:
                paying_rent = True
            
            # Get expense
            expense = np.random.choice(df.ypred.values)
            if paying_rent:
                expense += RENT_COST
                
            cash -= expense
            path.append(cash)
            month += 1
        paths.append(path)
    
    # Create matrix
    max_months = max(len(p) for p in paths)
    mat = np.zeros((N_SIMULATIONS, max_months))
    for i, path in enumerate(paths):
        mat[i, :len(path)] = path[:max_months]
    mat[mat < 0] = 0
    
    # Calculate statistics
    msk = (mat <= 0).astype(int)
    p = msk.mean(axis=0)
    n_ruin = msk.sum(axis=0)
    n_survive = N_SIMULATIONS - n_ruin
    
    # Plot trajectories (first run gets full paths, others get median only)
    if run_idx == 2:
        sample_paths = np.random.choice(N_SIMULATIONS, 50, replace=False)
        for i in sample_paths:
            ax1.plot(mat[i], color=COLORS[run_idx-2], alpha=0.08, lw=0.5)
    
    # Plot median path
    median_path = np.median(mat, axis=0)
    ax1.plot(median_path, color=COLORS[run_idx-2], lw=2, 
             label=f'Run {run_idx} (n={len(df)})')
    
    # Plot ruin probability with CI
    ax2.plot(p, color=COLORS[run_idx-2], lw=1.5, alpha=0.7)

# Finalize plot
ax1.axhline(0, color='red', ls='--', alpha=0.7)
ax1.set_ylabel('Cash Balance ($)')
ax1.set_title('Burn Rate Simulations Across Data Increments')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Months')
ax2.set_ylabel('Ruin Probability')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# Create custom legend for probabilities
prob_legend = [Line2D([0], [0], color='gray', lw=2, label='Ruin Probability'),
               Line2D([0], [0], color='gray', alpha=0.1, lw=10, label='95% CI')]
ax2.legend(handles=prob_legend)

plt.tight_layout()
plt.show()