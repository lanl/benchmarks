# %%
import sys
from glob import glob

import matplotlib.pyplot as plt
from cycler import cycler

default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
                  cycler(linestyle=['-', '--', ':', '-.']))

plt.rc('axes', prop_cycle=default_cycler)

# sys.path.append("/usr/gapps/spot/dev/hatchet-venv/x86_64/lib/python3.9/site-packages/")
# sys.path.append("/usr/gapps/spot/dev/hatchet/x86_64/")
# sys.path.append("/usr/gapps/spot/dev/thicket-playground-dev/")

import thicket as th

# %%
tk = th.Thicket.from_caliperreader(glob("AMG/*.cali"))

# %%
problem_sizes = list(sorted(tk.metadata["Problem"].unique()))
ranks = list(sorted(tk.metadata["mpi.world.size"].unique()))

gb = tk.groupby("mpi.world.size")
thickets = list(gb.values())
ctk = th.Thicket.concat_thickets(
    thickets=thickets,
    headers=list(gb.keys()),
    axis="columns",
    metadata_key="Problem",
)

# %%
for p in problem_sizes:
    for r in ranks:
        ctk.dataframe.loc[(slice(None), p), (r, "perc")] = (ctk.dataframe.loc[(slice(None), p), (r, "Avg time/rank (exc)")] / ctk.dataframe.loc[(slice(None), p), (r, "Avg time/rank (exc)")].sum()) * 100

# %%
for p in problem_sizes:
    ax = ctk.dataframe.loc[(slice(None), p), [(r, "perc") for r in ranks]].T.reset_index(1, drop=True).plot(
        kind="area",
        title=f"AMG2023 Problem {p} (Weak Scaling)",
        xlabel="Number of MPI ranks",
        ylabel="% of Runtime",
        figsize=(10,5)
    )
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
    labels = ctk.dataframe.loc[(slice(None), 1), ("name", "")].tolist()
    labels.reverse()
    for i, label in enumerate(legend.get_texts()):
        label.set_text(labels[i])
    # Set custom x labels
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([xlabels[0]] + ranks + [xlabels[-1]])
    plt.savefig(f"prob{p}-pct.png", bbox_inches="tight")
    #plt.show()

# %%
for p in problem_sizes:
    ax = ctk.dataframe.loc[(slice(None), p), [(r, "Avg time/rank") for r in ranks]].T.reset_index(1, drop=True).plot(
        kind="area",
        title=f"AMG2023 Problem {p} (Weak Scaling)",
        xlabel="Number of MPI ranks",
        ylabel="Runtime (sec)",
        figsize=(10,5)
    )
    ax.set_prop_cycle(default_cycler)
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
    labels = ctk.dataframe.loc[(slice(None), 1), ("name", "")].tolist()
    labels.reverse()
    for i, label in enumerate(legend.get_texts()):
        label.set_text(labels[i])
    # Set custom x labels
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([xlabels[0]] + ranks + [xlabels[-1]])
    plt.savefig(f"prob{p}-totaltime.png", bbox_inches="tight")
    #plt.show()

# %%
for p in problem_sizes:

    ax = ctk.dataframe.loc[(slice(None), p), [(r, "Avg time/rank") for r in ranks]].T.reset_index(1, drop=True).plot(
        kind="line",
        title=f"AMG2023 Problem {p} (Weak Scaling)",
        xlabel="Number of MPI ranks",
        ylabel="Runtime (sec)",
        figsize=(10,5)
    )
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
    labels = ctk.dataframe.loc[(slice(None), 1), ("name", "")].tolist()
    labels.reverse()
    for i, label in enumerate(legend.get_texts()):
        label.set_text(labels[i])
    # Set custom x labels
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([xlabels[0]] + ranks + [xlabels[-1]])
    ax.set_prop_cycle(default_cycler)
    plt.savefig(f"prob{p}-time.png", bbox_inches="tight")
    #plt.show()
