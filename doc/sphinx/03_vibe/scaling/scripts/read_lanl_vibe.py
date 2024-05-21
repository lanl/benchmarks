# %%
import sys
from glob import glob
from matplotlib.style import context

import matplotlib.pyplot as plt

# from cycler import cycler
 
# default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
#                   cycler(linestyle=['-', '--', ':', '-.']))


# sys.path.append("/usr/gapps/spot/dev/hatchet-venv/x86_64/lib/python3.9/site-packages/")
# sys.path.append("/usr/gapps/spot/dev/hatchet/x86_64/")
# sys.path.append("/usr/gapps/spot/dev/thicket-playground-dev/")

import thicket as th

plotx=12
ploty=7

def filter_df(df, category, criteria):
    dfclean    = df.reset_index('profile', drop=True).T
    totalmeans = dfclean.loc[(slice(None), category),:].mean()
    mean_filt = list(totalmeans[totalmeans > criteria].index)
    cols = list(dfclean.columns)
    for code_section in cols:
        if code_section not in mean_filt:
            dfclean.drop(code_section, axis=1, inplace=True)
    
    return dfclean

# %%
tk = th.Thicket.from_caliperreader(glob("VIBE/*.cali"))

# %%
#problem_sizes = list(sorted(tk.metadata["Problem"].unique()))
ranks = list(sorted(tk.metadata["mpi.world.size"].unique()))

gb = tk.groupby("mpi.world.size")
thickets = list(gb.values())
ranks = list(gb.keys())
problem_sizes = [0]
ctk = th.Thicket.concat_thickets(
    thickets=thickets,
    headers=ranks,
    axis="columns"
)
rank_str=[str(r) for r in ranks]
print(ranks)

df_filtered = filter_df(ctk.dataframe, 'Avg time/rank', 3.5)

#quit()
# %%
for r in ranks:
    df_filtered.loc[(r, "perc"),:] = (df_filtered.loc[(r, "Avg time/rank (exc)"),:] / df_filtered.loc[(r, "Avg time/rank (exc)"),:].sum()) * 100

# %%

plt.style.use('fivethirtyeight')
ax = df_filtered.loc[[(r, "perc") for r in ranks],:].reset_index(1, drop=True).plot(
    kind="area",
    title=f"Parthenon-VIBE (Weak Scaling)",
    xlabel="Number of Nodes",
    ylabel="% of Runtime (Exclusive)",
    figsize=(10,5), 
    colormap="tab20"
)
# Custom legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
#print (ctk.dataframe[(slice(None), 1), ("name", "")].tolist())
labels = df_filtered.loc[("name", "")].tolist()

labels.reverse()
print(labels)
for i, label in enumerate(legend.get_texts()):
    label.set_text(labels[i])

f=ax.get_figure()
f.set_size_inches(plotx,ploty)
plt.tight_layout()
plt.savefig(f"parthenon-pct.png", bbox_inches="tight")
#plt.show()

# %%
for p in problem_sizes:
    plt.style.use('fivethirtyeight')
    ax = df_filtered.loc[[(r, "Avg time/rank (exc)") for r in ranks],:].reset_index(1, drop=True).plot(
        kind="line",
        title=f"Parthenon-VIBE (Weak Scaling)",
        xlabel="Number of Nodes",
        ylabel="Runtime (sec) (Exclusive)",
        figsize=(10,5),
        colormap="tab20"
    )
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
    labels = df_filtered.loc[("name", "")].tolist()
    labels.reverse()
    for i, label in enumerate(legend.get_texts()):
        label.set_text(labels[i])
    f=ax.get_figure()
    f.set_size_inches(plotx,ploty)
    plt.tight_layout()
    plt.savefig(f"parthenon-totaltime-line.png", bbox_inches="tight")
    #plt.show()

for p in problem_sizes:
    plt.style.use('fivethirtyeight')
    ax = df_filtered.loc[[(r, "Avg time/rank (exc)") for r in ranks],:].reset_index(1, drop=True).plot(
        kind="area",
        title=f"Parthenon-VIBE (Weak Scaling)",
        xlabel="Number of Nodes",
        ylabel="Runtime (sec) (Exclusive)",
        figsize=(10,5),
        colormap="tab20"
    )
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))#, title='Line', loc='upper left')
    labels = df_filtered.loc[("name", "")].tolist()
    labels.reverse()
    for i, label in enumerate(legend.get_texts()):
        label.set_text(labels[i])

    f=ax.get_figure()
    f.set_size_inches(plotx,ploty)
    plt.tight_layout()
    plt.savefig(f"parthenon-totaltime-area.png", bbox_inches="tight")

#ctk.dataframe.to_csv("out.csv", index=True)
    #print (plt.style.available) 
    #print(plt.colormaps)
