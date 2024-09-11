#!/usr/bin/env python3

## TAKES ONE ARGUMENT: PATH TO DIR WITH CALIPER FILES RELATIVE OR ABSOLUTE
# WILL CREATE A DIRECTORY ADJACENT TO ARGUMENT DIR WITH THE NAME OF THAT DIR + _plots 
## REQUIRES THICKET:
# pip install llnl-thicket
# IF THEY'RE IN A NON_STANDARD PLACE, MAKE SURE TO ADD THEM TO PYTHONPATH ON CL.
# IF RUNNING IN IPYTHON, JUPYTER, ETC, import sys AND APPEND THEIR LOC TO sys.path BEFORE LAUNCHING.


import sys
import os
import os.path as op
from glob import glob
# from matplotlib.style import context
import matplotlib.pyplot as plt
import thicket as th

# from cycler import cycler
 
# default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
#                   cycler(linestyle=['-', '--', ':', '-.']))

# sys.path.append("/usr/gapps/spot/dev/hatchet-venv/x86_64/lib/python3.9/site-packages/")
# sys.path.append("/usr/gapps/spot/dev/hatchet/x86_64/")
# sys.path.append("/usr/gapps/spot/dev/thicket-playground-dev/")

### CONFIG
plotx=12
ploty=7
cat_to_plot=10 #Number of categories to plot.

### FILTER DF
def filter_df(df, category, ncat=10, criteria=None):
    dfclean    = df.reset_index('profile', drop=True).T
    totalmeans = dfclean.loc[(slice(None), category),:].mean().sort_values(ascending=False)
    
    # If given a criteria (a minimum value cutoff), use that instead
    if criteria:
        mean_filt = list(totalmeans[totalmeans > criteria].index)
    else:
        mean_filt = list(totalmeans.head(ncat).index)

    cols = list(dfclean.columns)
    for code_section in cols:
        if code_section not in mean_filt:
            dfclean.drop(code_section, axis=1, inplace=True)
    
    return dfclean

def get_cali_files(rawpath):
    cali_path = op.realpath(rawpath)
    califiles = glob(cali_path + "/*.cali")

    if not califiles:
        print(f"\nERROR:\n    Path {cali_path} has no caliper files.\n")
        sys.exit(1)

    cali_name = op.basename(cali_path)
    cali_dirname = op.dirname(cali_path)
    return cali_name, cali_dirname, califiles
    

if __name__ == "__main__":

    # %%
    if len(sys.argv) < 2:
        error="\nERROR:\n  TAKES ONE ARGUMENT: PATH TO DIR WITH CALIPER FILES RELATIVE OR ABSOLUTE\n"
        error+="  WILL CREATE A DIRECTORY ADJACENT TO ARGUMENT DIR WITH THE NAME OF THAT DIR + _plots\n"
        print(error)
        sys.exit(1)
    
    cali_name, cali_dirname, cali_files = get_cali_files(sys.argv[1])

    plt_outdir = op.join(cali_dirname, cali_name+"_plots")
    os.makedirs(plt_outdir, exist_ok=True)

    tk = th.Thicket.from_caliperreader(cali_files)

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
    # print(ranks)

    df_filtered = filter_df(ctk.dataframe, 'Avg time/rank', cat_to_plot)

    # %%
    for r in ranks:
        df_filtered.loc[(r, "perc"),:] = (df_filtered.loc[(r, "Avg time/rank (exc)"),:] / df_filtered.loc[(r, "Avg time/rank (exc)"),:].sum()) * 100

    '''
    PLOT PCT
    '''
    fig_path = op.join(plt_outdir, "parthenon-pct.png")
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
    plt.savefig(fig_path, bbox_inches="tight")
    #plt.show()

    '''
    PLOT TOTAL TIME LINES
    '''
    for p in problem_sizes:
        fig_path = op.join(plt_outdir, f"parthenon-totaltime-line_{p}.png")
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
        plt.savefig(fig_path, bbox_inches="tight")
        #plt.show()

    '''
    PLOT TOTAL TIME AREA
    '''
    for p in problem_sizes:
        fig_path = op.join(plt_outdir, f"parthenon-totaltime-area_{p}.png")
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
        plt.savefig(fig_path, bbox_inches="tight")

    #ctk.dataframe.to_csv("out.csv", index=True)
        #print (plt.style.available) 
        #print(plt.colormaps)
