#!/usr/bin/env python3

'''
    Collect memory recorder data, move all data to new folder
    And parse into generally useful, pandas ready, csvs.
'''

import os
import sys

import pandas as pd
import shutil
import argparse

def collect_args():
    parser = argparse.ArgumentParser(
        prog='',
        description='Collect and move the files created by memory recorder.'
        )

    inwd=os.getcwd()
    parser.add_argument('-i', '--input_dir', type=str, default=inwd,
        help="Result directory holding memrecorder results.")
    parser.add_argument('-o', '--output_dir', type=str, default=None,
        help="Output directory for memrecorder results and summary. Default is input_dir")

    return parser

def read_rss_file(file):
    filename = os.path.basename(file)
    rel_node, name_node = filename.split('.')[0].split('_')[1:]
    pctrss = None
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("Fraction"):
                pctrss = line.split()[-1]
                break

    return rel_node, name_node, pctrss

if __name__ == "__main__":

    args = collect_args().parse_args()
    memrec_dir = args.input_dir
    outdir = args.output_dir

    rssout = 'rssfractions'
    rawmemout = 'memraw'
    pctmemout = 'mempct'
    outfiles = [rssout, rawmemout, pctmemout]

    meminfo_pct = {}
    meminfo_raw = {}
    nodelist = {}
    rssraw = []
    nfiles = 0
    memrec_ls = os.listdir(memrec_dir)

    for csvfile in memrec_ls:
        relpath = os.path.join(memrec_dir, csvfile)

        if any([csvfile.startswith(k) for k in outfiles]):
            os.remove(relpath)

        elif csvfile.endswith('.csv'):
            filestem = csvfile.split(".")[0]
            fileids  = filestem.split("_")[1:]
            nodenum = int(fileids[0])
            nodelist[nodenum] = fileids[1]
            if csvfile.startswith("pct"):
                meminfo_pct[nodenum] = pd.read_csv(relpath, index_col=0)
            if csvfile.startswith("meminfo"):
                meminfo_raw[nodenum] = pd.read_csv(relpath, index_col=0)

        elif csvfile.endswith('.memout'):
            rssraw.append(read_rss_file(relpath))

        else:
            continue

        if outdir:
            if nfiles == 0:
                os.makedirs(outdir, exist_ok=True)
            nfiles += 1
            outfile = os.path.join(outdir, csvfile)
            shutil.move(relpath, outfile)

    if nfiles == 0:
        print("COLLECT RESULTS: NO FILES TO MOVE OR CREATE. EXITING.")
        sys.exit(0)
    else:
        print(f"MOVED {nfiles} FILES.")

    # Set outdir to make summary files.
    if not outdir:
        outdir = memrec_dir

    # Add summary subfolder to outdir.
    outdir = os.path.join(outdir, "summary")

    if rssraw:
        os.makedirs(outdir, exist_ok=True)

        # Collect and write out rss info
        outfile = os.path.join(outdir,rssout+".csv")
        rssframe = pd.DataFrame(rssraw, columns=['NodeNum',"NodeName","RamFraction"]).set_index('NodeNum')
        rssframe.sort_values("RamFraction").to_csv(outfile)

    if not meminfo_pct or not meminfo_raw:
        print("No meminfo files. RSS summarized.")
        sys.exit(0)

    os.makedirs(outdir, exist_ok=True)

    # Handle meminfo outputs. Concatenate the dataframes.
    codeloc = 'code_location'
    idx = pd.IndexSlice
    pct_df = pd.concat(meminfo_pct)
    raw_df = pd.concat(meminfo_raw)
    pct_df.index.rename("NodeNum", level=0, inplace=True)
    raw_df.index.rename("NodeNum", level=0, inplace=True)

    # Get node name, relative node number, free mem pct, and code location
    # for global minimum free RAM percentage, and write out to file
    min_total_pct = pct_df.sort_values("Total").groupby(codeloc).head(1)
    minloc = min_total_pct['Total'].idxmin() # Index of global min
    mintotal_global = min_total_pct.loc[minloc]['Total'] # Value of global min
    minline=[mintotal_global] + list(minloc) + [nodelist[minloc[0]]]
    minoutstr = ",".join([str(m) for m in minline]) #Stringify
    outfile = os.path.join(outdir, f"{pctmemout}_minline")
    with open(outfile, 'w') as wo:
        wo.write(minoutstr)

    # Get the mean of each code location
    pct_means = pct_df.groupby(codeloc).mean()
    raw_means = raw_df.groupby(codeloc).mean()

    for g, d in pct_df.groupby(codeloc):
        outfile = os.path.join(outdir, f"{pctmemout}_{g}.csv")
        ddrop = d.droplevel(codeloc)
        ddrop.sort_values("Total").to_csv(outfile)

    for g, d in raw_df.groupby(codeloc):
        outfile = os.path.join(outdir, f"{rawmemout}_{g}.csv")
        ddrop = d.droplevel(codeloc)
        ddrop.sort_values("Total").to_csv(outfile)

    #Find the code location with the minimum mean free RAM percentage.

    minfree_loc = pct_means['Total'].idxmin()
    sorted_min_pct_loc = pct_df.loc[idx[:,minfree_loc],:].sort_values("Total").droplevel(codeloc)
    outfile = os.path.join(outdir, f"minloc_{pctmemout}_{minfree_loc}.csv")
    sorted_min_pct_loc.to_csv(outfile)

