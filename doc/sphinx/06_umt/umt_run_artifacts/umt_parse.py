#!/usr/bin/env python3

import pandas as pd
import os

if __name__ == "__main__":
    umtdata = pd.read_csv("umt_summary.csv")
    # Add summary columns
    outfiles = [f for f in os.listdir('.') if f.endswith('.out')]
    umtdata["TotalMemory"] = umtdata["memory"]*umtdata["nprocs"]
    umtdata["PctMemory"] = umtdata["TotalMemory"]/1280
    umtsplit = dict()
    for grp, data in umtdata.groupby('Problem'):
        data.drop('Problem', axis=1, inplace=True)
        data.set_index('nprocs', inplace=True)
        data['Ideal']  = data['single_throughput'][1]*data.index
        umtsplit[grp] = data
        
    for prob, data in umtsplit.items():
        name=f"../roci_spr_Problem{prob}.csv"
        data.to_csv(name)