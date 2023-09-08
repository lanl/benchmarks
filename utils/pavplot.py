# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import os.path as op
import sys 

import argparse
import json
import pandas as pd
import subprocess as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

DROP_KEYS=['created','duration','pav_version','test_version',
           'uuid','results_log','finished','job_info','permute_on']

YVAR     = 'iter_t'
XVAR     = 'tpm'
LEGVAR   = ['compiler', 'mpi']
SUBVAR   = None
CORRVAR  = 'inner_n'
TESTCORR = False #True 

TESTPLOT={'partisn': {
            "legendcols": ['compiler','part_n','ncores'],
            "xcol": 'tnodes',
            "ycol": 'iter_t',
            "ylbl": "Iteration Time"
             },
          'parthenon': {
              "legendcols": ['mpi','nx'],
              "xcol": 'tpm',
              "ycol": 'zcycles_sec',
              "xlbl": "Nprocs",
              "ylbl": "Zone Cycles Per Second"
              },
          "branson": {
              "legendcols": ['compiler','mpi','nphotons'],
              "xcol": 'tpm',
              "ycol": 'mphotons_per_second',
              "xlbl": "Nprocs",
              "ylbl": "Photons per second"
              }
          }

def flatten_dict(result, prefix=''):
    aa=dict(); bb=dict()
    for k, v in result.items():
        if not v:
            continue
        kk = prefix+k        
        if not isinstance(v, dict):
            aa[kk] = v
        else:
            bb[kk] = v
            
    return aa, bb

def merge_df_col(df, name1, name2, nameout, sepset):
    df[nameout] = df[name1].str.cat(df[name2].astype(str), sep=sepset)
    return df.drop([name1,name2], axis=1)
    
def correctness_test(df):
    if not all(df['result'] == "PASS"):
        raise ValueError("The not all the tests in this json passed")
    if CORRVAR:
        if not all(df[CORRVAR]==df[CORRVAR].iloc[0]):
            raise ValueError("Some of the result values that should be " +
                             "identical across runs do not match.")
                             
    print("CORRECTNESS TESTS PASS")
    return True

def runcmd(stringcmd):
    wtcmd = sp.Popen(stringcmd, shell=True)
    sp.Popen.wait(wtcmd)

def readjs(f):
    fobj = open(f, 'r')
    fr = fobj.read()
    fobj.close()
    return json.loads(fr)

def makeList(v):
    if isinstance(v, list):
        return v
    else:
        return [v]
        
def args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('json_in', type=str, 
        default='partisn_result.json',
        help="The Json file from pavilion to consume and plot.")
    
    # set CORRVAR/SKIP CORRECTNESS
    
    return parser
    

if __name__ == "__main__":
    arguments = args().parse_args()
    print(arguments)
    jsi = arguments.json_in #'partisn_rslt.json'
    result_dict = readjs(jsi) #readjs(arguments.json_in)
    
    # print(json.dumps(result_dict[0], indent=2))

    result_collection=[]
    for i, result in enumerate(result_dict):
        _ = [result.pop(k) for k in DROP_KEYS if k in result.keys()]
        a_dict, b_dict = flatten_dict(result)
        prefix=''
        while b_dict:
            bd1={}
            if i==0: print(f"PREFIX: {prefix}")
            for bk, bv in b_dict.items():
                if i==0: 
                    print(f"KEY: {bk}")
                    print(f"VAL: {bv}")
                    
                prefixbk = prefix
                if not bk == 'var': 
                    prefixbk=prefix+bk+'.'
                a_dictnew, b_dictnew = flatten_dict(bv, prefixbk)
                a_dict.update(a_dictnew)
                bd1.update(b_dictnew)
                
            b_dict = bd1.copy()
            prefix=prefixbk
        
        result_collection.append(pd.Series(a_dict))s
        
    results_frame_i = pd.DataFrame(result_collection)
    
    if TESTCORR: correctness_test(results_frame_i)
    testn,testsubn = results_frame_i['name'].iloc[0].split('.')[:2]
    sysname = results_frame_i['sys_name'].iloc[0]
    
    results_frame_c = merge_df_col(results_frame_i, 
        'compilers.name', 'compilers.version', "compiler", "/")
    
    results_frame   = merge_df_col(results_frame_c, 
        'mpis.name', 'mpis.version', "mpi", "/")
    
    results_frame.update(results_frame.apply(pd.to_numeric, errors='coerce'))
    
    f, ax=plt.subplots()
    tconf = TESTPLOT[testn]
    plt_legend = tconf['legendcols']
    results_frame.sort_values(by=tconf['xcol'], inplace=True)
    for kf, res in results_frame.groupby(plt_legend):
        str_lbl=[str(k) for k in kf]
        res.plot(ax=ax, x=tconf['xcol'], y=tconf['ycol'], ylabel=tconf['ylbl'], 
                 grid=True, label='-'.join(str_lbl))
        
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1,1.5), 
                       title='-'.join(plt_legend))
    
    f.suptitle(f"{testn} {testsubn} test on {sysname}")
    
    plt.show()
    # for keys, results in results_frame.groupby('nx'):
        
    
        
    
    