#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import glob
from matplotlib import pyplot
import numpy as  np
from functools import reduce


# In[2]:


reAt = re.compile(' (\s\S+?) atoms')
rePe = re.compile('Performance:(\s\S+?) ns\/day,(\s\S+?) hours\/ns,(\s\S+?) timesteps\/s,(.*?)atom')
reMPI = re.compile('(\s\S+?) by(\s\S+?) by(\s\S+?) MPI')

def readLog(fileName):
    txt = open(fileName,'r').read()
    nAt = reAt.findall(txt)[-1]
    Per = rePe.findall(txt)[4]
    MPI = reMPI.findall(txt)[1]
    nGPU = int(MPI[0])*int(MPI[1])*int(MPI[2])
    if Per[3][-1] =='k':
        stepS = float(Per[3][:-1])*1000
    elif Per[3][-1]=='M':
        stepS = float(Per[3][:-1])*1000000
    else:
        raise RuntimeError('Unkown atom postfix')
    outList = [nGPU,int(nAt),float(Per[0]),float(Per[1]),float(Per[2]),stepS]
    return(outList)


# In[3]:


StrongSingle = []
for curF in glob.glob('Strong_Single_*.out'):
    try:
        curD = readLog(curF)
    except:
        print(curF)
    StrongSingle.append([curD[1],curD[2],curD[5]])
StrongSingle = np.array(StrongSingle).T
StrongSingle = StrongSingle[:,StrongSingle[0].argsort()]


# In[4]:


fig= pyplot.figure(figsize=(4,3),dpi=1200)
pyplot.scatter(StrongSingle[0],StrongSingle[2],color='k')
pyplot.xlabel('# Atoms')
pyplot.ylabel('grads/sec')
pyplot.suptitle('Throughput on 1 GPU')
pyplot.tight_layout()
fig.savefig('StrongSingle-t.png',dpi=1200)
fig.savefig('StrongSingle-t.pdf',dpi=1200)


# In[5]:


fig= pyplot.figure(figsize=(4,3),dpi=1200)
pyplot.scatter(StrongSingle[0],StrongSingle[1],color='k')
pyplot.xlabel('# Atoms')
pyplot.ylabel('ns/day')
pyplot.suptitle('Speed on 1 GPU')
pyplot.tight_layout()
fig.savefig('StrongSingle-s.png',dpi=1200)
fig.savefig('StrongSingle-s.pdf',dpi=1200)


# In[10]:


StrongParallel = []
for curF in glob.glob('Strong_Parallel_*.out'):
    try:
        curD = readLog(curF)
    except:
        print(curF)
    StrongParallel.append([curD[1],curD[2],curD[5]])
StrongParallel = np.array(StrongParallel).T
StrongParallel = StrongParallel[:,StrongParallel[0].argsort()]


# In[7]:


WeakParallel = []
for curF in glob.glob('Weak_Parallel_*.out'):
    try:
        curD = readLog(curF)
    except:
        print(curF)
    WeakParallel.append([curD[0],curD[2],curD[5]])
WeakParallel = np.array(WeakParallel).T
WeakParallel = WeakParallel[:,WeakParallel[0].argsort()]


# In[9]:


fig= pyplot.figure(figsize=(4,3),dpi=1200)
pyplot.scatter(WeakParallel[0],WeakParallel[1],color='k')
pyplot.xlabel('# of GPUs (A-100)')
pyplot.ylabel('ns/day')
pyplot.suptitle('Speed with 85200 atoms per GPU')
pyplot.tight_layout()
fig.savefig('WeakParallel-s.png',dpi=1200)
fig.savefig('WeakParallel-s.pdf',dpi=1200)




