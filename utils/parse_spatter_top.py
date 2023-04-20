# Alex Long, along@lanl.gov
# Jered Dominguez-Trujillo, jereddt@lanl.gov

import argparse
import time
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--program', help='Program to sample', required=False, default='spatter')
parser.add_argument('-u', '--user', help='User running program', required=False, default=None)
parser.add_argument('-f', '--outfile', help='Output file', required=False, default='output.txt')
parser.add_argument('-n', '--collections', help='Number of collections', required=False, default=1000)
parser.add_argument('-t', '--interval', help='Interval between collections (seconds)', required=False, default=0.01)
args = vars(parser.parse_args())

program = args['program']
user = args['user']
outfile = args['outfile']
collections = int(args['collections'])
dt = float(args['interval'])

print("Running with:")
print("Program:", program)
if user is not None:
  print("User:", user)
print("Output File:", outfile)
print("Num Collections:", collections)
print("Interval:", dt)
print()

virt = []
res = []
shr = []
times = []

data = []
outputs = [None] * collections


print("Starting top sampling")

for i in range(collections):
  if user is not None:
    outputs[i] = subprocess.Popen(["top", "-b", "-n", "1", "-u", user], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  else:
    outputs[i] = subprocess.Popen(["top", "-b", "-n", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if (i + 1) % ((i + 1) / 10) == 0:
    print("Collection:", i + 1, "of", collections)
  time.sleep(dt)

print("Analyzing Results...")

for i in range(collections):
  out, err = outputs[i].communicate()
  data.append(out)

for out in data:
  as_lines = out.decode().split('\n')

  total_virt = 0.0
  total_res = 0.0
  total_shr = 0.0

  found = False

  # first get the timestamp, just use the first instance of spatter in multicore runs
  for line in as_lines:
    if program in line and line.split()[7] == 'R':
      raw_time = line.split()[10]
      time_split = raw_time.split(":")
      minutes = int(time_split[0])
      seconds = float(time_split[1])
      times.append(minutes*60.0 + seconds)
      break

  # next get each memory amount for each instance of spatter (one per MPI rank)
  for line in as_lines:
    if program in line and line.split()[7] == 'R':
      line = line.split()
      virt_raw = line[4]
      res_raw = line[5]
      shr_raw = line[6]
      if("g" in virt_raw):
        virt_raw = virt_raw.strip("g")
        total_virt = total_virt +  float(virt_raw)*(1024*1024)
      else:
        total_virt = total_virt + int(virt_raw)
      if("g" in res_raw):
        res_raw = res_raw.strip("g")
        total_res = total_res +  float(res_raw)*(1024*1024)
      else:
        total_res = total_res + int(res_raw)
      if("g" in shr_raw):
        shr_raw = shr_raw.strip("g")
        total_shr = total_shr + float(shr_raw)*(1024*1024)
      else:
        total_shr = total_shr + int(shr_raw)
      # reset flag so we keep looking for output from spatter
      found = True

  if (found):
    virt.append(total_virt)
    res.append(total_res)
    shr.append(total_shr)

with open(outfile, "w") as txt_file:
  fmt = "%-8s %-8s %-8s %-8s\n"
  txt_file.write(fmt % ("Time", "Virt", "Res", "Shr"))
  fmt = "%08.5f %08.5f %08.5f %08.5f\n"
  for i in range(0, len(virt)):
    txt_file.write(fmt % (times[i], virt[i] / 1.0e6, res[i] / 1.0e6, shr[i] / 1.0e6))
