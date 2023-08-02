# Alex Long, along@lanl.gov

import time
import shutil
import subprocess
import sys
import re

re_top = re.compile(
  "[0-9]+ along \s*[0-9]+ \s*[0-9]+ \s*([0-9]+) \s*([0-9.g]+) \s*([0-9]*) R \s*[0-9.]* \s*([0-9.]*) \s*([0-9:.]*) BRANSON")

virt = []
res = []
shr = []
times = []


print("Listenting to top, waiting for branson to start")

found_branson = False
while not found_branson:
  top_output = subprocess.Popen(["top", "-b", "-n", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  out, err = top_output.communicate()
  as_lines = out.decode().split('\n')
  for line in as_lines:
    if re_top.search(line):
      found_branson= True

# put this in because top doesn't always report Branson as running, I don't know why
time.sleep(1.1)

print("Found BRANSON on top line, starting memory data collection")

while found_branson:
  top_output = subprocess.Popen(["top", "-b", "-n", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  out, err = top_output.communicate()
  as_lines = out.decode().split('\n')
  found_branson = False
  total_virt = 0.0
  total_res = 0.0
  total_shr = 0.0
  # first get the timestamp, just use the first instance of branson in multicore runs
  for line in as_lines:
    if re_top.search(line):
      raw_time = re_top.findall(line)[0][4]
      time_split = raw_time.split(":")
      minutes = int(time_split[0])
      seconds = float(time_split[1])
      times.append(minutes*60.0 + seconds)
      break

  # next get each memory amount for each instance of branson (one per MPI rank)
  for line in as_lines:
    if re_top.search(line):
      virt_raw = re_top.findall(line)[0][0]
      if("g" in virt_raw):
        virt_raw = virt_raw.strip("g")
        total_virt = total_virt +  float(virt_raw)*(1024*1024)
      else:
        total_virt = total_virt + int(virt_raw)
      res_raw = re_top.findall(line)[0][1]
      if("g" in res_raw):
        res_raw = res_raw.strip("g")
        total_res = total_res +  float(res_raw)*(1024*1024)
      else:
        total_res = total_res + int(res_raw)
      total_shr = total_shr + float(re_top.findall(line)[0][2])
      # reset flag so we keep looking for output from BRANSON
      found_branson = True

  if (found_branson):
    virt.append(total_virt)
    res.append(total_res)
    shr.append(total_shr)
  # sleep for a moment so we don't have too much data
  time.sleep(0.3)

with open("output.txt", "w") as txt_file:
  for i,i_virt in enumerate(virt):
    txt_file.write("{0} {1} {2} {3}\n".format(times[i], virt[i]/1.0e6, res[i]/1.0e6, shr[i]/1.0e6))
    #print("{0}  {1}  {2}  {3}".format(times[i], virt[i]/1.0e6, res[i]/1.0e6, shr[i]/1.0e6))
