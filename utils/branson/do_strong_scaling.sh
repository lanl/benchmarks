
input=3D_hohlraum_single_node_cpu.xml
cp ../inputs/3D_hohlraum_single_node.xml ./${input}

sed -i 's/<use_gpu_transporter>TRUE<\/use_gpu_transporter>/<use_gpu_transporter>FALSE<\/use_gpu_transporter>/g' ${input}

for s in 10 66 200
do
  sed -i 's/<photons>.*<\/photons>/<photons>'${s}'000000\<\/photons>/g' ${input}
 #4 12 16 24 32 48  #4 8 16 32 36 64 72
  for i in 8 32 56 88 112
  do
    grep photons ${input}
    #mpirun -mca coll basic,self,libnbc  -n $i ./BRANSON ./3D_hohlraum_single_node_cpu.xml | tee run.$s.$i.out
    srun -n $i -m block:block --hint=nomultithread ./BRANSON ./${input} | tee run.$s.$i.out
  done
done
