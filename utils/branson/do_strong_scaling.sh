export CALI_CONFIG=sample-report

cp ../inputs/3D_hohlraum_single_node_cpu.xml . 
for s in 10 66 200
do
  sed -i 's/<photons>.*<\/photons>/<photons>'${s}'000000\<\/photons>/g' 3D_hohlraum_single_node_cpu.xml 
 #4 12 16 24 32 48  #4 8 16 32 36 64 72
  for i in 4 12 16 24 32 48 96 192
  do
    grep photons 3D_hohlraum_single_node_cpu.xml 
    #mpirun -mca coll basic,self,libnbc  -n $i ./BRANSON ./3D_hohlraum_single_node_cpu.xml | tee run.$s.$i.out
    srun -n $i -m block:block --hint=nomultithread ./BRANSON ./3D_hohlraum_single_node_cpu.xml | tee run.$s.$i.out
  done
done
