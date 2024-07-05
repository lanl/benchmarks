cp ../inputs/3D_hohlraum_single_node_gpu.xml . 
for s in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 66 100 133 200 500 1000 2000 4000 5000
do
  sed -i 's/<photons>.*<\/photons>/<photons>'${s}'00000\<\/photons>/g' 3D_hohlraum_single_node_gpu.xml 
  grep photons 3D_hohlraum_single_node_gpu.xml 
  #mpirun -np 1 ./BRANSON ./3D_hohlraum_single_node_gpu.xml | tee run.$s.out
  srun -n 1 ./BRANSON ./3D_hohlraum_single_node_gpu.xml | tee run.$s.out
done
