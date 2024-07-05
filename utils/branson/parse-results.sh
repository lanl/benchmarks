grep FOM run.*.out  | awk -F '.' '{print  $2 " " $3  " " $4  " " $5 }' | awk '{printf "%d\t%d\t%d.%d \n", $1, $2,  $7,  $8     }' | sort -n  --key=1,1 --key=2,2
