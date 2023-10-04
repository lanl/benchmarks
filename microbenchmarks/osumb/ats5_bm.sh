#!/bin/bash

#### RESULT CSV 
# "Test","Ranks","Message Size","Nodes","Result"
# "osu_latency","1 per node","8 B","2",
# "osu_bibw","1 per node","1 MB","2",
# "osu_mbw_mr","1 per NIC","16 KB","2",
# "osu_mbw_mr","1 per core","16 KB","2",
# "osu_get_acc_latency","1 per node","8 B","2",
# "osu_barrier","1 per physical core","N/A","full-system",
# "osu_ibarrier","1 per physical core","N/A","full-system",
# "osu_put","1 per node","8 B","2",
# "osu_get","1 per node","8 B","2",
# "osu_allreduce","1 per physical core","8B, 25 MB","full-system",
# "osu_alltoall","1 per physical core","1 MB","full-system",
########################################################################

runosu() {

}

test_name="osu_latency
osu_bibw
osu_mbw_mr
osu_mbw_mr
osu_get_acc_latency
osu_barrier
osu_ibarrier
osu_put
osu_get
osu_allreduce
osu_alltoall"