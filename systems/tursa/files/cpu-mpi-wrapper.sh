#!/usr/bin/env bash

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
numa=${lrank}
cpus="$(( lrank*64 ))-$(( (lrank+1)*64-1 ))"
places="$(( lrank*64 )):$(( (lrank+1)*64 ))"

BINDING="taskset -c ${cpus} numactl -m ${numa}"
export OMP_PLACES=${places}

echo "$(hostname) - ${lrank} binding='${BINDING}'"

${BINDING} "$@"
