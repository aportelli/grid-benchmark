#!/usr/bin/env bash

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
numa=${lrank}
cpus="$(( lrank*16 ))-$(( (lrank+1)*16-1 ))"
places="$(( lrank*16 )):$(( (lrank+1)*16 ))"

BINDING="taskset -c ${cpus} numactl -m ${numa}"
export OMP_PLACES=${places}

echo "$(hostname) - ${lrank} binding='${BINDING}'"

${BINDING} "$@"
