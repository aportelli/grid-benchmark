#!/usr/bin/env bash

set -uoe pipefail

if (( $# != 3 )); then
    echo "usage: $(basename "$0") <nvidia-smi dmon log> <database file> <table>" 1>&2
    exit 1
fi
log=$1
db=$2
table=$3

schema=$(head -n1 "${log}" | sed 's/#//g' \
         | awk '{for(i=1;i<=NF;i++){printf "%s TEXT", $i; if(i<NF) printf ", "}}')
sqlite3 -batch "${db}" << EOF
DROP TABLE IF EXISTS ${table};
CREATE TABLE ${table} (
  ${schema}
);
EOF
tmp=$(mktemp)
grep -vE '^#' "${log}" \
  | awk '{for(i=1;i<=NF;i++){printf "%s", $i; if(i<NF) printf "|"} printf "\n"}' > "${tmp}"
sqlite3 -batch "${db}" ".import ${tmp} ${table}"
