#!/usr/bin/env bash

set -uoe pipefail

if (( $# != 3 )); then
    echo "usage: $(basename "$0") <telemetry log> <database file> <table>" 1>&2
    exit 1
fi
log=$1
db=$2
table=$3

sqlite3 -batch "${db}" << EOF
DROP TABLE IF EXISTS ${table};
.mode csv
.headers on
.import ${log} ${table}
EOF
