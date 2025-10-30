#!/usr/bin/env bash

if [ ! "${_grid_wrapped_+set}" = set ]; then
  _grid_wrapped_=1
  echo "$0" "$@"
fi
