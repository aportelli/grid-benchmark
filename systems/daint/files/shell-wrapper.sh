#!/usr/bin/env bash

export _grid_wrapped_=1
uenv image pull prgenv-gnu/25.6:v1
uenv run --view=default prgenv-gnu/25.6:v1 -- "$@"
