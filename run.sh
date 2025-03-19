#!/bin/bash

# srun -p PV-Short --exclusive --gres=gpu:1 main $@

srun -p PV-Short --exclusive --gres=gpu:1 \
  ./main -v -w -n 1 33554432 $@

  # nvprof main -w -n 1 33554432