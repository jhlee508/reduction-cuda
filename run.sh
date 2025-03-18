#!/bin/bash

# srun -p PV-Short --exclusive --gres=gpu:1 main $@

srun -p PV-Short --exclusive --gres=gpu:1 \
  main -v -w -n 4 33554432 $@

  # nvprof main -v -w -n 5 33554432