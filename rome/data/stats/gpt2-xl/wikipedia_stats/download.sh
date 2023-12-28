#!/bin/bash

for layer in {0..47}
do
  eval wget https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/transformer.h.${layer}.mlp.c_proj_float32_mom2_100000.npz
done
