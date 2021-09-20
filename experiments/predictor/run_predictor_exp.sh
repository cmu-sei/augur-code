#!/bin/bash
CONFIG=$1
cd ../../tools
bash predictor.sh --store --exp_config "$CONFIG"
