#!/bin/bash
CONFIG=$1
cd ../../tools
bash drifter.sh --exp_config "$CONFIG"
