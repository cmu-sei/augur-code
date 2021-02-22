#!/usr/bin/env bash
bash stop_compose.sh
export CMD_PARAMS="$@"
docker-compose up -d
bash logs_compose.sh
