#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No tool name provided"
    exit 1
fi

# Calls the python script with parameters defined in the bash file passed as a parameter.
TOOL_FILE="$1.sh"
TOOL_SCRIPT_RAW=$(cat $TOOL_FILE)
TOOL_SCRIPT=$(tr '\n' ' ' <<< "$TOOL_SCRIPT_RAW")
TOOL_PATH="."$(cut -d '.' -f2- <<< "$TOOL_SCRIPT")
echo "Executing Python script: $TOOL_PATH"
python $TOOL_PATH
