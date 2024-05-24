#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PORT=5000

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -p|--port)
        PORT="$2"
        shift 
        shift
        ;;
        *)    
        shift 
        ;;
    esac
done


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=$PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION

streamlit run app.py --server.port $PORT
