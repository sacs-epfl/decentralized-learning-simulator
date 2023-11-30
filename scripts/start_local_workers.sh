#!/bin/bash

# Check if sufficient arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <coordinator> <number_of_workers>"
    exit 1
fi

coordinator=$1
num_workers=$2
base_port=6000
pids=()

# Function to kill all spawned processes
cleanup() {
    echo "Killing all spawned processes..."
    for pid in "${pids[@]}"; do
        kill "$pid"
    done
}

# Trap CTRL+C (SIGINT) to run the cleanup function
trap cleanup SIGINT

# Spawn workers
for (( i=0; i<num_workers; i++ )); do
    port=$(($base_port + $i))
    python3 worker.py --coordinator "$coordinator" --port "$port" > /dev/null 2>&1 &
    pids+=($!)
    echo "Spawned worker at port $port"
done

# Wait for all processes
echo "Waiting for all workers to complete..."
wait

echo "All workers completed."