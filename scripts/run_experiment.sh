#!/bin/bash

cleanup() {
    echo "Killing all spawned processes..."
    PGID=$(ps -o pgid= $$ | grep -o '[0-9]*')
    kill -TERM -"$PGID"
}

trap cleanup SIGINT

python3 main.py --rounds 500 --peers 100 --test-interval 10 --model resnet18 > /dev/null 2>&1 &
python3 broker.py --coordinator tcp://localhost:5555 --workers 5 > /dev/null 2>&1 &

echo "Waiting for experiment to complete..."
wait

echo "Experiment done"