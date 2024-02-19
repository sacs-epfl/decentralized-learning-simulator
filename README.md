## Decentralized Learning Simulator

This repository contains the code related to a Decentralized Learning simulator.
The main idea of this simulator is that first, the timestamps all events in the system (training, model transfers, aggregation etc.) are determined using a discrete-event simulator.
Meanwhile, the simulator devises a compute graph, containing all compute tasks (train, aggregate or test).
Afterwards, the compute graph is solved, possibly using different machines and multiple workers.

There are three different process types: the coordinator, a broker and a worker.
The coordinator node runs the discrete-event simulation and then pushes the compute graph to the brokers.
A broker process starts one or more workers, and workers are responsible for carrying out model training, aggregation or testing.

A main advantage of this simulator over other ones is that it natively supports reporting total time duration of the learning process.
It also supports the integration of traces that include the training and network capacity of each node.
The trace data can be found in `data/client_device_capacity`.

Currently, only the standard [D-PSGD algorithm](https://proceedings.neurips.cc/paper/2017/file/f75526659f31040afeb61cb7133e4e6d-Paper.pdf) algorithm has been implemented.
Furthermore, the only supported datasets right now are CIFAR-10 and FEMNIST.

### Running the Simulator

Make sure you have all required dependencies first, which can be found in `requirements.txt`.
The simulator expects the CIFAR-10 dataset to be located at `~/dfl-data`.
FEMIST should be located at `~/leaf`.
You can run the coordinator node using:

```
python3 main.py
```

You can inspect the possible parameters by running `python3 main.py -h`.
The coordinator node will then run the discrete-event simulator and generate a compute graph.
By default, the coordinator expects only a single broker to be started but this number can be changed with the `--brokers` flag.
You can start a broker as follows:

```
python3 broker.py --coordinator tcp://localhost:5555
```

The coordinator will by default listen on port `5555`.
After the broker started, it will spawn a worker process, receive the compute graph from the coordinator and starts to solve the compute graph.
The resulting models (i.e., the outcome of the leaf nodes in the compute graph) will be communicated back to the coordinator.