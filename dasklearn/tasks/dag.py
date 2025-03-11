from collections import Counter
from typing import Dict, List, Set

from torch import nn
from networkx import DiGraph

from dasklearn.tasks.task import Task


class WorkflowDAG:

    def __init__(self):
        self.tasks: Dict[str, Task] = {}  # Workflow DAG

    def get_source_tasks(self) -> List[Task]:
        """
        Get the tasks without any input.
        """
        return [task for task in self.tasks.values() if not task.inputs]

    def get_sink_tasks(self) -> List[Task]:
        """
        Get the tasks without any output.
        """
        return [task for task in self.tasks.values() if not task.outputs]

    def serialize(self) -> List[Dict]:
        """
        Serialize the DAG.
        """
        return [task.to_json_dict() for task in self.tasks.values()]
    
    def build_task_indices(self):
        for task in self.tasks.values():
            task.build_index(task.data)

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "w") as dag_file:
            dag_file.write("from,to\n")
            for task_name, task in self.tasks.items():
                for output_task, _ in task.outputs:
                    dag_file.write("%s,%s\n" % (task_name, output_task.name))

    @classmethod
    def unserialize(cls, serialized_tasks):
        dag = WorkflowDAG()
        for serialized_task in serialized_tasks:
            task: Task = Task.from_json_dict(serialized_task)
            dag.tasks[task.name] = task

        # Now that we created all Task objects, fix the inputs and outputs
        for serialized_task in serialized_tasks:
            task = dag.tasks[serialized_task["name"]]
            for input_task_name, input_task_idx in serialized_task["inputs"]:
                task.inputs.append((dag.tasks[input_task_name], input_task_idx))
            for output_task_name, output_task_idx in serialized_task["outputs"]:
                task.outputs.append((dag.tasks[output_task_name], output_task_idx))

        return dag

    def print_tasks(self):
        for task in self.tasks.values():
            print("Task %s, in: %s, out: %s" % (task, task.inputs, task.outputs))

    @staticmethod
    def count_models(d, model_hashes: Set[int]):
        if isinstance(d, nn.Module):
            model_hashes.add(id(d))
        elif isinstance(d, dict):
            for item in d.values():
                WorkflowDAG.count_models(item, model_hashes)
        elif isinstance(d, list):
            for item in d:
                WorkflowDAG.count_models(item, model_hashes)

        return model_hashes

    def get_num_models(self) -> int:
        """
        Compute recursively the number of models in all tasks.
        """
        model_hashes = set()
        WorkflowDAG.count_models([d.data for d in self.tasks.values() if d.data is not None], model_hashes)
        return len(model_hashes)

    def check_validity(self):
        # Check 1 - make sure that the data dependencies are sane, e.g., the data in each task should actually be
        # dependent on its previous (input) tasks.
        for task_name, task in self.tasks.items():
            for output_task, output_index in task.outputs:
                replaced: int = output_task.set_data((task.name, output_index), "dummy", do_replace=False)
                if replaced == 0:
                    raise RuntimeError("Data of output task %s does not contain dependency on task %s!" %
                                       (output_task.name, task_name))

    def to_nx(self, max_size: int) -> tuple[DiGraph, dict[str, tuple[float, float]], list[str], dict[str, str]]:
        """
        Converts DAG to a networkx directed graph and computes the positions of nodes in a plot.
        """
        graph = DiGraph()
        position = {}
        colors = []
        color_key = {
            "train": "red",
            "compute_gradient": "pink",
            "gradient_update": "orange",
            "aggregate": "blue",
            "test": "green",
            "chunk": "purple",
            "reconstruct_from_chunks": "brown"
        }
        x_coordinate = {}
        x_last = Counter()
        # y coordinate is the peer ID

        task_types = set()
        for task in self.tasks.values():
            # Stop when we reached maximum size
            if max_size <= 0:
                break
            max_size -= 1
            # Add initial nodes to the graph
            if len(task.inputs) == 0:
                x_coordinate[task.name] = 0
                graph.add_node(task.name)
            else:
                # Position the node after its inputs
                max_input_pos: int = max(map(lambda x: x_coordinate[x[0].name], task.inputs))
                x_coordinate[task.name] = max(max_input_pos, x_last[task.data["peer"]]) + 1
                x_last[task.data["peer"]] = x_coordinate[task.name]
                # Add edges to the task's inputs
                for inp, _ in task.inputs:
                    graph.add_edge(task.name, inp.name)
            # Shift test tasks for better visibility
            if task.func == "test":
                position[task.name] = x_coordinate[task.name], task.data["peer"] + 0.5
            else:
                position[task.name] = x_coordinate[task.name], task.data["peer"]
            # Color according to the key
            colors.append(color_key[task.func])
            task_types.add(task.func)

        # Remove unused tasks from the legend
        unused_keys = set(color_key.keys()) - task_types
        for key in unused_keys:
            del color_key[key]

        return graph, position, colors, color_key
