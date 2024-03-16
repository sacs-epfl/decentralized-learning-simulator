import json
from typing import Dict, List


class Task:

    def __init__(self, name: str, func: str, data: Dict):
        self.name = name
        self.func = func
        self.data = data

        self.inputs: List[Task] = []
        self.outputs: List[Task] = []

        self.inputs_resolve: int = 0
        self.done: bool = False

    @staticmethod
    def replace_values_recursively(d, string_to_replace, new_value, do_replace: bool = True) -> int:
        values_replaced: int = 0
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, (dict, list)):
                    values_replaced += Task.replace_values_recursively(value, string_to_replace, new_value, do_replace)
                elif value == string_to_replace:
                    if do_replace:
                        d[key] = new_value
                    values_replaced += 1
        elif isinstance(d, list):
            for i, item in enumerate(d):
                if isinstance(item, (dict, list)):
                    Task.replace_values_recursively(item, string_to_replace, new_value, do_replace)
                elif item == string_to_replace:
                    if do_replace:
                        d[i] = new_value
                    values_replaced += 1

        return values_replaced

    def set_data(self, input_task_name: str, data, do_replace: bool = True) -> int:
        # Iteratively go through the dictionary and sub-dictionaries and replace instances
        values_replaced: int = Task.replace_values_recursively(self.data, input_task_name, data, do_replace)
        self.inputs_resolve += 1
        return values_replaced

    def has_all_inputs(self) -> bool:
        return self.inputs_resolve >= len(self.inputs)

    def clear_data(self):
        self.data = None

    def to_json_dict(self) -> Dict:
        return {
            "name": self.name,
            "func": self.func,
            "data": self.data,
            "inputs": [task.name for task in self.inputs],
            "outputs": [task.name for task in self.outputs]
        }

    @staticmethod
    def from_json_dict(json_dict) -> "Task":
        # We don't set the inputs/outputs here - this should be done later.
        return Task(json_dict["name"], json_dict["func"], json_dict["data"])

    def __str__(self):
        return "<%s, %s>" % (self.name, self.func)

    def __repr__(self):
        return str(self)
