from typing import Dict, List, Tuple
from collections import Counter
from dasklearn.tasks.references import TaskRef, is_reference, as_ref


class Task:
    COUNTER = Counter()

    def __init__(self, name: str, func: str, data: Dict, *, operation=None, available_at=None):
        self.name = name
        self.func = func
        self.data = data
        self.operation = operation
        self.available_at = available_at
        self.resolved_inputs = set()

        self.inputs: List[Tuple[Task, int]] = []
        self.outputs: List[Tuple[Task, int]] = []

        self.inputs_resolve: int = 0
        self.done: bool = False
        self.scheduled: bool = False
        self.index_map: Dict[TaskRef, List[Tuple]] = {}

    @staticmethod
    def generate_name(base: str) -> str:
        Task.COUNTER[base] += 1
        name = base + "_" + str(Task.COUNTER[base])
        return name

    def set_data(self, input_ref: TaskRef, new_data) -> int:
        pointers = self.index_map.get(input_ref, [])
        values_replaced = 0
        def replace_at(value, path):
            if not path:
                return new_data
            key, *rest = path
            if isinstance(value, tuple):
                items = list(value)
                items[key] = replace_at(items[key], rest)
                return tuple(items)
            value[key] = replace_at(value[key], rest)
            return value

        for path in pointers:
            self.data = replace_at(self.data, path)
            values_replaced += 1
        if pointers and input_ref not in self.resolved_inputs:
            self.resolved_inputs.add(input_ref)
            self.inputs_resolve += sum(TaskRef(task.name, idx) == input_ref for task, idx in self.inputs)
        return values_replaced

    def has_all_inputs(self) -> bool:
        return self.inputs_resolve >= len(self.inputs)

    def clear_data(self):
        self.index_map = None
        self.data = None

    def to_json_dict(self) -> Dict:
        return {
            "name": self.name,
            "func": self.func,
            "data": self.data,
            "inputs": [(task.name, idx) for task, idx in self.inputs],
            "outputs": [(task.name, idx) for task, idx in self.outputs],
            "format_version": 2,
            "operation": self.operation,
            "available_at": self.available_at,
        }

    @staticmethod
    def from_json_dict(json_dict) -> "Task":
        if json_dict.get("format_version") != 2:
            raise ValueError("Unsupported task format; regenerate the workflow DAG with the current simulator")
        # We don't set the inputs/outputs here - this should be done later.
        return Task(json_dict["name"], json_dict["func"], json_dict["data"],
                    operation=json_dict.get("operation"), available_at=json_dict.get("available_at"))

    def __str__(self):
        return "<%s, %s>" % (self.name, self.func)

    def __repr__(self):
        return str(self)

    def build_index(self, container):
        self.index_map = {}

        def visit(value, path):
            if is_reference(value):
                self.index_map.setdefault(as_ref(value), []).append(path)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path + (key,))
            elif isinstance(value, (tuple, list)):
                for key, item in enumerate(value):
                    visit(item, path + (key,))

        visit(container, ())
