import pytest

from dasklearn.tasks.task import Task


@pytest.fixture
def task():
    task = Task("dummy_task", "func", {"a": "input_1", "b": "input_2", "c": [{"d": "input_1"}, {"e": "input_2"}]})
    t1 = Task("input_1", "func", {})
    t1.outputs = [task]
    t2 = Task("input_2", "func", {})
    t2.outputs = [task]
    task.inputs = [t1, t2]
    return task


def test_task_resolve_input(task):
    assert task.inputs_resolve == 0

    task.set_data("input_1", 42)
    assert task.inputs_resolve == 1
    assert task.data["a"] == 42
    assert task.data["c"][0]["d"] == 42

    task.set_data("input_2", 43)
    assert task.inputs_resolve == 2
    assert task.data["b"] == 43
    assert task.data["c"][1]["e"] == 43
    assert task.has_all_inputs()


def test_task_to_json(task):
    task_json = task.to_json_dict()
    assert task_json
    assert task_json["name"] == task.name
    assert task_json["func"] == task.func
    assert task_json["data"] == task.data
    assert len(task_json["inputs"]) == 2
    assert len(task_json["outputs"]) == 0
