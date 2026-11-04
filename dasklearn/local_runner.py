"""Execute the normal simulator DAG in this process, without brokers."""
import json
import os

from dasklearn.tasks.dag import WorkflowDAG

from dasklearn.tasks.local import execute_local


async def run_local(simulation):
    simulation.setup_directories()
    if simulation.settings.from_dir:
        simulation.workflow_dag = WorkflowDAG.load_from_file(
            os.path.join(simulation.settings.from_dir, 'workflow_dag'))
    else:
        simulation.simulate(replay=simulation.settings.feedback_replay)
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    execute_local(simulation, retain_results=False)
    simulation.merge_accuracies_files()
    summary = {"output": simulation.data_dir}
    if not simulation.settings.from_dir:
        summary["model_payload_bytes"] = simulation.model_size
    if simulation.settings.lora and not simulation.settings.from_dir:
        summary["adapter_payload_bytes"] = simulation.model_size
    print(json.dumps(summary))
