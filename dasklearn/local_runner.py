"""Execute the normal simulator DAG in this process, without brokers."""
import json

from dasklearn.tasks.local import execute_local


async def run_local(simulation):
    simulation.setup_directories()
    simulation.simulate(replay=simulation.settings.feedback_replay)
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    execute_local(simulation, retain_results=False)
    simulation.merge_accuracies_files()
    summary = {"output": simulation.data_dir, "model_payload_bytes": simulation.model_size}
    if simulation.settings.lora:
        summary["adapter_payload_bytes"] = simulation.model_size
    print(json.dumps(summary))
