"""Execute the normal simulator DAG in this process, without brokers."""
import json
from pathlib import Path

from dasklearn.tasks.local import execute_local


async def run_local(simulation):
    import torch
    torch.use_deterministic_algorithms(True)
    simulation.setup_directories()
    simulation.simulate(replay=simulation.settings.feedback_replay)
    if simulation.settings.final_heldout:
        if simulation.settings.evaluation_split != "validation":
            raise ValueError("final_heldout requires validation curves")
        from dasklearn.tasks.task import Task
        for peer, (round_nr, model) in sorted(simulation.latest_models.items()):
            simulation.clients[peer].create_compute_task("test", {
                "model": model, "round": round_nr, "peer": peer, "time": simulation.settings.duration,
                "evaluation_split": "test", "evaluation_examples": 0, "output_prefix": "heldout",
            }, name=Task.generate_name("final_test"))
        simulation.workflow_dag.save_to_file(str(Path(simulation.data_dir) / "workflow_dag"))
    simulation.workflow_dag.build_task_indices()
    simulation.workflow_dag.check_validity()
    def progress(status):
        print(json.dumps({"execution": status}), flush=True)
        (Path(simulation.data_dir) / "execution-progress.json").write_text(json.dumps(status, indent=2) + "\n")
    execute_local(simulation, retain_results=False, progress=progress)
    simulation.merge_accuracies_files()
    stats = {"model_payload_bytes": simulation.model_size,
             "training_tasks": sum(task.func in {"train", "qgm_train", "demo_prepare"}
                                   for task in simulation.workflow_dag.tasks.values()),
             "bytes_sent": sum(client.bw_scheduler.total_bytes_sent for client in simulation.clients),
             "simulated_compute_seconds": sum(client.compute_time for client in simulation.clients) / 1000000,
             "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    (Path(simulation.data_dir) / "execution-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (Path(simulation.data_dir) / "training-executor.json").write_text(
        json.dumps(simulation.execution_training_stats, indent=2) + "\n")
    summary = {"output": simulation.data_dir, "model_payload_bytes": simulation.model_size}
    if simulation.settings.lora:
        summary["adapter_payload_bytes"] = simulation.model_size
    print(json.dumps(summary))
