"""One CLI-to-settings conversion shared by all maintained entry points."""
from dasklearn.session_settings import LearningSettings
from dasklearn.util import MICROSECONDS


def session_options(args, device):
    algorithm = args.algorithm
    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_steps=args.local_steps,
    )

    settings_dict = {
        "synchronous": args.synchronous,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "dataset": args.dataset,
        "dataset_base_path": args.dataset_base_path,
        "validation_set_fraction": args.validation_set_fraction,
        "compute_validation_loss_global_model": args.compute_validation_loss_global_model,
        "torch_device_name": device,
        "work_dir": args.work_dir,
        "learning": learning_settings,
        "participants": args.peers,
        "partitioner": args.partitioner,
        "alpha": args.alpha,
        "model": args.model,
        "test_interval": args.test_interval,
        "brokers": args.brokers,
        "traces": args.traces,
        "churn": args.churn,
        "min_bandwidth": args.min_bandwidth,
        "rounds": args.rounds,
        "port": args.port,
        "log_level": args.log_level,
        "torch_threads": args.torch_threads,
        "dry_run": args.dry_run,
        "duration": args.duration * MICROSECONDS,
        "test_period": args.test_period * MICROSECONDS,
        "test_method": args.test_method,
        "evaluation_split": args.evaluation_split,
        "evaluation_examples": args.evaluation_examples,
        "model_size_bytes": args.model_size_bytes,
        "release_cuda_cache": args.release_cuda_cache,
        "compute_graph_plot_size": args.compute_graph_plot_size,
        "stop": args.stop,
        "wait": args.wait,
        "stragglers_ratio": args.stragglers_ratio,
        "stragglers_proportion": args.stragglers_proportion,
        "profile": args.profile,
        "log_bandwidth_utilization": args.log_bandwidth_utilization,
        "dag_checkpoint_interval": args.dag_checkpoint_interval,
        "feedback_execution": args.feedback_execution,
        "feedback_replay": args.feedback_replay,
        "control_message_bytes": args.control_message_bytes,
        "network_latency_ms": args.network_latency_ms,
        "gpu_residency": args.gpu_residency,
        "training_batching": args.training_batching,
        "training_batch_size": args.training_batch_size,
        "gpu_state_cache_mb": args.gpu_state_cache_mb,
    }
    from dasklearn.lora import lora_options
    from dasklearn.datasets.huggingface import dataset_options
    settings_dict.update(lora_options(args))
    settings_dict.update(dataset_options(args))
    # Add algorithm-specific parameters
    from dasklearn.simulation.dspodfl.settings import dspodfl_options
    from dasklearn.simulation.demo.settings import demo_options
    from dasklearn.simulation.choco.settings import choco_options
    from dasklearn.simulation.qgm.settings import qgm_options
    algorithm_params = {
        "qgm": qgm_options(args),
        "choco": choco_options(args),
        "demo": demo_options(args),
        "dspodfl": dspodfl_options(args),
        "fl": {"sample_size": args.sample_size, "aggregation_weighting": args.fl_weighting},
        "dpsgd": {"el": args.el, "topology": args.topology, "k": args.k},
        "epidemic": {"el": args.el, "topology": args.topology, "k": args.k},
        "gossip": {"gl_period": args.gl_period * MICROSECONDS, "agg": args.agg},
        "super-gossip": {"gl_period": args.gl_period * MICROSECONDS, "agg": args.agg, "k": args.k},
        "adpsgd": {"agg": args.agg},
        "lubor": {"k": args.k, "no_weights": args.no_weights},
        "conflux": {"sample_size": args.sample_size, "chunks_in_sample": args.chunks_in_sample, "success_fraction": args.success_fraction},
        "teleportation": {"sample_size": args.sample_size, "topology": args.teleportation_topology},
        "moshpit": {"group_size": args.group_size, "averaging_steps": args.averaging_steps},
        "shatter": {"k": args.k, "r": args.r},
    }

    settings_dict.update(algorithm_params.get(algorithm, {}))
    if args.local:
        settings_dict["dry_run"] = True  # The local runner handles numerical execution.
    return settings_dict
