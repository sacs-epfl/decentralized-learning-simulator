"""Real tiny transformer checks: sharing, wire state, learning, and protocol DAGs."""
import copy
import pickle
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from dasklearn import functions
from dasklearn.datasets import create_dataset
from dasklearn.datasets.text import text_forward
from dasklearn.gradient_aggregation.fedavg import FedAvg
from dasklearn.lora import shared_runtime
from dasklearn.model_evaluator import ModelEvaluator
from dasklearn.model_trainer import ModelTrainer
from dasklearn.models import create_model, copy_model_state, serialize_model, unserialize_model
from dasklearn.session_settings import SessionSettings, LearningSettings
from dasklearn.simulation.conflux.chunk_manager import ChunkManager
from dasklearn.util import MICROSECONDS
from dasklearn.tasks.local import execute_local
from helpers import settings, simulate


@pytest.fixture(scope="module")
def bert_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "happy", "nice", "bad", "sad", "awful", "movie", "example"]
    (path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = transformers.BertTokenizer(vocab_file=str(path / "vocab.txt"),
                                             vocab={word: i for i, word in enumerate(vocab)})
    tokenizer.save_pretrained(path)
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = transformers.BertForSequenceClassification(transformers.BertConfig(
            vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=48, max_position_embeddings=64,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0, num_labels=2))
        # Pretrain a tiny representation on a separate toy corpus, then reset
        # its task head. A frozen random BERT pooler has almost no class signal.
        encoded = tokenizer(["good happy nice movie", "bad sad awful movie"] * 8,
                            padding=True, return_tensors="pt")
        labels = torch.tensor([1, 0] * 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
        model.train()
        for _ in range(60):
            optimizer.zero_grad()
            model(**encoded, labels=labels).loss.backward()
            optimizer.step()
        model.classifier.reset_parameters()
        model.save_pretrained(path)
    return str(path)


def configuration(path, **kwargs):
    values = dict(algorithm="test", seed=42, work_dir="", dataset="text", participants=2,
        learning=LearningSettings(.4, 0, 0, 16, 50), lora=True, base_model=path,
        lora_rank=4, lora_alpha=8, lora_target_modules="query,value", max_sequence_length=16,
        torch_threads=1)
    values.update(kwargs)
    return SessionSettings(**values)


@pytest.fixture(autouse=True)
def reset_task_globals():
    functions.dataset = functions.model_trainers = functions.evaluator = None
    yield
    functions.dataset = functions.model_trainers = functions.evaluator = None


def test_adapter_only_copies_wire_chunks_and_aggregation(bert_checkpoint):
    config = configuration(bert_checkpoint)
    a = create_model("text", settings=config)
    runtime = shared_runtime(a.spec)
    b = copy.deepcopy(a)
    assert shared_runtime(b.spec) is runtime
    assert not list(a.buffers())
    base_params = dict(runtime.model.named_parameters())
    assert any("lora_A" in name for name in a.parameter_names)
    assert any("classifier" in name for name in a.parameter_names)
    assert all(not p.requires_grad for p in base_params.values())
    assert sum(p.numel() for p in a.parameters()) < sum(p.numel() for p in base_params.values()) / 4
    before = {k: v.clone() for k, v in a.state_dict().items()}
    with torch.no_grad():
        for p in b.parameters():
            p.add_(1)
    for original, copied in zip(a.parameters(), b.parameters()):
        assert original.data_ptr() != copied.data_ptr()
    assert all(torch.equal(v, before[k]) for k, v in a.state_dict().items())
    for result in (pickle.loads(pickle.dumps(b)), copy_model_state(b, "text"),
                   unserialize_model(serialize_model(b), "text", settings=config)):
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in result.state_dict().items())
        assert shared_runtime(result.spec) is runtime
    # The full object sent by brokers must also omit frozen tensors.
    assert len(pickle.dumps(a)) < len(pickle.dumps(runtime.model.state_dict())) / 3
    mixed = FedAvg.aggregate([a, b], [.25, .75])
    assert all(torch.allclose(v, before[k] + .75) for k, v in mixed.state_dict().items())
    chunks = ChunkManager.chunk_model(b, 3)
    rebuilt = ChunkManager.reconstruct_model([[chunk] for chunk in chunks], copy.deepcopy(a))
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in rebuilt.state_dict().items())


def test_training_improves_validation_preserves_base_and_other_client(bert_checkpoint):
    config = configuration(bert_checkpoint, compute_validation_loss_global_model=True)
    data = create_dataset(config)
    model = functions.initialize(config, {})[0]
    peer = copy.deepcopy(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    runtime = shared_runtime(model.spec)
    base_before = {k: p.clone() for k, p in runtime.model.named_parameters()}
    evaluator = ModelEvaluator(data, config)
    evaluator.settings = copy.copy(config)
    evaluator.settings.evaluation_split = "validation"
    accuracy_before, loss_before = evaluator.evaluate_accuracy(model)
    info = ModelTrainer(data, config, 0).train(model, 100, seed=2)
    accuracy_after, loss_after = evaluator.evaluate_accuracy(model)
    assert info["validation_loss_global"] == pytest.approx(loss_before)
    assert loss_after < loss_before * .6
    assert accuracy_after >= 90
    assert all(torch.equal(p, base_before[k]) for k, p in runtime.model.named_parameters())
    assert all(p.grad is None for p in runtime.model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in peer.state_dict().items())
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(p, before[f"adapter.{i}"]) for i, (name, p) in
               enumerate(zip(model.parameter_names, model.adapter)) if "lora_B" in name)
    print(f"BERT validation: loss {loss_before:.4f} -> {loss_after:.4f}, accuracy {accuracy_before:.1f}% -> {accuracy_after:.1f}%")


def test_gradient_algorithms_and_validation_csv(bert_checkpoint, tmp_path):
    config = configuration(bert_checkpoint, data_dir=str(tmp_path))
    model = functions.initialize(config, {})[0]
    gradient = functions.compute_gradient(config, {"model": model, "peer": 0, "round": 1, "time": 0})[0]
    assert len(gradient.gradient) == len(list(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient.gradient)
    updated = functions.gradient_update(config, {"model": model, "gradient_model": gradient, "peer": 0, "round": 1, "time": 0})[0]
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), updated.parameters()))
    functions.test(config, {"model": updated, "peer": 0, "round": 1, "time": 0})
    assert "accuracy,loss" in (tmp_path / "accuracies_0.csv").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["fl", "dpsgd", "epidemic", "gossip", "super-gossip", "lubor", "adpsgd", "shatter", "conflux", "moshpit", "teleportation", "choco", "qgm", "demo", "dspodfl"])
async def test_existing_algorithm_dag_with_lora(bert_checkpoint, tmp_path, algorithm):
    from dasklearn.algorithms import class_paths, import_class
    settings_path, simulation_path = class_paths(algorithm)
    settings_cls = import_class(settings_path)
    specific = {"fl": {"sample_size": 2}, "dpsgd": {"topology": "ring"}, "gossip": {"gl_period": MICROSECONDS},
                "dspodfl": {"degree": 2}, "super-gossip": {"gl_period": MICROSECONDS, "k": 2},
                "lubor": {"k": 2}, "epidemic": {"topology": "ring"},
                "shatter": {"k": 2, "r": 1}, "conflux": {"sample_size": 2, "chunks_in_sample": 2},
                "moshpit": {"group_size": 2}, "teleportation": {"sample_size": 2},
                "choco": {"topology": "ring"}, "qgm": {"topology": "ring"}}.get(algorithm, {})
    config = settings(settings_cls, tmp_path, algorithm=algorithm, dataset="text", participants=4,
        learning=LearningSettings(.1, 0, 0, 8, 1), lora=True, base_model=bert_checkpoint,
        lora_rank=2, max_sequence_length=16, rounds=2, test_interval=1, **specific)
    sim = simulate(import_class(simulation_path), config)
    results = execute_local(sim)
    assert results
    assert any(task.func == "test" for task in sim.workflow_dag.tasks.values())
    sim.merge_accuracies_files()
    assert (Path(sim.data_dir) / "accuracies.csv").exists()


def test_causal_lm_shift_mask_and_learning(bert_checkpoint, tmp_path):
    checkpoint = tmp_path / "gpt2"
    tok = transformers.AutoTokenizer.from_pretrained(bert_checkpoint)
    tok.save_pretrained(checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=len(tok), n_embd=32,
            n_layer=1, n_head=2, n_positions=32, resid_pdrop=0, attn_pdrop=0, embd_pdrop=0,
            pad_token_id=tok.pad_token_id, bos_token_id=tok.cls_token_id,
            eos_token_id=tok.sep_token_id)).save_pretrained(checkpoint)
    config = configuration(str(checkpoint), lora_task="causal-lm", lora_target_modules="c_attn",
                           learning=LearningSettings(.2, 0, 0, 16, 60))
    model = create_model("text", settings=config)
    data = create_dataset(config)
    evaluator = ModelEvaluator(data, config)
    before = evaluator.evaluate_accuracy(model)
    ModelTrainer(data, config, 0).train(model, 60, seed=3)
    after = evaluator.evaluate_accuracy(model)
    assert after[1] < before[1] * .98
    batch = next(iter(torch.utils.data.DataLoader(data.test, batch_size=4)))
    loss, logits, labels = text_forward(model.eval(), batch, "cpu")
    assert logits.shape[1] == batch["input_ids"].shape[1] - 1
    assert torch.equal(labels, batch["labels"][:, 1:])
    assert (labels == -100).any()
    output = model(**{k: v for k, v in batch.items() if k != "token_type_ids"})
    # GPT-2's own shifted CE agrees with our token-normalized loss.
    loss_without_types = text_forward(model, {k:v for k,v in batch.items() if k != "token_type_ids"}, "cpu")[0]
    torch.testing.assert_close(loss_without_types, output.loss)


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet"])
def test_local_text_data_holds_out_validation_and_partitions_once(bert_checkpoint, tmp_path, partitioner):
    import json
    from dasklearn.datasets.text import toy_records
    for split, count, offset in (("train", 40, 0), ("test", 7, 1000)):
        (tmp_path / (split + ".jsonl")).write_text("\n".join(
            json.dumps(row) for row in toy_records(count, offset)))
    config = configuration(bert_checkpoint, text_dataset=str(tmp_path), partitioner=partitioner,
                           validation_set_fraction=.2, participants=4, alpha=.1)
    data = create_dataset(config)
    assert len(data.validation) == 8
    assert len(data.test) == 7
    indices = [int(i) for part in data.partitions for i in part.indices]
    assert sorted(indices) == list(range(32))
    assert all(len(part) > 0 for part in data.partitions)
    assert data.load_split("validation") is not data.load_split("test")
    assert len(data.test.evaluation_subset(3)) == 3
    restored = SessionSettings.from_json(config.to_json())
    assert restored.lora and restored.lora_rank == config.lora_rank
    assert restored.base_model == bert_checkpoint


def test_bad_lora_options_fail_before_loading_weights():
    from dasklearn.lora import LoRASpec
    for option in ({"base_model": None}, {"lora_rank": 0}, {"lora_dropout": 1},
                   {"lora_alpha": float("nan")}, {"max_sequence_length": 1}, {"dataset": "digits"}):
        with pytest.raises(ValueError):
            config = configuration("not-a-real-model")
            for name, value in option.items():
                setattr(config, name, value)
            LoRASpec.from_settings(config)
