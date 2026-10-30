# LoRA fine-tuning

Activate your simulator conda environment and install the optional dependencies:

```sh
conda activate decentralized-learning
python -m pip install -r requirements-lora.txt
```

Run a pretrained small BERT on the bundled sentiment smoke dataset, locally with
no brokers. The first run downloads the checkpoint and tokenizer; later runs
reuse the Hugging Face cache. A local checkpoint directory also works.

```sh
python scripts/run_lora.py --algorithm dpsgd --dataset text --lora \
  --base-model google/bert_uncased_L-2_H-128_A-2 --lora-target-modules query,value \
  --lora-rank 4 --lora-alpha 8 --lora-dropout 0 \
  --device cpu --peers 4 --topology ring --rounds 10 --local-steps 20 \
  --learning-rate 0.1 --momentum 0 --batch-size 16 \
  --max-sequence-length 32 --test-interval 1
```

The runner prints its output directory and adapter payload size. Normal
`settings.json`, `workflow_dag`, and `accuracies.csv` artifacts are produced.
Use `--compute-validation-loss-global-model` to also record the loss of each
client's incoming model on a separate validation split in `validation_losses.csv`.
The small toy dataset checks the pipeline; its scores are not benchmark results.

The same LoRA arguments work with `main.py` and normal broker execution. Start
with **one worker per GPU**: clients within a worker share a base, but independent
OS processes each load their own base. More workers do not share CUDA storage.

## Options and data

- `--base-model`: Hugging Face model ID or local pretrained checkpoint directory.
- `--model-revision`: pin the model/tokenizer revision, ideally a commit hash.
- `--tokenizer`: optional separate tokenizer ID/path.
- `--lora-rank` (alias `--lora-r`), `--lora-alpha`, `--lora-dropout`.
- `--lora-target-modules`: comma-separated names (BERT: `query,value`; Llama:
  `q_proj,v_proj`), `all-linear`, or omit for PEFT's architecture defaults.
- `--lora-task sequence-classification` (default), `--num-labels 2`.
- `--lora-task causal-lm`: next-token prediction using a causal checkpoint.
- `--max-sequence-length`: fixed truncation/padding length (default 128).
- `--dataset-name`: `toy-text`, a Hugging Face dataset ID, or a directory with
  `train.jsonl`, `test.jsonl`, and optionally `validation.jsonl`.
- `--dataset-config`: optional Hugging Face dataset subset/config.
- `--text-dataset` and `--text-dataset-config`: compatibility aliases for older commands.
- `--text-column`, `--label-column`: default to `text`, `label`.

For example, use `--dataset-name ag_news --num-labels 4`, or
`--dataset-name imdb`. Datasets must have train and test splits; if validation is
absent, a deterministic 10% of training data is held out (configurable via
`--validation-set-fraction`). Classification labels must be integers in
`[0, num_labels)`. Local JSONL rows look like `{"text":"a good movie","label":1}`.
For causal LM, rows need only the text column. Each row is tokenized separately;
padding positions are ignored. Sequences need at least two tokens. IID partitions
work for both tasks; Dirichlet and shard partitions require labels (or an explicit
`--partition-column` for causal LM). Flower handles IID, Dirichlet, shard and
natural client partitions. See [dataset configuration](datasets.md) for local
CSV/JSON/Parquet data, custom splits, and natural client IDs.

Example causal LM run:

```sh
python scripts/run_lora.py --algorithm fl --dataset text --lora \
  --base-model HuggingFaceTB/SmolLM2-135M --lora-task causal-lm \
  --lora-target-modules q_proj,v_proj --text-dataset ./my-text \
  --peers 2 --sample-size 2 --rounds 5 --local-steps 20 \
  --learning-rate 0.01 --momentum 0 --batch-size 2 --test-interval 1
```

Existing learning flags still control SGD; learning rates should be tuned for the
checkpoint/task. Classification accuracy is the percentage of correct examples.
Causal LM accuracy is the percentage of correct **next tokens**, excluding padding;
loss is token-weighted shifted cross entropy. Classification loss is weighted by
example count. Test and validation use the same tokenizer and masking as training.

## Memory and algorithm behavior

`LoRAModel` contains only trainable adapter tensors and, for classification, PEFT's
trainable classification head. Frozen tensors live in a lazy process-local runtime,
not in the client model object. Deep copies, DAG values, broker pickle messages,
parameter vectors, chunk reconstruction, and simulated transfer sizes therefore
exclude the frozen base. The runtime stays on the last execution device while
client adapters return to CPU. It holds one base for the current configuration;
changing configurations evicts the cached base.

Forward passes use PyTorch `functional_call` to substitute each client's adapter
parameters into the shared PEFT model. The base remains frozen and is never merged
with adapters. Mode/device changes and substitution are protected by a lock.
Workers execute tasks serially; the accelerator batching path is disabled for LoRA
to keep one base resident. Base precision follows the checkpoint loader's default;
quantization, device-map sharding, and gradient checkpointing are not exposed here.

Algorithms continue to operate on their normal parameter/state interfaces, now
covering only adapters and the optional head. This includes FedAvg, D-PSGD,
AD-PSGD, gossip, chunk-based Shatter/Conflux, Moshpit, Teleportation, CHOCO, QGM,
DSpodFL, and DeMo. Their usual topology/optimizer restrictions still apply.
Averaging LoRA A/B factors implements **parameter averaging**; it is not equal to
averaging the effective dense updates `B @ A`. Chunk algorithms can mix portions
of those factors, as they do with normal parameters. No special LoRA aggregation
rule silently replaces the chosen algorithm.

## Validation

```sh
python -m pytest tests/test_lora.py -q -s
```

These offline tests save tiny local BERT/GPT-2 checkpoints (the BERT backbone gets a short
synthetic pretraining stage and its task head is reset) and use
real Transformers/PEFT training. They check validation loss reduction, accuracy,
unchanged base weights, isolated clients, adapter-only copies/serialization,
chunk reconstruction, gradients, and execution of existing protocol DAGs. They
are pipeline tests; they do not establish quality on a pretrained benchmark.

A CPU smoke run on 2026-10-30 used Python 3.12, PyTorch 2.15.0,
Transformers 5.15.0, and PEFT 0.21.2 in a dedicated conda environment. With
`google/bert_uncased_L-2_H-128_A-2`, rank 4, alpha 8, `query,value` targets,
and SGD lr 0.1, held-out toy validation loss fell from 0.7107 to 0.00086 over
100 steps; accuracy went from 50% to 100%. A two-peer D-PSGD run (four rounds,
25 steps/round) produced a 20,534-byte adapter payload and final test loss
0.00079. These measurements validate the plumbing on an easy synthetic task.
