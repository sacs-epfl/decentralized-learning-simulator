# Datasets

Hugging Face `datasets` loads data, and Flower `flwr-datasets` assigns examples
to clients. New sources do not need a Python loader or a dataset registry entry.
`--dataset` selects an existing model/preprocessing preset; `--dataset-name`
selects the source. Images must match the preset's input dimensions and class
count. Text supports different class counts through `--num-labels`.

For example, train LoRA on AG News with four labels and non-IID clients:

```sh
python scripts/run_lora.py --dataset text --dataset-name ag_news \
  --num-labels 4 --lora --base-model google/bert_uncased_L-2_H-128_A-2 \
  --partitioner dirichlet --alpha .5 --peers 8 --rounds 5
```

Existing presets retain default sources: `cifar10` uses `uoft-cs/cifar10`,
`femnist` uses `coscotuff/femnist`, `tiny_imagenet` uses
`zh-plus/tiny-imagenet`, and `text` uses the bundled `toy-text` smoke workload.
`synthetic` and `digits` remain offline validation workloads; they do not use
`--dataset-name`. The synthetic generator retains its fixed smoke-test sampling;
all real datasets, including digits and tokenized text, use Flower partitioners.

## Sources and columns

- `--dataset-name`: any Hub ID, a local data directory, or a Hugging Face builder
  such as `json`, `csv`, `parquet`, or `imagefolder`.
- `--dataset-config`, `--dataset-revision`, `--dataset-cache-dir`: optional
  configuration/subset, pinned data revision, and download cache directory.
- `--dataset-data-files`: a JSON split-to-files mapping, passed to `load_dataset`.
- `--dataset-path` (also `--dataset-base-path`): local data directory, including
  a `DatasetDict.save_to_disk` directory. With a builder, supplies `data_dir`.
- `--feature-column`, `--label-column`: map image/label columns to the preset.
  For text, use `--text-column` and `--label-column` (defaults: `text`, `label`).

For local classification files, add these options to the LoRA command:

```sh
--dataset-name json \
--dataset-data-files '{"train":"./train.jsonl","test":"./test.jsonl"}' \
--text-column sentence --label-column category
```

Image presets expect `img`/`label` for CIFAR-10, `x`/`y` for FEMNIST, and
`image`/`label` for Tiny ImageNet. Override the column flags for a source with
different names. Labels must be integer class IDs within the model's output
range. Hub sources must contain data files supported by Hugging Face; loaders
do not execute repository Python scripts.

## Splits and partitions

`--train-split`, `--test-split`, and `--validation-split` map source split names.
Defaults are `train`, `test` (`valid` for Tiny ImageNet), and `validation` when
present. Train and final test data must exist. Without validation data, the
loader holds out 10% of training data (5% for Tiny ImageNet) before client
partitioning. Set `--validation-set-fraction` to change this. Holdout seed 1729
is fixed across training seeds; ClassLabel datasets use stratified holdouts.
Validation and test cannot use the same source split. If a dataset only offers
`train` and `validation`, use `--test-split validation`: tuning data will be
held out from training separately.

Flower supplies all four partitioners:

| `--partitioner` | Settings |
| --- | --- |
| `iid` | Seeded shuffling and even client sizes. |
| `dirichlet` | `--alpha` controls label heterogeneity; `--min-partition-size` defaults to 1. |
| `shards` | Label-sorted shards; `--shards-per-partition` defaults to approximately 2, with remainder shards distributed. |
| `natural` | One partition per unique `--partition-column` client ID; `--peers` must match the number of IDs. |

`--partition-column` can also override the label column used for Dirichlet or
shards. FEMNIST and speech default natural partitioning to `writer_id` and
`speaker_id`. `--seed` controls partition randomness. Flower reports an error
when a Dirichlet configuration cannot meet its minimum size; increase alpha,
reduce the client count, or reduce the minimum. Cached native image and digits
workloads require at least eight examples per client.

## Offline image benchmarks and migration

Use the same preparation command for any dataset:

```sh
python scripts/prepare_dataset.py uoft-cs/cifar10 --data data/cifar10
python scripts/prepare_dataset.py zh-plus/tiny-imagenet --data data/tiny-imagenet
```

Add `--revision COMMIT` to pin a Hub revision. Preparation saves standard Hugging
Face Arrow data and source metadata. Pass `--dataset-base-path data/cifar10` to
the CIFAR runner, or `--data data/tiny-imagenet` to the Tiny ImageNet runner.
These runners retain normalized tensor caches, native image sizes, and balanced
evaluation subsets. CIFAR native caching supports LeNet/GNLeNet.

The old `prepare_cifar.py`, `prepare_tiny.py`, custom NPZ/NPY archive loaders,
and hand-written real-data Dirichlet allocators have been removed. Regenerate
old prepared directories using the commands above. Historical experiment
artifacts retain their original provenance; reruns use Flower's allocation and
the shared holdout implementation, so their exact partitions and scores can
differ. `--text-dataset` and `--text-dataset-config` still work for saved commands;
the shared `--dataset-name`/`--dataset-config` options take precedence.
