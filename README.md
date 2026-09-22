# Vebra

Vebra is an experimental small language model (SLM) built from scratch with PyTorch.

The project focuses on implementing a decoder-only Transformer architecture and building the supporting tokenizer, training, checkpointing, and evaluation infrastructure required for language-model experiments.

> Vebra is currently an experimental research and development project. Large-scale training results are not claimed yet.

## Architecture

Vebra currently implements:

* Decoder-only Transformer
* Token embeddings
* RMSNorm
* Rotary Positional Embeddings (RoPE)
* Causal self-attention
* SwiGLU feed-forward networks
* Pre-norm residual Transformer blocks
* Final RMSNorm
* Tied token embedding and language-model head weights
* Next-token prediction

### Model Pipeline

```text
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
Transformer Blocks
    ├── RMSNorm
    ├── Causal Self-Attention + RoPE
    ├── Residual Connection
    ├── RMSNorm
    ├── SwiGLU
    └── Residual Connection
    │
    ▼
Final RMSNorm
    │
    ▼
LM Head
    │
    ▼
Logits
```

## Tokenizer

Vebra includes a byte-level Byte Pair Encoding (BPE) tokenizer implemented specifically for the project.

The tokenizer supports:

* Vocabulary training
* BPE merge rules
* Encoding text into token IDs
* Decoding token IDs back into text
* JSON serialization
* Vocabulary size validation

The current tokenizer is intentionally small and is primarily used for development and pipeline validation.

## Training

The training stack currently provides:

* Causal language-model loss
* AdamW optimization
* Configurable batch size and learning rate
* Persistent training data iteration
* CPU training
* CUDA device support
* BF16 autocast on supported CUDA devices
* FP16 autocast support
* Checkpoint saving and loading
* YAML-based training configuration
* PyTorch DataLoader integration
* Optional pinned-memory DataLoader support

### Training Flow

```text
Corpus
  │
  ▼
BPE Tokenizer
  │
  ▼
Tokenized Dataset
  │
  ▼
DataLoader
  │
  ▼
Vebra Transformer
  │
  ▼
Causal LM Loss
  │
  ▼
AdamW
  │
  ▼
Checkpoint
```

## Evaluation

The evaluation layer currently supports:

* Evaluation loss
* Perplexity
* Evaluation without gradient updates
* Preservation and restoration of model training/evaluation mode
* CUDA device validation

Evaluation uses the same causal language-model objective as training.

## Configuration

Model and training parameters are defined through YAML configuration files.

Example:

```yaml
model:
  vocab_size: 284
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

tokenizer:
  path: tokenizer.json
  vocab_size: 284
  min_frequency: 2

training:
  batch_size: 4
  learning_rate: 0.001
  steps: 10
  device: cpu
  precision: fp32
```

GPU experiments can use CUDA and BF16:

```yaml
training:
  device: cuda
  precision: bf16
```

## Current Experimental Model

The current Tiny configuration is intentionally small:

| Parameter               | Value |
| ----------------------- | ----: |
| Vocabulary size         |   284 |
| Hidden size             |    64 |
| Transformer layers      |     2 |
| Attention heads         |     8 |
| Intermediate size       |   256 |
| Maximum sequence length |    16 |

This configuration is a pipeline and architecture smoke-test model, not the intended final Vebra model.

### Planned Model Scales

| Model        |    Approx. Scale |
| ------------ | ---------------: |
| Vebra Tiny   |  ~10M parameters |
| Vebra Base   |  ~50M parameters |
| Vebra Medium | ~100M parameters |
| Vebra Large  | ~350M parameters |

These are development targets and do not represent completed trained models.

## Running Tests

Install the project dependencies and run:

```bash
pytest
```

The test suite covers the model, tokenizer, data pipeline, training components, checkpointing, configuration, and evaluation.

## Training

A local experimental run can be started with:

```bash
python scripts/train.py corpus.txt configs/tiny.yaml --checkpoint vebra-checkpoint.pt
```

A CUDA/BF16 experiment can use:

```bash
python scripts/train.py corpus.txt configs/h200.yaml --checkpoint vebra-checkpoint.pt
```

The GPU configuration is intended for compatible external GPU infrastructure.

Large model checkpoints and datasets are not stored in the repository.

## Project Structure

```text
Vebra/
├── configs/
│   ├── tiny.yaml
│   └── h200.yaml
├── docs/
├── scripts/
│   ├── train.py
│   └── train_tokenizer.py
├── tests/
├── vebra/
│   ├── data/
│   ├── evaluation/
│   ├── model/
│   ├── tokenizer/
│   └── training/
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Development Status

Vebra is under active development.

### Implemented

* [x] Decoder-only Transformer
* [x] RMSNorm
* [x] RoPE
* [x] Causal self-attention
* [x] SwiGLU
* [x] Transformer blocks
* [x] Tied embeddings
* [x] Byte-level BPE tokenizer
* [x] Dataset and DataLoader pipeline
* [x] Training step
* [x] Trainer
* [x] AdamW optimization
* [x] Checkpoint save/load
* [x] YAML training configuration
* [x] CUDA support
* [x] BF16 training support
* [x] Evaluation loss
* [x] Perplexity

### Next

* [ ] Text generation
* [ ] Train/validation dataset pipeline
* [ ] First real SLM training experiment
* [ ] GPU training experiments
* [ ] Training metrics and experiment logging
* [ ] Larger model configurations
* [ ] Sampling-based generation
* [ ] Model and tokenizer documentation

## Design Goals

Vebra is intended to remain:

* **Understandable** — core model components are implemented explicitly rather than hidden behind a large framework.
* **Modular** — model, tokenizer, data, training, and evaluation are separated into independent components.
* **Reproducible** — experiments are driven by version-controlled configuration files.
* **Testable** — core functionality is covered by automated tests.
* **Scalable** — the same codebase should support local CPU development and larger CUDA-based experiments.

## License

Vebra is released under the MIT License.