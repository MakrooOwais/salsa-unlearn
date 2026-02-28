# SALSA Unlearning Library (`salsa-unlearn`)

**SALSA Unlearning Library** is a cohesive, comprehensive PyTorch framework for Machine Unlearning in classification. It serves as the official accompanying codebase for the paper *SALSA: A Secure, Adaptive and Label-Agnostic Scalable Algorithm for Machine Unlearning*, and provides a unified interface for evaluating multiple state-of-the-art unlearning algorithms, including SFR-on, SCRUB, SalUn, and more.

With `salsa-unlearn`, researchers can seamlessly pretrain models, execute various unlearning strategies, evaluate the retained performance (accuracy), and assess the effectiveness of the unlearning via standard Membership Inference Attacks (MIA).

## What is Machine Unlearning?

Machine unlearning selectively removes the influence of certain training samples or classes from a trained model without retraining it from scratch. This is vital for maintaining data privacy, regulatory compliance (e.g., the "Right to be Forgotten" under GDPR), and mitigating the cost of model retraining.

## Installation

### From Source (Recommended for Development)

```bash
git clone https://github.com/your-username/salsa-unlearn.git
cd salsa-unlearn
pip install -e .
```

### Manual Requirements Installation

```bash
pip install -r requirements.txt
```

## Available Unlearning Methods

The library currently supports:
- `SALSA`: A Secure, Adaptive and Label-Agnostic Scalable Algorithm for Machine Unlearning (Ours)
- `SFRon`
- `SCRUB`
- `SalUn`
- `BadTeacher`
- `GradAscent`
- `RandomLabel`
- `Finetune`
- `Retrain`
- `Baseline`

## Usage Examples

The library comes with driver scripts (`main.py`, `main_pretrain.py`, `main_random.py`) that demonstrate how to use the framework.

### 1. Pretrain the model
To pretrain a model (e.g., ResNet-18 on CIFAR-10) before executing unlearning:
```bash
python main_pretrain.py --dataset CIFAR10 --model resnet18
```

### 2. Unlearn
You can use `main.py` which demonstrates how to instantiate the unlearning methods, run them, and evaluate them automatically using the built-in MIA attacker.

```bash
python main.py
```

### 3. Programmatic Usage

You can also easily use `salsa-unlearn` inside your own Python projects. Everything is neatly packaged under `unlearn_lib`.

```python
import torch
import torch.nn as nn
from unlearn_lib.models import create_model
from unlearn_lib import create_unlearn_method

# Load your base model
model = create_model("ResNet18", num_classes=10)
model.load_state_dict(torch.load("pretrained.pth")["state_dict"])
model.cuda()

# Define the unlearning method
method_name = "SALSA"  # Or "SFRon", "SCRUB", etc.
loss_fn = nn.CrossEntropyLoss()

# args can be a simple dataclass holding hyperparameters (batch_size, num_classes, etc.)
unlearn_method = create_unlearn_method(method_name)(model, loss_fn, "./results", args)

# Prepare dataloaders
# Requires a dictionary containing: 'forget_train', 'retain_train', 'forget_valid', 'retain_valid', 'train'
unlearn_method.prepare_unlearn(unlearn_dataloaders)

# Execute unlearning and get the modified model
unlearn_model = unlearn_method.get_unlearned_model()
```

## Repository Structure

```bash
├── src/unlearn_lib/        # Core unlearning library
│   ├── dataset/            # Dataset loading and splitting logic
│   ├── evaluation/         # Evaluation metrics (JS Divergence, MIA)
│   ├── models/             # Supported network architectures (ResNet, Swin, ViT)
│   ├── trainer/            # Base training and validation loops
│   ├── unlearn/            # Implementations of unlearning algorithms (SALSA, SFRon, etc.)
│   ├── attack.py           # Attacker models for evaluating forgetting (MIA)
│   └── utils.py            # Helpful utilities
├── scripts/                # Shell scripts for standard experiments
├── main.py                 # Example script to unlearn and evaluate
├── main_pretrain.py        # Example script to pretrain models
└── setup.py                # Python package setup configuration
```

## Citation

If you use this library or the SALSA algorithm in your work, please cite:
```bibtex
@inproceedings{makroosalsa,
  title={SALSA: A Secure, Adaptive and Label-Agnostic Scalable Algorithm for Machine Unlearning},
  author={Makroo, Owais and Hassan, Atif and Khare, Swanand},
  booktitle={The 41st Conference on Uncertainty in Artificial Intelligence}
}
```
