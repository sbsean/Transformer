# main.py ( WMT14 Translation )
import os
import sys
import json
import torch
import torch.nn as nn
from typing import Type, TypeVar, Optional
from dataclasses import dataclass, field, fields

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import Transformer
from utils.data_preprocess import load_wmt14_data, save_vocab, load_vocab
from scripts.train import Trainer

T = TypeVar("T")

def parse_config(config_path: str, cls: Type[T]) -> T:
    """Parses a JSON configuration file and creates a dataclass object."""
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)

@dataclass
class Args:
    # --- Model Related Arguments ---
    d_model: int = field(
        default=512,
        metadata={"help": "Model dimension"}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads"}
    )
    d_ff: int = field(
        default=2048,
        metadata={"help": "Feed-forward dimension"}
    )
    num_layers: int = field(
        default=6,
        metadata={"help": "Number of encoder/decoder layers"}
    )
    max_len: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )

    # --- Training Related Arguments ---
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training and validation."}
    )
    num_epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the optimizer."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for the optimizer."}
    )
    optimizer: str = field(
        default="Adam",
        metadata={"help": "Optimizer to use (e.g., 'Adam', 'SGD')."}
    )
    scheduler: str = field(
        default="NoamLR",
        metadata={"help": "Learning rate scheduler to use."}
    )
    warmup_steps: int = field(
        default=4000,
        metadata={"help": "Number of warmup steps for NoamLR scheduler."}
    )
    step_size: int = field(
        default=5,
        metadata={"help": "Step size for the StepLR scheduler."}
    )
    gamma: float = field(
        default=0.5,
        metadata={"help": "Gamma value for the StepLR scheduler."}
    )

    # --- Data Related Arguments ---
    min_freq: int = field(
        default=2,
        metadata={"help": "Minimum frequency for vocabulary building"}
    )
    max_samples: Optional[int] = field(
        default=50000,
        metadata={"help": "Maximum number of samples to use (for memory management)"}
    )
    vocab_dir: str = field(
        default="./vocab",
        metadata={"help": "Directory to save vocabulary files"}
    )

   
    checkpoint_dir: str = field(
        default="./checkpoints",
        metadata={"help": "Directory to save model checkpoints."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for training"}
    )

def main():
    """Main execution function."""
    default_config_path = "config/wmt14_config.json"
    
    # 1. Load configuration
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config_path = sys.argv[1]
        args = parse_config(config_path, Args)
        print(f"Loaded configuration from: {config_path}")
    elif os.path.exists(default_config_path):
        args = parse_config(default_config_path, Args)
        print(f"Loaded default configuration from: {default_config_path}")
    else:
        args = Args()
        print("No configuration file provided. Running with default arguments.")

   
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vocab_dir, exist_ok=True)

    
    print("Loading WMT14 dataset...")
    train_loader, val_loader, src_vocab, tgt_vocab = load_wmt14_data(
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        max_samples=args.max_samples
    )

    
    save_vocab(src_vocab, os.path.join(args.vocab_dir, "src_vocab.pkl"))
    save_vocab(tgt_vocab, os.path.join(args.vocab_dir, "tgt_vocab.pkl"))

   
    print("Initializing Transformer model...")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        max_len=args.max_len,
        dropout=args.dropout
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

   
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_data=train_loader,
        val_data=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        # Pass all arguments to the trainer
        **vars(args)
    )

    
    print("\n=== Starting Training ===")
    print(f"Configuration:")
    print(f"  - Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Device: {args.device}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Scheduler: {args.scheduler}")
    print(f"  - Max samples: {args.max_samples}")
    
    trainer.train()
    print("=== Training Finished ===")

if __name__ == "__main__":
    main()