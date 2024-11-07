import argparse

import random
import numpy as np
import torch
import wandb

from model import CustomCLIP
from dataset import MSCOCODataLoader
from trainer import Trainer

def set_seed(seed):
    """
    Set the seed for reproducibility
    
    Parameters:
    - seed: The seed value to set.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_vision_model_name(sz):

    assert sz in ["small", "medium", "large"]
    
    sz2modelname = {
        "small": "google/vit-base-patch16-224-in21k",
        "medium": "google/vit-large-patch16-224-in21k",
        "large": "google/vit-huge-patch14-224-in21k"
    }

    return sz2modelname[sz]

def get_text_model_name(sz):

    assert sz in ["small", "medium", "large"]
    
    sz2modelname = {
        "small": "openai-community/gpt2",
        "medium": "openai-community/gpt2-medium",
        "large": "openai-community/gpt2-large"
    }

    return sz2modelname[sz]

def get_args():
    parser = argparse.ArgumentParser(description="Custom CLIP Model Training Script")

    # Model configuration
    parser.add_argument("--text_model_size", type=str, default="medium", 
                        help="Pretrained text model size")
    parser.add_argument("--vision_model_size", type=str, default="medium", 
                        help="Pretrained vision model size")
    parser.add_argument("--embedding_dim", type=int, default=1024, 
                        help="Dimension of the joint embedding space")
    parser.add_argument("--use_peft", action="store_true", 
                        help="Whether to use Parameter-Efficient Fine-Tuning (PEFT)")

    # Training configuration
    parser.add_argument("--seed", type=int, default=44,
                        help="Set the seed for the training")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="How many batches to wait before logging training status")
    parser.add_argument("--eval_interval", type=int, default=5000, 
                        help="How many batches to wait before evaluating and checkpointing")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to log metrics to Weights and Biases")
    parser.add_argument("--checkpoint_dir", type=str, default="/users/thua5/ssl_proj/checkpoints", 
                        help="Directory where model checkpoints will be saved")

    # Paths to MSCOCO Data
    parser.add_argument("--train_captions_json_path", type=str, default="/gpfs/data/superlab/datasets/coco/annotations/captions_train2017.json")
    parser.add_argument("--train_images_folder_path", type=str, default="/gpfs/data/superlab/datasets/coco/train2017")
    parser.add_argument("--val_captions_json_path", type=str, default="/gpfs/data/superlab/datasets/coco/annotations/captions_val2017.json")
    parser.add_argument("--val_images_folder_path", type=str, default="/gpfs/data/superlab/datasets/coco/val2017")

    return parser.parse_args()



if __name__ == "__main__":

    args = get_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_model_name   = get_text_model_name(args.text_model_size)
    vision_model_name = get_vision_model_name(args.vision_model_size)

    # Define model and data loader
    clip_model = CustomCLIP(
        text_model_name, 
        vision_model_name, 
        embedding_dim=args.embedding_dim,
        use_peft=args.use_peft
    ).to(device)

    # Setup optimizer
    

    if args.use_peft:
        lora_params = [p for n, p in clip_model.named_parameters() if 'lora' in n]
        other_params = list(clip_model.text_projection.parameters()) + list(clip_model.vision_projection.parameters()) + [clip_model.logit_scale]
        optimizer = torch.optim.Adam(lora_params + other_params, lr=args.learning_rate)
    else:
        other_params = list(clip_model.text_projection.parameters()) + list(clip_model.vision_projection.parameters()) + [clip_model.logit_scale]
        optimizer = torch.optim.Adam(other_params, lr=args.learning_rate)
    
    # Initialize the trainer
    trainer = Trainer(
        clip_model, 
        optimizer, 
        args,
        batch_size=args.batch_size, 
        log_interval=args.log_interval, 
        eval_interval=args.eval_interval,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )

    train_dataloader = MSCOCODataLoader(
        args.train_captions_json_path, 
        args.train_images_folder_path, 
        text_model_name, 
        vision_model_name,
        batch_size = args.batch_size
    ).load_datasets()

    val_dataloader = MSCOCODataLoader(
        args.val_captions_json_path,
        args.val_images_folder_path,
        text_model_name, 
        vision_model_name,
        batch_size = args.batch_size
    ).load_datasets()

    # Edit before individual runs
    if args.use_wandb:
        wandb.login(key="")
        wandb.init(
            project="CSCI2950X_ImprovingCLIP", 
            name=f"clip_{args.text_model_size}-text_{args.vision_model_size}-vision_{'peft' if args.use_peft else 'projection_only'}_seed{args.seed}",
            config=vars(args),
            entity="ethathua"
        )

    # Train the model
    trainer.train(
        train_dataloader, 
        val_dataloader, 
        epochs=args.epochs
    )