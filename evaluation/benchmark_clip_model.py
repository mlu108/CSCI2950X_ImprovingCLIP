from functools import partial
from io import open_code
from lib2to3.pgen2.token import tok_name
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import torch
from transformers import GPT2Tokenizer
import argparse
from tqdm import tqdm

import json


import sys
import os
sys.path.append(os.path.abspath("../training"))
from model import CustomCLIP
from dataset import MSCOCODataLoader
from train_clip import get_vision_model_name, get_text_model_name


def run_unibench(vision_model_size, text_model_size, lora, device):
    vision_model_name = get_vision_model_name(vision_model_size)
    text_model_name = get_text_model_name(text_model_size)

    custom_clip = CustomCLIP(
        text_model_name = text_model_name,
        vision_model_name = vision_model_name,
        embedding_dim=1024,
        use_peft = lora
    ).to(device)
    peft = "peft" if lora else "projection_only"
    full_model_name = f"clip_{text_model_size}-text_{vision_model_size}-vision_{peft}_seed44"
    print(full_model_name)
    ckpt = torch.load(f"/users/thua5/data/thua5/clip_weights/checkpoints/{full_model_name}.pt", weights_only=False) # load from the given path
    custom_clip.load_state_dict(ckpt["model_state_dict"]) # load to the model

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = partial(
        ClipModel,
        model=custom_clip,
        model_name=full_model_name.replace('-', '_'),
        tokenizer=tokenizer,
        input_resolution=224,
        logit_scale=custom_clip.logit_scale,
    )

    eval = Evaluator(
        num_workers=2,
    )

    eval.add_model(model=model)
    eval.update_benchmark_list(["vg_attribution", "vg_relation", "winoground", "cifar100", "flickr30k_order", "coco_order"])# , "imagenetc", "imagenet1k"])
    eval.update_model_list([full_model_name.replace('-', '_')])
    eval.evaluate()


def evaluate_coco_acc(vision_model_size, text_model_size, lora, device):
    vision_model_name = get_vision_model_name(vision_model_size)
    text_model_name = get_text_model_name(text_model_size)

    model = CustomCLIP(
        text_model_name = text_model_name,
        vision_model_name = vision_model_name,
        embedding_dim=1024,
        use_peft = lora
    ).to(device)
    peft = "peft" if lora else "projection_only"
    full_model_name = f"clip_{text_model_size}-text_{vision_model_size}-vision_{peft}_seed44"
    print(full_model_name)
    ckpt = torch.load(f"/users/thua5/data/thua5/clip_weights/checkpoints/{full_model_name}.pt", weights_only=False) # load from the given path
    model.load_state_dict(ckpt["model_state_dict"]) # load to the model

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    correct_i2t = 0
    correct_t2i = 0
    total_samples = 0


    eval_dataloader = MSCOCODataLoader(
        "/gpfs/data/superlab/datasets/coco/annotations/captions_val2017.json",
        "/gpfs/data/superlab/datasets/coco/val2017",
        text_model_name, 
        vision_model_name,
        batch_size = 32
    ).load_datasets()



    progress_bar = tqdm(eval_dataloader, desc="Evaluating Accuracy")

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch data to the target device
            batch_device = {key: val.to(device) for key, val in batch.items()}

            # Forward pass
            logits_per_image, logits_per_text = model(
                {
                    "input_ids": batch_device["input_ids"],
                    "attention_mask": batch_device["attention_mask"]
                },
                {"pixel_values": batch_device["pixel_values"]}
            )

            # Create target labels (diagonal 1s for correct matches)
            labels = torch.arange(len(batch["pixel_values"]), device=device)

            # Compute predictions and count correct matches
            preds_i2t = logits_per_image.argmax(dim=1)  # Image-to-Text predictions
            preds_t2i = logits_per_text.argmax(dim=1)  # Text-to-Image predictions
            correct_i2t += (preds_i2t == labels).sum().item()
            correct_t2i += (preds_t2i == labels).sum().item()
            total_samples += len(labels)

    accuracy_i2t = correct_i2t / total_samples
    accuracy_t2i = correct_t2i / total_samples

    results = {
        "accuracy_i2t": accuracy_i2t,
        "accuracy_t2i": accuracy_t2i,
        "average_accuracy": (accuracy_i2t + accuracy_t2i) / 2.0
    }

    print("Validation Results:")
    print(f"Image-to-Text Accuracy: {results['accuracy_i2t']}")
    print(f"Text-to-Image Accuracy: {results['accuracy_t2i']}")
    print(f"Average Accuracy: {results['average_accuracy']}")

    with open(f"/users/thua5/ssl_proj/eval_results/{full_model_name}.json", "w") as f:
        json.dump(results, f)

    return results



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Specify the arguments to use.")

    parser.add_argument(
        "--vision_model_size",
        type=str,
        help="The size of the vision model to be used"
    )

    parser.add_argument(
        "--text_model_size",
        type=str,
        help="The name of the text model to be used"
    )

    parser.add_argument(
        "--lora",
        action="store_true",
        help="Whether to use the lora fine-tuned model or projection-only fine-tuned model)"
    )

    # parser.add_argument('benchmarks', '--list', help='benchmark names separated by only commas', type=lambda s: [item for item in s.split(',')])

    args = parser.parse_args()

    # results = evaluate_coco_acc(args.vision_model_size, args.text_model_size, args.lora, device)
    run_unibench(args.vision_model_size, args.text_model_size, args.lora, device)

main()