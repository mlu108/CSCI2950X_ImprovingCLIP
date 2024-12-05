from functools import partial
from io import open_code
from lib2to3.pgen2.token import tok_name
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import torch
from transformers import GPT2Tokenizer
import argparse

import sys
import os
sys.path.append(os.path.abspath("../training"))
from model import CustomCLIP
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
    ckpt = torch.load(f"../../../thua5/clip_weights/checkpoints/{full_model_name}.pt", weights_only=False) # load from the given path
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

    eval = Evaluator()

    eval.add_model(model=model)
    eval.update_benchmark_list(["imagenetv2", "resisc45", "dtd", "vg_attribution", "dspr_x_position", "dspr_y_position", "countbench"])# "winoground", "imagenetc", "imagenet1k"])
    eval.update_model_list([full_model_name.replace('-', '_')])
    eval.evaluate()

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

    run_unibench(args.vision_model_size, args.text_model_size, args.lora, device)

main()
