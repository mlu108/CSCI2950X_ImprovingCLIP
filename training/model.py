import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from transformers import AutoModel
from peft import get_peft_model, LoraConfig


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)


class CustomCLIP(nn.Module):
    def __init__(self, text_model_name, vision_model_name, embedding_dim=768, use_peft=False, pretrained_projector_ckpt_path=""):
        super().__init__()
        # Load models from Huggingface
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)


        # Check if PEFT (LoRA) is used
        if use_peft:
            lora_config_text = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.1, target_modules='all-linear')
            lora_config_vision = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.1, target_modules='all-linear')

            # Apply LoRA to both text and vision models
            self.text_model = get_peft_model(self.text_model, lora_config_text)
            self.vision_model = get_peft_model(self.vision_model, lora_config_vision)
        else:
            # Freeze the weights of the encoders if PEFT is not used
            for param in self.text_model.parameters():
                param.requires_grad = False

            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.use_peft = use_peft

        # Get hidden size from both models
        text_hidden_size = self.text_model.config.hidden_size
        vision_hidden_size = self.vision_model.config.hidden_size

        # Define projection heads to map to joint space
        self.text_projection = ProjectionHead(text_hidden_size, embedding_dim)
        self.vision_projection = ProjectionHead(vision_hidden_size, embedding_dim)

        if pretrained_projector_ckpt_path != "":
            #### FOR NOW, Initializing the projection layers with pretrained weights:
            ckpt = torch.load(pretrained_projector_ckpt_path, weights_only=False)
            self.text_projection.load_state_dict({".".join(k.split(".")[1:]):v for k,v in ckpt['model_state_dict'].items() if 'text_projection' in k})
            self.vision_projection.load_state_dict({".".join(k.split(".")[1:]):v for k,v in ckpt['model_state_dict'].items() if 'vision_projection' in k})

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, vision_inputs):
        """Encode images and project to joint embedding space."""

        if type(vision_inputs) != dict:
            vision_inputs = {"pixel_values": vision_inputs}

        if self.use_peft:
            vision_embeddings = self.vision_model(**vision_inputs).last_hidden_state[:, 0, :]  # CLS token
        else:
            with torch.no_grad():
                vision_embeddings = self.vision_model(**vision_inputs).last_hidden_state[:, 0, :]  # CLS token
        
        vision_projected = self.vision_projection(vision_embeddings)
        vision_projected = F.normalize(vision_projected, dim=-1)
        
        return vision_projected

    def encode_text(self, text_inputs):
        """Encode text and project to joint embedding space."""
        
        if self.use_peft:
            text_embeddings = self.text_model(**text_inputs).last_hidden_state[:, -1, :]  # EOS token for gpt2
        else:
            with torch.no_grad():
                text_embeddings = self.text_model(**text_inputs).last_hidden_state[:, -1, :]  # EOS token for gpt2
        
        text_projected = self.text_projection(text_embeddings)
        text_projected = F.normalize(text_projected, dim=-1)
        
        return text_projected

    def forward(self, text_inputs, vision_inputs):
        """
        Compute the encodings of text and vision inputs

        Compute pairwise cosine similarities between text and image embeddings.
        Returns a matrix where each element [i, j] represents the cosine similarity
        between the i-th text input and the j-th image input.
        """

        image_features = self.encode_image(vision_inputs)
        text_features = self.encode_text(text_inputs)

        # Acknowledgement: the logits computation are borrowed from the official CLIP implementation
        # reference: https://github.com/openai/CLIP/blob/main/clip/model.py 

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
