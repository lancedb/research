import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor
from colpali_engine.models import ColPali, ColQwen2, ColQwen2_5, ColIdefics3, ColIdefics3Processor, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor(model_id: str):
    """
    Loads the specified model and its corresponding processor based on the model_id.
    This function correctly handles models that require specific processor classes.
    """
    dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    
    if "clip" in model_id.lower():
        print(f"Loading CLIP model ({model_id})...")
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)
    elif "colpali" in model_id.lower():
        print(f"Loading ColPali model ({model_id})...")
        model = ColPali.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    elif "colqwen2.5" in model_id.lower():
        print(f"Loading ColQwen2.5 model ({model_id})...")
        model = ColQwen2_5.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None
        )
        processor = ColQwen2_5_Processor.from_pretrained(model_id, trust_remote_code=True)
    elif "colqwen2" in model_id.lower():
        print(f"Loading ColQwen2 model ({model_id})...")
        model = ColQwen2.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    elif "colsmol" in model_id.lower():
        print(f"Loading ColIdefics3 (colsmol) model ({model_id})...")
        model = ColIdefics3.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(DEVICE)
        processor = ColIdefics3Processor.from_pretrained(model_id, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model type in model_id: {model_id}")
        
    model.eval()
    return model, processor

@torch.no_grad()
def embed_image(pil_image: Image.Image, model, processor, model_id: str):
    """
    Embeds a single PIL image using the correct processor method for the model.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    model_device = next(model.parameters()).device

    if "clip" in model_id.lower():
        inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
        out = model.get_image_features(**inputs)
        out /= out.norm(p=2, dim=-1, keepdim=True)
        emb_tensor = out
    elif "colsmol" in model_id.lower():
        batch_inputs = processor.process_images([pil_image]).to(model_device)
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "image_embeds", out)
    elif "colqwen2.5" in model_id.lower():
        batch_inputs = processor.process_images([pil_image]).to(model_device)
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "image_embeds", out)
    else: # Default for colpali, colqwen2
        batch_inputs = processor(images=[pil_image], return_tensors="pt", truncation=False, padding=False)
        batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "image_embeds", out)

    if emb_tensor is None:
        raise RuntimeError(f"Could not extract image embeddings from model outputs for {model_id}.")
    
    emb = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    return emb.detach().cpu().to(torch.float32).numpy()

@torch.no_grad()
def embed_text(text: str, model, processor, model_id: str):
    """
    Embeds a single text string using the correct processor method for the model.
    """
    model_device = next(model.parameters()).device

    if "clip" in model_id.lower():
        inputs = processor(text=text, return_tensors="pt").to(DEVICE)
        out = model.get_text_features(**inputs)
        out /= out.norm(p=2, dim=-1, keepdim=True)
        emb_tensor = out
    elif "colsmol" in model_id.lower():
        batch_inputs = processor.process_queries([text]).to(model_device)
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "text_embeds", out)
    elif "colqwen2.5" in model_id.lower():
        batch_inputs = processor.process_queries([text]).to(model_device)
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "text_embeds", out)
    else: # Default for colpali, colqwen2
        batch_inputs = processor(text=[text], return_tensors="pt")
        batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
        out = model(**batch_inputs)
        emb_tensor = getattr(out, "text_embeds", out)

    if emb_tensor is None:
        raise RuntimeError(f"Could not extract text embeddings from model outputs for {model_id}.")
        
    emb = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    return emb.detach().cpu().to(torch.float32).numpy()

@torch.no_grad()
def get_colqwen_vectors(data_input, model, processor, model_id, is_image=True):
    """
    Gets vectors for ColQwen models, using the correct processor method based on model_id.
    """
    model_device = next(model.parameters()).device

    if is_image:
        if data_input.mode != "RGB": data_input = data_input.convert("RGB")
        if "colqwen2.5" in model_id.lower():
            batch_inputs = processor.process_images([data_input]).to(model_device)
        else: # Default for colqwen2
            batch_inputs = processor(images=[data_input], return_tensors="pt", truncation=False, padding=False)
            batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
    else: # is_text
        if "colqwen2.5" in model_id.lower():
            batch_inputs = processor.process_queries([data_input]).to(model_device)
        else: # Default for colqwen2
            batch_inputs = processor(text=[data_input], return_tensors="pt")
            batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}

    out = model(**batch_inputs)
    emb_tensor = getattr(out, "image_embeds" if is_image else "text_embeds", out)
    if emb_tensor is None:
        raise RuntimeError(f"Could not extract base embeddings for {model_id}.")
        
    multi_vec = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    mean_pooled = torch.mean(multi_vec, dim=0)
    
    return mean_pooled.to(torch.float32).cpu().numpy(), multi_vec.to(torch.float32).cpu().numpy()