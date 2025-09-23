
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor, PaliGemmaProcessor
from colpali_engine.models import ColPali, ColQwen2, ColQwen2_5, ColIdefics3, ColIdefics3Processor, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor(model_id: str):
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
        print(f"Loading ColIdefics3 model ({model_id})...")
        model = ColIdefics3.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
        processor = ColIdefics3Processor.from_pretrained(model_id, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model type in model_id: {model_id}")
        
    model.eval()
    return model, processor

@torch.no_grad()
def embed_image(pil_image: Image.Image, model, processor, model_id: str):
    """Embeds a single PIL image."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    if "clip" in model_id.lower():
        inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze(0).cpu().numpy()
    
    # Common processing for ColPali, ColQwen2, ColQwen2.5, ColSmol
    if any(name in model_id.lower() for name in ["colpali", "colqwen2", "colsmol"]):
        if isinstance(processor, PaliGemmaProcessor):
            batch_inputs = processor(text=["<image>"], images=[pil_image], return_tensors="pt")
        else:
            batch_inputs = processor(images=[pil_image], return_tensors="pt")
        model_device = next(model.parameters()).device
        batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
        out = model(**batch_inputs)
    elif "colqwen2.5" in model_id.lower():
        batch_inputs = processor.process_images([pil_image]).to(model.device)
        out = model(**batch_inputs)
    else: # Fallback for other potential models, assuming a standard processor call
        batch_inputs = processor(images=[pil_image], return_tensors="pt").to(model.device)
        out = model(**batch_inputs)

    emb_tensor = getattr(out, "image_embeds", getattr(out, "last_hidden_state", out if isinstance(out, torch.Tensor) else None))
    if emb_tensor is None:
        raise RuntimeError("Could not extract image embeddings from model outputs.")
    
    emb = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    return emb.detach().cpu().to(torch.float32).numpy()

@torch.no_grad()
def embed_text(text: str, model, processor, model_id: str):
    if "clip" in model_id.lower():
        inputs = processor(text=text, return_tensors="pt").to(DEVICE)
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze(0).cpu().numpy()

    if any(name in model_id.lower() for name in ["colpali", "colqwen2", "colsmol"]):
        batch_inputs = processor(text=[text], return_tensors="pt", truncation=False, padding=False)
        model_device = next(model.parameters()).device
        batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
        out = model(**batch_inputs)
    elif "colqwen2.5" in model_id.lower():
        batch_inputs = processor.process_queries([text]).to(model.device)
        out = model(**batch_inputs)
    else:
        batch_inputs = processor(text=[text], return_tensors="pt").to(model.device)
        out = model(**batch_inputs)

    emb_tensor = getattr(out, "text_embeds", getattr(out, "last_hidden_state", out if isinstance(out, torch.Tensor) else None))
    if emb_tensor is None:
        raise RuntimeError("Could not extract text embeddings from model outputs.")
        
    emb = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    return emb.detach().cpu().to(torch.float32).numpy()

@torch.no_grad()
def get_colqwen_vectors(data_input, model, processor, is_image=True):
    if is_image:
        if data_input.mode != "RGB": data_input = data_input.convert("RGB")
        if isinstance(processor, PaliGemmaProcessor):
            batch_inputs = processor(text=["<image>"], images=[data_input], return_tensors="pt")
        else:
            batch_inputs = processor(images=[data_input], return_tensors="pt")
    else:
        batch_inputs = processor(text=[data_input], return_tensors="pt")
    
    model_device = next(model.parameters()).device
    batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
    out = model(**batch_inputs)
    
    emb_tensor = getattr(out, "image_embeds" if is_image else "text_embeds", getattr(out, "last_hidden_state", out if isinstance(out, torch.Tensor) else None))
    if emb_tensor is None:
        raise RuntimeError("Could not extract base embeddings.")
        
    multi_vec = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 else emb_tensor
    mean_pooled = torch.mean(multi_vec, dim=0)
    
    return mean_pooled.to(torch.float32).cpu().numpy(), multi_vec.to(torch.float32).cpu().numpy()
