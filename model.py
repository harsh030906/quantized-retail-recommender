import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig, prepare, convert

class RetailRecommenderLM(nn.Module):
    """
    Lightweight Causal Sequence Model for Shopping Basket Next-Item Recommendation.
    Processes sequences of item IDs and predicts the next most likely item.
    Supports both Dynamic Quantization and Static Post-Training Quantization (PTQ).
    """
    def __init__(self, vocab_size, d_model=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.item_embedding = nn.Embedding(vocab_size, d_model)
        
        # Sequence aggregation and representation projection
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len] of item token IDs
        # Embed items and aggregate basket context
        emb = self.item_embedding(x) # [batch_size, seq_len, d_model]
        pooled = emb.mean(dim=1)     # [batch_size, d_model]
        
        q_in = self.quant(pooled)
        h = self.relu(self.fc1(q_in))
        h = self.relu(self.fc2(h))
        logits = self.head(h)
        out = self.dequant(logits)
        return out

    def predict_next(self, cart_item_ids, top_k=5):
        """Predicts top-k next recommended items given current cart item ids."""
        self.eval()
        with torch.no_grad():
            if not isinstance(cart_item_ids, torch.Tensor):
                cart_item_ids = torch.tensor([cart_item_ids], dtype=torch.long)
            elif cart_item_ids.dim() == 1:
                cart_item_ids = cart_item_ids.unsqueeze(0)
            
            logits = self.forward(cart_item_ids)
            probs = F.softmax(logits[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, self.vocab_size))
            return top_indices.tolist(), top_probs.tolist()

    def probe_itemset(self, probe_item_id, candidate_item_ids):
        """
        Probing: Given a single probe item in cart, measures conditional likelihood
        of candidate co-purchased items vs control items.
        """
        self.eval()
        with torch.no_grad():
            inp = torch.tensor([[probe_item_id]], dtype=torch.long)
            logits = self.forward(inp)
            probs = F.softmax(logits[0], dim=-1)
            return {item_id: float(probs[item_id]) for item_id in candidate_item_ids}


def apply_dynamic_quantization(model):
    """Applies Dynamic Post-Training Quantization to Linear layers (INT8)."""
    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def apply_static_quantization(model, calibration_loader):
    """Applies Static Post-Training Quantization with calibration on basket data."""
    import copy
    torch.backends.quantized.engine = 'onednn'
    model_to_quant = copy.deepcopy(model)
    model_to_quant.eval()
    
    # Configure linear layers for static quantization
    model_to_quant.qconfig = None
    model_to_quant.quant.qconfig = get_default_qconfig('onednn')
    model_to_quant.fc1.qconfig = get_default_qconfig('onednn')
    model_to_quant.relu.qconfig = get_default_qconfig('onednn')
    model_to_quant.fc2.qconfig = get_default_qconfig('onednn')
    model_to_quant.head.qconfig = get_default_qconfig('onednn')
    model_to_quant.dequant.qconfig = get_default_qconfig('onednn')

    prepared = prepare(model_to_quant)
    with torch.no_grad():
        for x, _ in calibration_loader:
            prepared(x)
            
    quantized = convert(prepared)
    return quantized
