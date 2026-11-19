import os
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import RetailRecommenderLM, apply_dynamic_quantization, apply_static_quantization

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------------------
# Milestone 1: Synthetic Shopping Basket Data
# -------------------------------------------------------------
VOCAB = [
    "<PAD>",
    # Breakfast cluster
    "Milk", "Bread", "Eggs", "Butter", "Coffee", "Cereal", "Orange Juice",
    # Italian Dinner cluster
    "Pasta", "Tomato Sauce", "Parmesan Cheese", "Olive Oil", "Garlic", "Basil",
    # Snacks & Party cluster
    "Potato Chips", "Salsa", "Soda", "Popcorn", "Pretzels", "Dip",
    # Baking cluster
    "Flour", "Sugar", "Baking Powder", "Chocolate Chips", "Vanilla Extract",
    # Fresh Fruit cluster
    "Apple", "Banana", "Yogurt", "Granola", "Honey"
]

ITEM_TO_ID = {item: i for i, item in enumerate(VOCAB)}
ID_TO_ITEM = {i: item for i, item in enumerate(VOCAB)}

CLUSTERS = [
    ["Milk", "Bread", "Eggs", "Butter", "Coffee", "Cereal", "Orange Juice"],
    ["Pasta", "Tomato Sauce", "Parmesan Cheese", "Olive Oil", "Garlic", "Basil"],
    ["Potato Chips", "Salsa", "Soda", "Popcorn", "Pretzels", "Dip"],
    ["Flour", "Sugar", "Baking Powder", "Chocolate Chips", "Vanilla Extract"],
    ["Apple", "Banana", "Yogurt", "Granola", "Honey"]
]

ALL_ITEMS = [item for c in CLUSTERS for item in c]

def generate_synthetic_baskets(num_baskets=1500):
    baskets = []
    for _ in range(num_baskets):
        # Pick primary theme cluster
        theme = random.choice(CLUSTERS)
        basket_size = random.randint(3, 6)
        
        # Sample items from primary theme
        theme_items = random.sample(theme, min(len(theme), basket_size))
        
        # 20% chance to add 1 random noise item from any cluster (cross-category purchase)
        if random.random() < 0.2:
            noise_item = random.choice(ALL_ITEMS)
            if noise_item not in theme_items:
                theme_items.append(noise_item)
                
        random.shuffle(theme_items)
        baskets.append(theme_items)
    return baskets

class BasketNextItemDataset(Dataset):
    def __init__(self, baskets, max_len=6):
        self.samples = []
        for basket in baskets:
            # Create prefix -> next-item pairs: e.g. [A, B] -> C
            token_ids = [ITEM_TO_ID[item] for item in basket]
            for i in range(1, len(token_ids)):
                prefix = token_ids[:i]
                target = token_ids[i]
                # Pad prefix
                padded_prefix = prefix + [ITEM_TO_ID["<PAD>"]] * (max_len - len(prefix))
                self.samples.append((padded_prefix[:max_len], target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, target = self.samples[idx]
        return torch.tensor(prefix, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def measure_model_size_kb(model):
    """Measures the in-memory state dict storage size in Kilobytes."""
    temp_path = "temp_measure.pt"
    torch.save(model.state_dict(), temp_path)
    size_kb = os.path.getsize(temp_path) / 1024.0
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return size_kb

def evaluate_model(model, test_loader):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            start_t = time.perf_counter()
            outputs = model(inputs)
            latency = (time.perf_counter() - start_t) * 1000.0  # ms
            latencies.append(latency / inputs.size(0))

            _, top1_preds = torch.topk(outputs, k=1, dim=-1)
            _, top3_preds = torch.topk(outputs, k=3, dim=-1)

            correct_top1 += (top1_preds.squeeze(-1) == targets).sum().item()
            correct_top3 += (top3_preds == targets.unsqueeze(1)).any(dim=-1).sum().item()
            total += targets.size(0)

    top1_acc = (correct_top1 / total) * 100.0
    top3_acc = (correct_top3 / total) * 100.0
    avg_latency_ms = float(np.mean(latencies))
    return top1_acc, top3_acc, avg_latency_ms

def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 65)
    print(" QUANTIZED RETAIL RECOMMENDER: TRAINING & BENCHMARK")
    print("=" * 65)

    # 1. Prepare Synthetic Basket Data
    print("\n[Milestone 1] Generating synthetic retail basket data...")
    raw_baskets = generate_synthetic_baskets(num_baskets=2000)
    split_idx = int(0.8 * len(raw_baskets))
    train_baskets = raw_baskets[:split_idx]
    test_baskets = raw_baskets[split_idx:]

    train_dataset = BasketNextItemDataset(train_baskets)
    test_dataset = BasketNextItemDataset(test_baskets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    calib_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    print(f" -> Generated {len(raw_baskets)} baskets.")
    print(f" -> Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f" -> Example Basket: {raw_baskets[0]}")

    # 2. Train Small LM on Basket Data
    print("\n[Milestone 2] Training Small Language Model on basket data...")
    vocab_size = len(VOCAB)
    baseline_model = RetailRecommenderLM(vocab_size=vocab_size, d_model=64, hidden_dim=128)
    criterion = nn.CrossEntropyLoss(ignore_index=ITEM_TO_ID["<PAD>"])
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 8
    for epoch in range(1, epochs + 1):
        baseline_model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = baseline_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"    Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # 3. Next-Item Prediction & Itemset Probing
    print("\n[Milestone 3] Testing Next-Item Prediction & Itemset Probing...")
    test_cart = ["Pasta", "Tomato Sauce"]
    cart_ids = [ITEM_TO_ID[it] for it in test_cart]
    rec_ids, rec_probs = baseline_model.predict_next(cart_ids, top_k=3)
    rec_items = [f"{ID_TO_ITEM[rid]} ({prob*100:.1f}%)" for rid, prob in zip(rec_ids, rec_probs)]
    print(f" -> Cart: {test_cart} -> Next-Item Recommendations: {', '.join(rec_items)}")

    # Probing: Probe Pasta for expected Italian dinner co-items vs unrelated snack
    pasta_id = ITEM_TO_ID["Pasta"]
    cheese_id = ITEM_TO_ID["Parmesan Cheese"]
    chips_id = ITEM_TO_ID["Potato Chips"]
    probe_res = baseline_model.probe_itemset(pasta_id, [cheese_id, chips_id])
    print(f" -> Itemset Probing [Cart: 'Pasta']:")
    print(f"    - P('Parmesan Cheese' | 'Pasta'): {probe_res[cheese_id]*100:.2f}% (Expected Co-item)")
    print(f"    - P('Potato Chips' | 'Pasta'):    {probe_res[chips_id]*100:.2f}% (Unrelated Item)")

    # 4. Measure Baseline Metrics
    print("\n[Milestone 4] Measuring Baseline (FP32) Model Metrics...")
    baseline_size_kb = measure_model_size_kb(baseline_model)
    b_top1, b_top3, b_latency = evaluate_model(baseline_model, test_loader)
    print(f" -> Baseline Size:     {baseline_size_kb:.2f} KB ({baseline_size_kb/1024.0:.3f} MB)")
    print(f" -> Baseline Top-1 Acc: {b_top1:.2f}%")
    print(f" -> Baseline Top-3 Acc: {b_top3:.2f}%")
    print(f" -> Baseline Latency:  {b_latency:.3f} ms / item")

    # 5. Apply Dynamic and Static Post-Training Quantization
    print("\n[Milestone 5] Applying Dynamic and Static Post-Training Quantization (INT8)...")
    dyn_quant_model = apply_dynamic_quantization(baseline_model)
    stat_quant_model = apply_static_quantization(baseline_model, calib_loader)

    # 6. Demonstrate Size Reduction & Accuracy Impact
    print("\n[Milestone 6] Evaluating Quantized Models against Baseline...")
    dyn_size_kb = measure_model_size_kb(dyn_quant_model)
    d_top1, d_top3, d_latency = evaluate_model(dyn_quant_model, test_loader)

    stat_size_kb = measure_model_size_kb(stat_quant_model)
    s_top1, s_top3, s_latency = evaluate_model(stat_quant_model, test_loader)

    dyn_size_reduction = ((baseline_size_kb - dyn_size_kb) / baseline_size_kb) * 100.0
    stat_size_reduction = ((baseline_size_kb - stat_size_kb) / baseline_size_kb) * 100.0

    print("\n" + "=" * 65)
    print(" FINAL QUANTIZATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<24} | {'Baseline (FP32)':<16} | {'Dynamic INT8':<14} | {'Static INT8':<14}")
    print("-" * 75)
    print(f"{'Model Size (KB)':<24} | {baseline_size_kb:<16.2f} | {dyn_size_kb:<14.2f} | {stat_size_kb:<14.2f}")
    print(f"{'Size Reduction (%)':<24} | {'0.0%':<16} | {dyn_size_reduction:<13.1f}% | {stat_size_reduction:<13.1f}%")
    print(f"{'Top-1 Accuracy (%)':<24} | {b_top1:<16.2f} | {d_top1:<14.2f} | {s_top1:<14.2f}")
    print(f"{'Top-3 Accuracy (%)':<24} | {b_top3:<16.2f} | {d_top3:<14.2f} | {s_top3:<14.2f}")
    print(f"{'Inference Latency (ms)':<24} | {b_latency:<16.3f} | {d_latency:<14.3f} | {s_latency:<14.3f}")
    print("=" * 65)

    # Save artifacts for the Streamlit App
    torch.save(baseline_model.state_dict(), os.path.join(save_dir, "baseline_model.pt"))
    torch.save(dyn_quant_model.state_dict(), os.path.join(save_dir, "quantized_dynamic_model.pt"))
    torch.save(stat_quant_model.state_dict(), os.path.join(save_dir, "quantized_static_model.pt"))

    summary_data = {
        "vocab": VOCAB,
        "clusters": CLUSTERS,
        "metrics": {
            "baseline": {
                "size_kb": round(baseline_size_kb, 2),
                "top1_acc": round(b_top1, 2),
                "top3_acc": round(b_top3, 2),
                "latency_ms": round(b_latency, 3),
                "size_reduction_pct": 0.0
            },
            "dynamic_int8": {
                "size_kb": round(dyn_size_kb, 2),
                "top1_acc": round(d_top1, 2),
                "top3_acc": round(d_top3, 2),
                "latency_ms": round(d_latency, 3),
                "size_reduction_pct": round(dyn_size_reduction, 2)
            },
            "static_int8": {
                "size_kb": round(stat_size_kb, 2),
                "top1_acc": round(s_top1, 2),
                "top3_acc": round(s_top3, 2),
                "latency_ms": round(s_latency, 3),
                "size_reduction_pct": round(stat_size_reduction, 2)
            }
        }
    }

    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[DONE] Saved models and metrics summary to:\n -> {save_dir}")

if __name__ == "__main__":
    main()
