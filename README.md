# Quantized Retail Recommender

A lightweight, edge-ready retail recommendation system that models shopping basket sequences using a small Language Model (LM) and compresses it using Post-Training Quantization (PTQ).

---

## 🎯 Project Overview & Milestones

This project fulfills all 6 required milestones:

| # | Milestone | Implementation |
|---|---|---|
| **1** | **Prepare synthetic basket data** | Generated 2,000 realistic market baskets partitioned into thematic retail clusters (Breakfast, Italian Dinner, Snacks, Baking, Produce) with cross-cluster noise items. Formatted as prefix-to-next-item prediction sequences. |
| **2** | **Pretrain a small LM on basket data** | Implemented `RetailRecommenderLM` in PyTorch with embedding layers, linear projections, and vocabulary next-item prediction head, trained using Cross-Entropy loss. |
| **3** | **Next-item prediction & itemset probing** | Implemented `predict_next()` for top-K item recommendation and `probe_itemset()` to evaluate conditional co-occurrence probabilities (e.g. $P(\text{Parmesan Cheese} \mid \text{Pasta})$ vs unrelated controls). |
| **4** | **Measure baseline size and accuracy** | Recorded baseline FP32 model disk size (122.58 KB), Top-1 Accuracy (34.02%), Top-3 Accuracy (77.73%), and average latency. |
| **5** | **Apply dynamic and static PTQ** | Applied Dynamic Post-Training Quantization (`torch.ao.quantization.quantize_dynamic`) and Static Post-Training Quantization with calibration on representative shopping baskets. |
| **6** | **Demonstrate size reduction & accuracy impact** | Benchmarked and saved comparison metrics proving a **~66.3% size reduction** with negligible accuracy impact. |

---

## 📊 Quantization Benchmark Results

| Metric | Baseline (FP32) | Dynamic INT8 | Static INT8 |
| :--- | :---: | :---: | :---: |
| **Model Size** | **122.58 KB** | **41.27 KB** | **47.60 KB** |
| **Size Reduction** | Baseline (0%) | **-66.3%** | **-61.2%** |
| **Top-1 Accuracy** | 34.02% | 34.02% | 33.61% |
| **Top-3 Accuracy** | 77.73% | 77.46% | 78.14% |
| **Inference Latency** | 0.007 ms | 0.029 ms | 0.021 ms |

---

## 🚀 How to Run

### 1. (Optional) Re-train and Re-quantize
The models are already pre-trained and saved in this folder. If you ever want to re-run the benchmark from scratch:
```bash
python train_and_quantize.py
```

### 2. Launch the Streamlit Web App
Run the interactive demonstration UI:
```bash
streamlit run app.py
```

The web application provides:
- **🛍️ Cart Recommender**: Select items or presets and inspect top next-item recommendations with probability bars.
- **🔬 Itemset Probing**: Probe how strongly the model associates paired items (e.g. Pasta $\rightarrow$ Parmesan vs Pasta $\rightarrow$ Chips).
- **📊 Quantization Benchmark**: View side-by-side metric cards and verification tables.
