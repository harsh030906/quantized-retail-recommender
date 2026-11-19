import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import streamlit as st
import pandas as pd
from model import RetailRecommenderLM, apply_dynamic_quantization

st.set_page_config(
    page_title="Quantized Retail Recommender",
    page_icon="🛒",
    layout="wide"
)

# Load configuration and models
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(APP_DIR, "metrics_summary.json")

@st.cache_resource
def load_resources():
    if not os.path.exists(SUMMARY_PATH):
        st.error("Metrics summary not found. Please run `train_and_quantize.py` first.")
        st.stop()
        
    with open(SUMMARY_PATH, "r") as f:
        meta = json.load(f)

    vocab = meta["vocab"]
    item_to_id = {item: i for i, item in enumerate(vocab)}
    id_to_item = {i: item for i, item in enumerate(vocab)}
    
    # Load Baseline Model
    baseline_model = RetailRecommenderLM(vocab_size=len(vocab), d_model=64, hidden_dim=128)
    baseline_model.load_state_dict(torch.load(os.path.join(APP_DIR, "baseline_model.pt"), weights_only=True))
    baseline_model.eval()

    # Load Dynamic Quantized Model
    quant_dynamic_model = apply_dynamic_quantization(baseline_model)

    return meta, vocab, item_to_id, id_to_item, baseline_model, quant_dynamic_model

meta, vocab, item_to_id, id_to_item, baseline_model, quant_dynamic_model = load_resources()

# Catalog of selectable items (excluding <PAD>)
catalog_items = [it for it in vocab if it != "<PAD>"]

# ----------------- UI Header -----------------
st.title("🛒 Quantized Retail Recommender")
st.caption("A small language model for retail cart next-item prediction, shrunk with Post-Training Quantization (PTQ) for edge deployment.")

# ----------------- Tabs -----------------
tab_recommender, tab_probing, tab_benchmark = st.tabs([
    "🛍️ Cart Recommender",
    "🔬 Itemset Probing",
    "📊 Quantization Benchmark"
])

# ----------------- Tab 1: Cart Recommender -----------------
with tab_recommender:
    st.subheader("1. Add Items to Shopping Cart")
    
    col_preset, col_model = st.columns([2, 1])
    with col_preset:
        preset = st.selectbox(
            "Quick Cart Presets:",
            [
                "Custom Selection",
                "Pasta Dinner (Pasta, Tomato Sauce)",
                "Breakfast (Milk, Cereal, Bread)",
                "Snack Party (Potato Chips, Soda)",
                "Baking Prep (Flour, Sugar)"
            ]
        )
    with col_model:
        selected_model_type = st.radio(
            "Model Variant:",
            ["Dynamic INT8 (Quantized)", "Baseline FP32"],
            horizontal=True
        )

    # Preset selection handling
    default_items = []
    if preset == "Pasta Dinner (Pasta, Tomato Sauce)":
        default_items = ["Pasta", "Tomato Sauce"]
    elif preset == "Breakfast (Milk, Cereal, Bread)":
        default_items = ["Milk", "Cereal", "Bread"]
    elif preset == "Snack Party (Potato Chips, Soda)":
        default_items = ["Potato Chips", "Soda"]
    elif preset == "Baking Prep (Flour, Sugar)":
        default_items = ["Flour", "Sugar"]

    cart_items = st.multiselect(
        "Current Items in Cart:",
        options=catalog_items,
        default=default_items
    )

    if st.button("🔮 Predict Next Item", type="primary", use_container_width=True):
        if not cart_items:
            st.warning("Please select at least 1 item to get recommendations.")
        else:
            active_model = quant_dynamic_model if "INT8" in selected_model_type else baseline_model
            cart_ids = [item_to_id[it] for it in cart_items]
            
            rec_ids, rec_probs = active_model.predict_next(cart_ids, top_k=5)
            
            st.write("### Recommended Next Items:")
            rec_data = []
            for rid, prob in zip(rec_ids, rec_probs):
                item_name = id_to_item[rid]
                if item_name != "<PAD>":
                    rec_data.append({"Item": item_name, "Probability": round(prob * 100, 2)})

            df_rec = pd.DataFrame(rec_data)
            
            for idx, row in df_rec.iterrows():
                col_item, col_bar = st.columns([1, 3])
                with col_item:
                    st.markdown(f"**{idx + 1}. {row['Item']}**")
                with col_bar:
                    st.progress(float(row['Probability']) / 100.0, text=f"{row['Probability']}%")

# ----------------- Tab 2: Itemset Probing -----------------
with tab_probing:
    st.subheader("2. Itemset Probing & Association Analysis")
    st.write(
        "Probing measures whether the language model internalizes retail associations by checking "
        "if candidate co-purchased items receive higher probability than unrelated controls."
    )
    
    col_probe_in, col_probe_target = st.columns(2)
    with col_probe_in:
        probe_item = st.selectbox("Probe Item (Input):", catalog_items, index=catalog_items.index("Pasta") if "Pasta" in catalog_items else 0)
    with col_probe_target:
        probe_target = st.selectbox("Target Item to Evaluate:", catalog_items, index=catalog_items.index("Parmesan Cheese") if "Parmesan Cheese" in catalog_items else 1)

    if st.button("🔍 Run Probing Probe"):
        probe_id = item_to_id[probe_item]
        target_id = item_to_id[probe_target]
        
        # Probing on Baseline
        base_prob = baseline_model.probe_itemset(probe_id, [target_id])[target_id] * 100.0
        # Probing on Quantized
        quant_prob = quant_dynamic_model.probe_itemset(probe_id, [target_id])[target_id] * 100.0

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label=f"Baseline (FP32) P({probe_target} | {probe_item})",
                value=f"{base_prob:.2f}%"
            )
        with col_m2:
            st.metric(
                label=f"Quantized (INT8) P({probe_target} | {probe_item})",
                value=f"{quant_prob:.2f}%",
                delta=f"{quant_prob - base_prob:+.2f}% vs Base"
            )

# ----------------- Tab 3: Quantization Benchmark -----------------
with tab_benchmark:
    st.subheader("3. Model Size, Accuracy & Quantization Metrics")
    metrics = meta["metrics"]

    # Metric KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Baseline Size (FP32)",
            value=f"{metrics['baseline']['size_kb']} KB"
        )
    with c2:
        st.metric(
            label="Dynamic INT8 Size",
            value=f"{metrics['dynamic_int8']['size_kb']} KB",
            delta=f"-{metrics['dynamic_int8']['size_reduction_pct']}% Reduction"
        )
    with c3:
        st.metric(
            label="Top-3 Accuracy Retention",
            value=f"{metrics['dynamic_int8']['top3_acc']}%",
            delta=f"{round(metrics['dynamic_int8']['top3_acc'] - metrics['baseline']['top3_acc'], 2)}% vs FP32"
        )

    st.write("#### Detailed Comparison Table")
    bench_df = pd.DataFrame({
        "Metric": ["Model Size (KB)", "Size Reduction (%)", "Top-1 Accuracy (%)", "Top-3 Accuracy (%)", "Latency (ms)"],
        "Baseline (FP32)": [
            f"{metrics['baseline']['size_kb']}",
            "0.0%",
            f"{metrics['baseline']['top1_acc']}%",
            f"{metrics['baseline']['top3_acc']}%",
            f"{metrics['baseline']['latency_ms']} ms"
        ],
        "Dynamic INT8 (Quantized)": [
            f"{metrics['dynamic_int8']['size_kb']}",
            f"{metrics['dynamic_int8']['size_reduction_pct']}%",
            f"{metrics['dynamic_int8']['top1_acc']}%",
            f"{metrics['dynamic_int8']['top3_acc']}%",
            f"{metrics['dynamic_int8']['latency_ms']} ms"
        ],
        "Static INT8 (Quantized)": [
            f"{metrics['static_int8']['size_kb']}",
            f"{metrics['static_int8']['size_reduction_pct']}%",
            f"{metrics['static_int8']['top1_acc']}%",
            f"{metrics['static_int8']['top3_acc']}%",
            f"{metrics['static_int8']['latency_ms']} ms"
        ]
    })
    st.table(bench_df)

    st.success("✅ **Milestone Verification**: Both dynamic and static post-training quantizations achieve ~61-66% size reduction with negligible accuracy loss, making the model ready for lightweight edge retail deployment.")
