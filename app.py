import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from huggingface_hub import hf_hub_download

project_root = os.path.dirname(os.path.abspath(__file__))
for path in [
    project_root,
    os.path.join(project_root, 'src', 'scratch'),
    os.path.join(project_root, 'src', 'tfidf'),
    os.path.join(project_root, 'src', 'pretrained')
]:
    if path not in sys.path:
        sys.path.append(path)



from models.scratch import ScratchMCQModel
from src.scratch.preprocessed import BpeTokenizerScratch

st.set_page_config(
    page_title="Smart MCQ Solver Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------- CACHED MODEL LOADERS -----------------
def get_safe_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            x = torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")

@st.cache_resource
def load_tfidf_model():
    repo_id = "deeepakkk00/mcq-solvers"

    try:
        token = os.environ.get("HF_TOKEN")

        vec_path = hf_hub_download(
            repo_id=repo_id,
            filename="tfidf_vectorizer.joblib",
            token=token
        )

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="tfidf_model.joblib",
            token=token
        )

    except Exception as e:
        st.error(f"Failed to fetch TF-IDF files from Hugging Face: {e}")
        return None, None, None, None

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(model_path)

    return vectorizer, clf, repo_id, model_path


@st.cache_resource
def load_scratch_model():
    repo_id = "deeepakkk00/mcq-solvers"

    try:
        token = os.environ.get("HF_TOKEN")

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="scratch_model.pt",
            token=token
        )

        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="scratch_tokenizer.json",
            token=token
        )

    except Exception as e:
        st.error(f"Failed to fetch Custom GRU files from Hugging Face: {e}")
        return None, None, None, None, None

    device = get_safe_device()

    tok = BpeTokenizerScratch.load(tokenizer_path)

    model = ScratchMCQModel(
        vocab_size=tok.size,
        d_model=256,
        conv_ch=256,
        hidden=256,
        drop=0.3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return tok, model, device, repo_id, model_path


@st.cache_resource
def load_roberta_model():
    repo_id = "deeepakkk00/mcq-solvers"
    token = os.environ.get("HF_TOKEN")
    device = get_safe_device()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            token=token
        )

        model = AutoModelForMultipleChoice.from_pretrained(
            repo_id,
            token=token
        ).to(device)

        model.eval()

        return tokenizer, model, device, repo_id

    except Exception as e:
        st.error(f"Error loading RoBERTa model: {e}")
        return None, None, None, None



tfidf_vec, tfidf_clf, tfidf_repo, tfidf_path = load_tfidf_model()

scratch_tok, scratch_model, device, scratch_repo, scratch_path = load_scratch_model()

roberta_tok, roberta_model, roberta_device, roberta_repo = load_roberta_model()



# ----------------- MAIN INTERFACE -----------------
st.title("🧠 Smart MCQ Solver Dashboard")
st.caption("Evaluate state-of-the-art Deep Learning models on advanced Multiple Choice Questions — test your prompts and options interactively with real-time confidence scores.")

if torch.cuda.is_available() and get_safe_device().type == "cpu":
    st.warning("⚠️ NVIDIA GPU detected, but CUDA driver initialization failed (NVML driver error). The application has automatically fell back to **CPU execution** for stability.")

tab1, tab2 = st.tabs(["🎯  Solve Question", "📊  Model Status & Overview"])

# ----------------- TAB 1: SINGLE QUESTION SOLVER -----------------
with tab1:

    col_inputs, col_results = st.columns(2, gap="large")

    # -------- LEFT: INPUTS --------
    with col_inputs:
        with st.container(border=True):
            st.subheader("❓ Question")
            prompt = st.text_area(
                "Prompt / Question Context",
                value="Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options.",
                height=100
            )

        with st.container(border=True):
            st.subheader("📝 Answer Options")

            opt_A = st.text_input("Option A", value="Martin Heidegger believes that humans exist within a time continuum that is infinite.")
            opt_B = st.text_input("Option B", value="Martin Heidegger believes that humans do not exist inside time, but that they are time. The relationship to the past is a present awareness of having been.")
            opt_C = st.text_input("Option C", value="Martin Heidegger does not believe in the existence of time or that it has any effect.")
            opt_D = st.text_input("Option D", value="Martin Heidegger believes that the relationship between time and human existence is cyclical.")
            opt_E = st.text_input("Option E", value="Martin Heidegger believes that time is an illusion, and all events happen simultaneously.")

        options = {'A': opt_A, 'B': opt_B, 'C': opt_C, 'D': opt_D, 'E': opt_E}

        with st.container(border=True):
            st.subheader("⚙️ Run Inference")
            selected_model_name = st.selectbox(
                "Select Inference Model",
                [
                    "Model 3: Fine-Tuned RoBERTa-Large Model",
                    "Model 2: Custom Conv1D-GRU-Attention MCQ Model",
                    "Model 1: TF-IDF + SAGA Logistic Regression"
                ]
            )
            solve_clicked = st.button("Solve MCQ 🚀", use_container_width=True, type="primary")

    # -------- SOLVE (logic unchanged) --------
    if solve_clicked:
        if not prompt:
            st.error("Please enter a prompt/question.")
        elif not all(options.values()):
            st.error("Please fill in all five options.")
        else:
            with st.spinner("Running model inference..."):
                ranked, probs = [], []

                if "Model 1" in selected_model_name:
                    if tfidf_clf is None:
                        st.error("TF-IDF weights are missing. Please check models_weights/tfidf/ folder.")
                    else:
                        texts = [f"{prompt} [SEP] {options[opt]}" for opt in ['A', 'B', 'C', 'D', 'E']]
                        X = tfidf_vec.transform(texts)
                        raw_probs = tfidf_clf.predict_proba(X)[:, 1]
                        sorted_idx = np.argsort(-raw_probs)
                        ranked = [['A', 'B', 'C', 'D', 'E'][i] for i in sorted_idx]
                        probs = [raw_probs[i] for i in sorted_idx]

                elif "Model 2" in selected_model_name:
                    if scratch_model is None:
                        st.error("Custom GRU weights are missing. Please check models_weights/scratch/fold0/ folder.")
                    else:
                        ids_list, tids_list, mask_list = [], [], []
                        for opt in ['A', 'B', 'C', 'D', 'E']:
                            ids, tids, mask = scratch_tok.encode_pair(prompt, options[opt], max_len=192)
                            ids_list.append(ids)
                            tids_list.append(tids)
                            mask_list.append(mask)

                        ids_t = torch.tensor([ids_list], dtype=torch.long).to(device)
                        tids_t = torch.tensor([tids_list], dtype=torch.long).to(device)
                        mask_t = torch.tensor([mask_list], dtype=torch.long).to(device)

                        with torch.no_grad():
                            logits = scratch_model(ids_t, tids_t, mask_t)
                            raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                        sorted_idx = np.argsort(-raw_probs)
                        ranked = [['A', 'B', 'C', 'D', 'E'][i] for i in sorted_idx]
                        probs = [raw_probs[i] for i in sorted_idx]

                elif "Model 3" in selected_model_name:
                    if roberta_model is None:
                        st.error("RoBERTa model weights are missing. Please check models_weights/pretrained/fold_0/ folder.")
                    else:
                        option_labels = ['A', 'B', 'C', 'D', 'E']
                        first_sentences = [prompt] * 5
                        second_sentences = [options[opt] for opt in option_labels]

                        inputs = roberta_tok(
                            first_sentences,
                            second_sentences,
                            padding=True,
                            truncation=True,
                            max_length=386,
                            return_tensors="pt"
                        ).to(roberta_device)

                        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = roberta_model(**inputs)
                            logits = outputs.logits.cpu().numpy()[0]

                        raw_probs = np.exp(logits) / np.sum(np.exp(logits))
                        sorted_idx = np.argsort(-raw_probs)
                        ranked = [option_labels[i] for i in sorted_idx]
                        probs = [raw_probs[i] for i in sorted_idx]

                if ranked:
                    st.session_state["mcq_result"] = {
                        "ranked": ranked,
                        "probs": [float(p) for p in probs],
                        "options": options,
                    }

    # -------- RIGHT: RESULTS --------
    with col_results:
        with st.container(border=True):
            st.subheader("📊 Prediction Results")

            result = st.session_state.get("mcq_result")

            if result:
                ranked = result["ranked"]
                probs = result["probs"]
                opts = result["options"]
                top3_str = " ".join(ranked[:3])
                confident_answer = ranked[0]
                confident_pct = probs[0]

                m1, m2 = st.columns(2)
                m1.metric("MAP@3 Prediction", top3_str)
                m2.metric("Confident Answer", f"Option {confident_answer}", f"{confident_pct:.1%} confidence")

                st.divider()
                st.markdown("**🏆 Option Rankings**")

                medal = {0: "🥇", 1: "🥈", 2: "🥉"}
                for i, (label, prob) in enumerate(zip(ranked, probs)):
                    icon = medal.get(i, "▫️")
                    st.write(f"{icon} **Rank {i+1} — Option {label}**  ·  {prob:.2%}")
                    st.progress(float(prob))

                st.divider()
                with st.expander(f"📄 Option Text — Rank 1 ({confident_answer})", expanded=True):
                    st.write(opts[confident_answer])

            else:
                st.info("Fill in the question and options on the left, then click **Solve MCQ 🚀** to see predictions here.")

# ----------------- TAB 2: MODEL STATUS -----------------
with tab2:

    st.subheader("📊 Model Loading Status")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("#### Model 1")
            st.caption("TF-IDF + Logistic Regression")

            if tfidf_clf is not None:
                st.success("✅ Loaded Successfully")
                st.write(f"**Repository:** `{tfidf_repo}`")
                st.write(f"**File:** `{os.path.basename(tfidf_path)}`")
            else:
                st.error("Failed to Load")

    with col2:
        with st.container(border=True):
            st.markdown("#### Model 2")
            st.caption("Custom Conv1D + GRU + Attention")

            if scratch_model is not None:
                st.success("✅ Loaded Successfully")
                st.write(f"**Repository:** `{scratch_repo}`")
                st.write(f"**File:** `{os.path.basename(scratch_path)}`")
            else:
                st.error("Failed to Load")

    with col3:
        with st.container(border=True):
            st.markdown("#### Model 3")
            st.caption("Fine-Tuned RoBERTa Multiple Choice")

            if roberta_model is not None:
                st.success("✅ Loaded Successfully")
                st.write(f"**Repository:** `{roberta_repo}`")
                st.write("**Weights:** Automatically loaded by Transformers")
            else:
                st.error("Failed to Load")

    st.divider()

    backend = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"⚙️ Execution Backend: **{backend}**")

st.divider()
st.caption("© deeepak@2026 - Smart MCQ Solver Dashboard. All rights reserved.")
