import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import torch

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

HF_MODEL = "Abhay-learns/distilbert-genre"
THRESHOLD = 0.35

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=HF_MODEL, top_k=None, device=device)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🎬 Movie Genre Classifier")
st.markdown(
    "Predict genres from a movie plot description using a fine-tuned **DistilBERT** model "
    "trained on 182k IMDB movies across 20 genres."
)
st.markdown("---")

# ── Input ────────────────────────────────────────────────────────────────────
description = st.text_area(
    "Movie Description",
    placeholder="Enter a movie plot description...",
    height=120,
)

example = "A detective investigates a series of brutal murders in a small coastal town, uncovering dark secrets among the locals."
if st.button("Try an example"):
    description = example
    st.session_state["example_text"] = example

if "example_text" in st.session_state and not description:
    description = st.session_state["example_text"]

predict_btn = st.button("Predict Genres", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn and description.strip():
    with st.spinner("Loading model and predicting..."):
        classifier = load_model()
        raw = classifier(description)[0]

    # Filter and sort
    filtered = sorted(
        [p for p in raw if p["score"] > THRESHOLD],
        key=lambda x: x["score"],
        reverse=True
    )

    if not filtered:
        st.warning("No genres predicted above threshold. Try a more detailed description.")
    else:
        st.markdown("### Predicted Genres")

        # Genre badges
        badge_html = " ".join(
            f'<span style="background:#1f77b4;color:white;padding:4px 12px;'
            f'border-radius:12px;margin:3px;display:inline-block;font-size:15px">'
            f'{p["label"]} {p["score"]:.0%}</span>'
            for p in filtered
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("")

        # Bar chart
        labels = [p["label"] for p in filtered]
        scores = [p["score"] for p in filtered]

        fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.55)))
        bars = ax.barh(labels[::-1], scores[::-1], color="#1f77b4", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.axvline(THRESHOLD, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"Threshold ({THRESHOLD})")
        ax.legend(fontsize=9)

        for bar, score in zip(bars, scores[::-1]):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.0%}", va="center", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

elif predict_btn:
    st.warning("Please enter a movie description first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Model: [Abhay-learns/distilbert-genre](https://huggingface.co/Abhay-learns/distilbert-genre) · "
    "Code: [GitHub](https://github.com/AbhAy120204/movie-classification)",
    unsafe_allow_html=False
)
