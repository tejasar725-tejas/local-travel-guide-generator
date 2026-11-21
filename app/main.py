from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from app.core import baseline_guide, evaluate_overlap
from app.storage import load_examples

load_dotenv()
st.set_page_config(page_title="Local Travel Guide", page_icon="🧭", layout="wide")
st.title("🧭 Local Travel Guide Generator")
st.caption("Personalized mini-guides for travel agencies · LoRA fine-tuned model with a transparent baseline")
examples = load_examples()

with st.sidebar:
    st.header("Your trip")
    destination = st.text_input("Destination", "Aurelia Bay")
    interests = st.multiselect("What do you enjoy?", ["food", "history", "nature", "adventure", "art", "relaxation", "family"],
                               default=["food", "history", "nature"])
    days = st.slider("Trip length (days)", 1, 5, 2)
    mode = st.radio("Generation engine", ["Merged LoRA model", "Baseline (always available)"])
    generate = st.button("Generate my mini-guide", type="primary", use_container_width=True)

st.markdown("### How it works")
st.write("Choose a destination and interests. The merged-model option uses the locally fine-tuned GPT-2 checkpoint; the baseline ranks the included fictional-city examples by interest relevance.")
if generate:
    if not destination.strip():
        st.error("Enter a destination first.")
    elif not interests:
        st.error("Choose at least one interest.")
    else:
        if mode == "Merged LoRA model":
            try:
                from app.llm import LocalGuideModel
                with st.spinner("Loading the merged model and writing your guide..."):
                    model = LocalGuideModel()
                    result = model.generate(destination.strip(), interests, days)
                engine = "LoRA-merged GPT-2"
            except Exception as exc:
                st.error(f"Could not run merged model: {exc}")
                st.info("You can still try the baseline engine while you train and merge the adapter.")
                result = baseline_guide(destination.strip(), interests, examples, days)
                engine = "Baseline fallback (not an LLM)"
        else:
            result = baseline_guide(destination.strip(), interests, examples, days)
            engine = "Interest-ranked baseline"
        st.divider()
        st.subheader("Your personalized guide")
        st.markdown(result)
        st.caption(f"Engine: {engine}")
        st.download_button("Download guide", result, file_name="my-local-travel-guide.md", mime="text/markdown")
        report = evaluate_overlap(result, interests)
        st.subheader("Quick relevance check")
        st.metric("Requested-interest coverage", f"{report['coverage']:.0%}")
        st.write("Matched interests:", ", ".join(report["matched_interests"]) or "None detected")
else:
    st.info("Select your interests in the sidebar and click **Generate my mini-guide**.")
st.divider()
st.caption("Aurelia Bay is fictional. Verify real-world details independently before advising customers.")
