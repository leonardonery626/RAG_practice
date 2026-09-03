"""Streamlit front end: upload one PDF, ask a question, see the RAG output."""

from __future__ import annotations

import logging
from types import ModuleType

import streamlit as st

from src.local_ui import pipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄")


@st.cache_resource(show_spinner="Loading the AI model (first run can take a few minutes)…")
def _load_generator_module() -> ModuleType:
    from src.func import generator_builder

    return generator_builder


st.title("Ask questions about a PDF")
st.write(
    "Upload a single PDF, then ask a question about its contents. "
    "The app will show the passages it used and the generated answer."
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    current_upload = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("ingested_upload") != current_upload:
        with st.spinner("Reading and indexing your PDF…"):
            try:
                summary = pipeline.ingest_pdf(
                    uploaded_file.getvalue(), uploaded_file.name
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not re-raised
                st.session_state.pop("ingested_upload", None)
                st.error(f"Could not process this PDF: {exc}")
            else:
                st.session_state["ingested_upload"] = current_upload
                st.success(f"Indexed {summary['num_chunks']} chunks from your PDF.")

is_ready = "ingested_upload" in st.session_state

prompt = st.text_input("Your question", disabled=not is_ready)
ask_clicked = st.button("Ask", disabled=not is_ready)

if ask_clicked:
    if not prompt.strip():
        st.warning("Please enter a question.")
    else:
        _load_generator_module()
        with st.spinner("Thinking…"):
            try:
                result = pipeline.answer_query(prompt)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not re-raised
                st.error(f"Could not answer this question: {exc}")
            else:
                st.subheader("Retrieved passages")
                for item in result["retrieved"]:
                    chunk = item["chunk"]
                    page = chunk.get("metadata", {}).get("page", "?")
                    label = f"Passage {item['rank']} · page {page} · score {item['score']:.2f}"
                    with st.expander(label):
                        st.write(chunk["text"])

                st.subheader("Answer")
                st.write(result["answer"])
