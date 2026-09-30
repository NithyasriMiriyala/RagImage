"""Split-screen workspace for Image RAG.  Run: streamlit run app.py"""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import ImageRAG, RAGError, preprocess

st.set_page_config(page_title="Image RAG Workspace", page_icon="🖼️", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1400px}</style>", unsafe_allow_html=True)

THRESH = {"Otsu (auto)": "otsu", "Manual": "manual", "Adaptive": "adaptive", "Off": "none"}


@st.cache_data(show_spinner=False)
def live_preview(data, grayscale, denoise, threshold, thresh_value):
    """Cheap OpenCV-only preview, re-run whenever a control changes."""
    return preprocess(data, grayscale, denoise, threshold, thresh_value)


def show(img):
    if img.ndim == 3:
        st.image(img, channels="BGR", use_container_width=True)
    else:
        st.image(img, use_container_width=True, clamp=True)


for key, default in {"rag": None, "src_id": None, "chunks": 0}.items():
    st.session_state.setdefault(key, default)

st.title("🖼️ Image Text Extractor")
st.caption("Upload → tune OpenCV → extract text with OCR")

left, right = st.columns([1, 1.6], gap="large")

# ================= LEFT: input & controls =================
with left:
    st.subheader("1 · Input & controls")
    file = st.file_uploader("Drag and drop an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

    st.markdown("**OpenCV preprocessing**")
    grayscale = st.checkbox("Grayscale conversion", value=True)
    denoise = st.checkbox("Noise reduction", value=True)
    mode = st.radio("Thresholding", list(THRESH), horizontal=True, disabled=not grayscale)
    value = st.slider("Threshold value", 0, 255, 127, disabled=(not grayscale or mode != "Manual"))
    if not grayscale:
        st.caption("Thresholding needs grayscale, so it is skipped.")

    opts = dict(grayscale=grayscale, denoise=denoise,
                threshold=THRESH[mode] if grayscale else "none", thresh_value=value)
    run = st.button("🔍 Extract text", type="primary", use_container_width=True, disabled=file is None)

data = file.getvalue() if file else None
file_id = hashlib.sha256(data).hexdigest() if data else None

# Run OCR + embeddings only when the button is pressed
if run and data:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / file.name
        path.write_bytes(data)
        rag = ImageRAG()
        try:
            with st.spinner("Running OCR..."):
                st.session_state.chunks = rag.ingest(str(path), **opts)
            st.session_state.rag, st.session_state.src_id = rag, file_id
            st.toast("Text extracted ✅")
        except RAGError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

ready = st.session_state.rag is not None and st.session_state.src_id == file_id

# ================= RIGHT: output tabs =================
with right:
    st.subheader("2 · Output")
    tab_img, tab_text = st.tabs(["🖼️ Preprocessed image", "📝 Extracted text"])

    with tab_img:
        if data is None:
            st.info("Upload an image on the left to see the preview.")
        else:
            try:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.image(data, use_container_width=True)
                with c2:
                    st.caption("After OpenCV (live preview)")
                    show(live_preview(data, **opts))
            except RAGError as e:
                st.error(str(e))

    with tab_text:
        if not ready:
            st.info("Click **Extract text** to run OCR on the current settings.")
        else:
            rag = st.session_state.rag
            text = rag.extracted_text
            m1, m2, m3 = st.columns(3)
            m1.metric("Characters", len(text))
            m2.metric("Words", len(text.split()))
            m3.metric("Chunks stored", st.session_state.chunks)
            st.text_area("OCR result", text, height=320, label_visibility="collapsed")
            st.caption("Text is from the last extraction. Click the button again after changing settings.")