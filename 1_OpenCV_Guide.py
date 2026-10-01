"""Guide page: what each control means. Appears automatically in the Streamlit sidebar."""
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="OpenCV Guide", page_icon="📘", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1200px}</style>", unsafe_allow_html=True)

st.title("📘 What do these settings mean?")
st.write(
    "OCR (text reading) works best on a **clean, high-contrast, black-text-on-white** image. "
    "These settings clean up your picture *before* the OCR reads it."
)


# ---------- Live demo helpers ----------
@st.cache_data
def sample_png() -> bytes:
    """A synthetic photo: text, uneven lighting (dark on the right) and grainy noise."""
    h, w = 420, 1500
    light = np.tile(np.linspace(235, 110, w, dtype=np.float32), (h, 1))
    img = np.dstack([light] * 3).astype(np.uint8)
    cv2.putText(img, "Hello OCR 123", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (40, 40, 40), 6, cv2.LINE_AA)
    cv2.putText(img, "Image to text with OpenCV", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 40), 4, cv2.LINE_AA)
    noise = np.random.default_rng(0).normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def process(data, grayscale=True, denoise=True, threshold="none", value=127):
    """Same logic as the real pipeline (imported lazily so this page loads fast)."""
    from rag_pipeline import preprocess
    return preprocess(data, grayscale, denoise, threshold, value)


def show(img, caption):
    if img.ndim == 3:
        st.image(img, caption=caption, channels="BGR", use_container_width=True)
    else:
        st.image(img, caption=caption, use_container_width=True, clamp=True)


# ---------- Definitions ----------
st.header("The settings, one by one")

with st.container(border=True):
    st.subheader("⚫ Grayscale conversion")
    st.markdown(
        """
**What it is:** turns a colour picture into shades of gray, from 0 (black) to 255 (white).

**How it works:** a colour pixel stores 3 numbers (red, green, blue). Grayscale mixes them into 1 number
(roughly `0.30·R + 0.59·G + 0.11·B`), so each pixel becomes just "how bright is it?".

**Why we do it:** letters are recognised by their *shape*, not their colour. One number per pixel is simpler,
faster, and is required before thresholding.
        """
    )

with st.container(border=True):
    st.subheader("✨ Noise reduction")
    st.markdown(
        """
**What it is:** removes tiny random dots and grain (from phone cameras, low light, scanning or compression).

**How it works:** each pixel is replaced by a value based on its neighbours. A *median blur* takes the middle
value of the 3×3 pixels around it, which wipes out lone specks. A light *Gaussian blur* then smooths what is left.

**Watch out:** too much blur makes thin letters fuzzy. We use a very small amount.
        """
    )

with st.container(border=True):
    st.subheader("◼️◻️ Thresholding")
    st.markdown(
        """
**What it is:** turns the gray image into pure **black and white**, nothing in between.

**How it works:** pick a cut-off number `T`. Every pixel **brighter than T becomes white (255)**, and every pixel
**darker than T becomes black (0)**. Text ends up solid black on a clean white page, which is what OCR likes best.

The four options below are four ways of choosing `T`.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**🤖 Otsu (auto)**
The computer picks the single best `T` for the whole image. It looks at how many pixels are at each brightness
and chooses the `T` that best separates "dark text" from "light background".
✅ Best for clean scans and screenshots with even lighting.
⚠️ Struggles with shadows.
            """
        )
        st.markdown(
            """
**🌗 Adaptive**
Chooses a different `T` for every small area (a 31×31 pixel neighbourhood), based on the pixels around it.
✅ Best for photos of paper with shadows or uneven light.
⚠️ Can create specks on blank areas.
            """
        )
    with c2:
        st.markdown(
            """
**🎚️ Manual**
You choose `T` yourself with the *Threshold value* slider.
- **Lower T** → fewer pixels stay black, so thin or faint text can disappear.
- **Higher T** → more pixels turn black, so text gets thick and the background can fill with black.

✅ Useful when Otsu gives a bad result.
            """
        )
        st.markdown(
            """
**🚫 Off**
No black-and-white step. The gray image goes straight to OCR, which does its own internal clean-up.
✅ Good for images that are already clean, or when thresholding damages the letters.
            """
        )

with st.container(border=True):
    st.subheader("🔢 Threshold value (127)")
    st.markdown(
        """
**What it is:** the cut-off number `T` for **Manual** thresholding, on a scale of 0 to 255.

**Why 127:** it is the midpoint, so anything brighter than middle-gray becomes white and anything darker becomes black.

**Note:** the slider only works in **Manual** mode. In Otsu and Adaptive the computer picks the number, so the slider is greyed out.
        """
    )

# ---------- Which one should I use? ----------
st.header("Which settings should I use?")
st.table(
    {
        "Your image": [
            "Clean screenshot or scanned page",
            "Photo of paper with shadows",
            "Grainy or low-light photo",
            "Faint or light-gray text",
            "OCR result looks wrong",
        ],
        "Try this": [
            "Grayscale ✔  Noise ✘  Otsu",
            "Grayscale ✔  Noise ✔  Adaptive",
            "Grayscale ✔  Noise ✔  Otsu",
            "Grayscale ✔  Manual with a higher value (160–200)",
            "Switch thresholding to Off, or change the mode and extract again",
        ],
    }
)

# ---------- Live demo ----------
st.header("🧪 See it in action")
st.caption("A sample image with grainy noise and a shadow on the right. Watch how each setting changes it.")

data = sample_png()
a, b = st.columns(2)
with a:
    show(process(data, grayscale=False, denoise=False), "Original (colour, noisy, shadow on the right)")
with b:
    show(process(data, True, False, "none"), "Grayscale only")

a, b = st.columns(2)
with a:
    show(process(data, True, True, "none"), "Grayscale + noise reduction (grain is smoother)")
with b:
    show(process(data, True, True, "otsu"), "+ Otsu (the shadow side starts to suffer)")

a, b = st.columns(2)
with a:
    t = st.slider("Try Manual threshold value", 0, 255, 127)
    show(process(data, True, True, "manual", t), f"+ Manual, T = {t}")
with b:
    st.write("")
    st.write("")
    show(process(data, True, True, "adaptive"), "+ Adaptive (handles the shadow best)")

st.info("Go back to the **app** page in the sidebar and try these settings on your own image.")
