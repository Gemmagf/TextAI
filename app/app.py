import streamlit as st
from PIL import Image
import numpy as np
import io

# Intentem importar pillow-heif per HEIC
heic_supported = True
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception as e:
    heic_supported = False
    st.warning(f"HEIC images might not be supported: {e}")

# Pytesseract
try:
    import pytesseract
except Exception as e:
    st.error(f"pytesseract not installed: {e}")

# Configuració pàgina
st.set_page_config(
    page_title="Image to Text OCR",
    page_icon="✍️",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; color:#6C63FF;'>De Fotos a Text ✍️</h1>",
    unsafe_allow_html=True
)

st.write("Puja una imatge (JPG, PNG, HEIC) i el model transcriurà el text.")

# Upload d’imatge
uploaded_file = st.file_uploader("Tria una imatge", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    try:
        # Obrim la imatge
        img = Image.open(uploaded_file).convert("L")  # Convertim a gris
        st.image(img, caption="Imatge pujada", use_column_width=True)

        if st.button("🔮 Transcriure text"):
            text = pytesseract.image_to_string(img, lang="eng")
            st.success("Text reconegut:")
            st.text(text)

    except Exception as e:
        st.error(f"No s'ha pogut obrir la imatge: {e}")
        if uploaded_file.name.lower().endswith(".heic") and not heic_supported:
            st.info("Instal·la pillow-heif per suportar HEIC: pip install pillow-heif")
