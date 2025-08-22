import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from utils import predict

st.set_page_config(page_title="Handwriting Recognition", page_icon="✍️", layout="centered")

st.markdown("<h1 style='text-align:center; color:#6C63FF;'>✍️ Reconeixement de text escrit a mà</h1>", unsafe_allow_html=True)

st.write("Dibuixa un dígit o lletra i el model et dirà què és.")

# Canvas interactiu
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))  # invertir colors
    if st.button("🔮 Predir"):
        char, conf = predict(img)
        st.success(f"El model ha reconegut: **{char}** (confiança: {conf:.2f})")
        st.image(img.resize((100,100)), caption="Entrada normalitzada")
