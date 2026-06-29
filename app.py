import streamlit as st
import numpy as np
from PIL import Image
import random

# Coba import tensorflow
try:
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

st.set_page_config(page_title="Deteksi Masker Wajah", page_icon="🛡️", layout="wide")

# --- UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F4F7FB; }
.footer { text-align: center; color: #94A3B8; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_AVAILABLE: return None
    
    # Arsitektur MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    model.load_weights("best_weights_final.weights.h5")
    return model

st.title("🛡️ Deteksi Masker Wajah")
st.caption("Deteksi penggunaan masker menggunakan model MobileNetV2.")
st.divider()

uploaded_file = st.file_uploader("📤 Upload gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.subheader("🖼️ Preview")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Hasil Prediksi")
        if st.button("Deteksi Sekarang"):
            with st.spinner("Menganalisis..."):
                if MODEL_AVAILABLE:
                    try:
                        # FIX: Pastikan gambar selalu jadi RGB (3 channel)
                        image = image.convert("RGB")
                        img = image.resize((224, 224))
                        img = np.array(img).astype("float32") / 255.0
                        img = np.expand_dims(img, 0)
                        
                        model = load_model()
                        score = float(model.predict(img)[0][0])
                        
                        if score < 0.35:
                            st.success(f"✅ Memakai Masker (Conf: {(1-score)*100:.1f}%)")
                        else:
                            st.error(f"❌ Tidak Memakai Masker (Conf: {score*100:.1f}%)")
                    except Exception as e:
                        st.error(f"Error prediksi: {e}")
                else:
                    st.warning("Mode simulasi (Cloud tidak mendukung model AI saat ini).")
                    if random.choice([True, False]):
                        st.success("Hasil: Memakai Masker ✅")
                    else:
                        st.error("Hasil: Tidak Memakai Masker ❌")

st.markdown('<div class="footer">Built with ❤️ TensorFlow • MobileNetV2 • Streamlit</div>', unsafe_allow_html=True)