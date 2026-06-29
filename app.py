import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Deteksi Masker Wajah",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

header {
    background: transparent;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #1E293B;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #CBD5E1;
    border-radius: 18px;
    padding: 22px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2563EB;
    background: #F8FAFC;
}

.result-box {
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,.06);
}

.footer {
    text-align: center;
    color: #94A3B8;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


THRESHOLD_LOW = 0.35
THRESHOLD_HIGH = 0.65


@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=output
    )

    model.load_weights(r"D:\uas_cvs\best_weights_final.weights.h5")

    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model : {e}")
    st.stop()


st.title("🛡️ Deteksi Masker Wajah")
st.caption("Deteksi penggunaan masker wajah menggunakan model MobileNetV2.")
st.divider()


uploaded_file = st.file_uploader(
    "📤 Upload gambar wajah",
    type=["jpg", "jpeg", "png"]
)

st.markdown("""
<div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:16px 20px;">
    <b>💡 Tips</b><br><br>
    • Gunakan foto yang terang<br>
    • Wajah menghadap kamera<br>
    • Hindari blur<br>
    • Pastikan hanya ada satu wajah pada gambar
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    img = image.resize((224, 224))
    img = np.array(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, 0)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("🖼️ Preview")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Hasil Prediksi")

        with st.spinner("Sedang menganalisis gambar..."):
            prediction = model.predict(img)

        score = float(prediction[0][0])

        if score < THRESHOLD_LOW:
            conf = (1 - score) * 100

            with st.container(border=True):
                st.success("✅ Memakai Masker")
                st.metric("Confidence", f"{conf:.1f}%")
                st.progress(conf / 100)
                st.write("Model mendeteksi bahwa orang pada gambar **menggunakan masker**.")
                st.caption("Prediksi memiliki tingkat keyakinan yang tinggi.")

        elif score > THRESHOLD_HIGH:
            conf = score * 100

            with st.container(border=True):
                st.error("❌ Tidak Memakai Masker")
                st.metric("Confidence", f"{conf:.1f}%")
                st.progress(conf / 100)
                st.write("Model mendeteksi bahwa orang pada gambar **tidak menggunakan masker**.")
                st.caption("Prediksi memiliki tingkat keyakinan yang tinggi.")

        else:
            conf = abs(score - 0.5) * 200

            with st.container(border=True):
                st.warning("🤔 Model Tidak Yakin")
                st.metric("Confidence", f"{conf:.1f}%")
                st.progress(conf / 100)
                st.write("Coba gunakan foto yang lebih jelas, pencahayaan lebih terang, dan wajah menghadap kamera.")
                st.caption("Confidence masih rendah sehingga model belum yakin.")


st.divider()

st.markdown(
    """
    <div class="footer">
        <b>🛡️ Deteksi Masker Wajah</b><br>
        Built with ❤️ TensorFlow • MobileNetV2 • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)