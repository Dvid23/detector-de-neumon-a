import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageOps
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="PneumoAI — Detector de Neumonía",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "dima806/chest_xray_pneumonia_detection"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp { background: linear-gradient(135deg, #020818 0%, #040d24 50%, #020818 100%); }

.main-header {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00d4ff, #0099ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 6px;
    padding: 20px 0 6px;
    text-transform: uppercase;
}

.sub-header {
    text-align: center;
    color: #4a9eff;
    font-size: 0.95rem;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.7;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7b2fff, transparent);
    margin: 10px 0 24px;
}

.prediction-box {
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 1rem 0;
    letter-spacing: 3px;
}

.normal-box {
    background: linear-gradient(135deg, #003d1a, #005c28);
    color: #00ff88;
    border: 1px solid #00ff88;
    box-shadow: 0 0 30px rgba(0,255,136,0.2);
}

.pneumonia-box {
    background: linear-gradient(135deg, #3d0000, #5c0000);
    color: #ff4444;
    border: 1px solid #ff4444;
    box-shadow: 0 0 30px rgba(255,68,68,0.2);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040d24, #020818);
    border-right: 1px solid #0d2847;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0066ff, #7b2fff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    padding: 14px 24px;
    width: 100%;
    box-shadow: 0 0 20px rgba(0,102,255,0.3);
    animation: pulso 2s ease-in-out infinite;
}

@keyframes pulso {
    0%   { box-shadow: 0 0 10px rgba(0,102,255,0.3); }
    50%  { box-shadow: 0 0 40px rgba(123,47,255,0.8), 0 0 60px rgba(0,102,255,0.4); }
    100% { box-shadow: 0 0 10px rgba(0,102,255,0.3); }
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace;
    color: #00d4ff !important;
    font-size: 2.2rem !important;
}

.footer {
    text-align: center;
    color: #1a4a7a;
    font-size: 0.75rem;
    letter-spacing: 2px;
    padding: 20px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with st.spinner("Inicializando Vision Transformer..."):
        try:
            processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            model.eval()
            return processor, model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.stop()


def predict_image(processor, model, pil_img):
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    idx = int(probs.argmax())
    label = model.config.id2label[idx]
    confidence = float(probs[idx])
    all_labels = [model.config.id2label[i] for i in range(len(probs))]
    prob_dict = {all_labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    return label, confidence, prob_dict


if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<h1 class="main-header">🫁 Sistema de Detección de Neumonía</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

processor, model = load_model()
st.markdown(
    '<p class="footer">UNIVERSIDAD PRIVADA ANTENOR ORREGO · INTELIGENCIA ARTIFICIAL — PRINCIPIOS Y TÉCNICAS · UPAO 2026</p>',
    unsafe_allow_html=True
)# Sidebar
with st.sidebar:
    st.markdown("### 👤 Datos del Paciente")
    patient_name = st.text_input("Nombre completo", placeholder="Ej: Juan Pérez")
    document_id  = st.text_input("DNI / Documento", placeholder="Ej: 12345678")
    age          = st.number_input("Edad", min_value=0, max_value=120, value=None, step=1, placeholder="Edad")
    notes        = st.text_area("Notas clínicas", placeholder="Observaciones del médico...")
    st.markdown("---")
    st.markdown("### 🤖 Modelo IA")
    st.info("""
**Arquitectura:** Vision Transformer (ViT)
**Fuente:** Hugging Face — dima806
**Base:** google/vit-base-patch16-224-in21k
**Accuracy:** 96.08%
**F1-Score:** 0.9608
**Dataset:** Kermany et al. 2018
**Licencia:** Apache 2.0
    """)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📡 Cargar Radiografía")
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen de tórax",
        type=["jpg", "jpeg", "png"],
        help="JPG, JPEG, PNG — máximo 10 MB"
    )

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ Imagen mayor a 10 MB.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Radiografía cargada", use_container_width=True)

            if st.button("⚡ ANALIZAR CON IA", type="primary"):
                with st.spinner("Procesando con Vision Transformer..."):
                    label, confidence, prob_dict = predict_image(processor, model, image)

                label_display = "NORMAL" if "NORMAL" in label.upper() else "NEUMONÍA"

                st.session_state.history.insert(0, {
                    "Fecha/Hora":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Paciente":    patient_name or "Sin nombre",
                    "DNI":         document_id or "N/A",
                    "Edad":        int(age) if age else "N/A",
                    "Diagnóstico": label_display,
                    "Confianza":   f"{confidence*100:.2f}%",
                    "Notas":       notes or "—",
                })

                with col2:
                    st.markdown("### 📊 Resultado del Análisis")
                    box_class = "normal-box" if "NORMAL" in label.upper() else "pneumonia-box"
                    emoji = "✅" if "NORMAL" in label.upper() else "⚠️"
                    st.markdown(
                        f'<div class="prediction-box {box_class}">{emoji} {label_display}</div>',
                        unsafe_allow_html=True
                    )
                    st.metric("Nivel de Confianza", f"{confidence*100:.2f}%")
                    st.markdown("#### Distribución de Probabilidades")
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(prob_dict.keys()),
                            y=list(prob_dict.values()),
                            marker=dict(
                                color=["#00ff88" if "NORMAL" in k.upper() else "#ff4444" 
                                    for k in prob_dict.keys()],
                                line=dict(color=["#00ff88" if "NORMAL" in k.upper() else "#ff4444" 
                                                for k in prob_dict.keys()], width=1)
                            ),
                            text=[f"{v:.1f}%" for v in prob_dict.values()],
                            textposition="outside",
                            textfont=dict(size=18, color="white", family="Orbitron"),
                        )
                    ])
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white", size=14, family="Rajdhani"),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        xaxis=dict(tickfont=dict(size=16, color="#4a9eff")),
                        margin=dict(t=40, b=20),
                        height=280,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if patient_name or document_id:
                        st.markdown("#### 👤 Paciente")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.text(f"Nombre: {patient_name or 'N/A'}")
                            st.text(f"Edad: {int(age) if age else 'N/A'}")
                        with c2:
                            st.text(f"DNI: {document_id or 'N/A'}")
                        if notes:
                            st.text_area("Notas:", notes, disabled=True,
                                        key=f"notes_{len(st.session_state.history)}")

                    if "NORMAL" in label.upper():
                        st.success("✅ No se detectaron signos de neumonía.")
                    else:
                        st.error("⚠️ Posible neumonía detectada. Consulte con un especialista.")

                    if confidence < 0.75:
                        st.warning(f"⚠️ Confianza baja ({confidence*100:.1f}%). Revisión manual recomendada.")

with col2:
    if uploaded_file is None:
        st.markdown("### 📊 Resultado del Análisis")
        st.info("👈 Carga una radiografía para comenzar el análisis")

# Historial
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📋 Historial de Análisis")

if st.session_state.history:
    c_btn, c_fil = st.columns([1, 3])
    with c_btn:
        if st.button("🗑️ Limpiar"):
            st.session_state.history = []
            st.rerun()
    with c_fil:
        filtro = st.selectbox("Filtrar", ["Todos", "NORMAL", "NEUMONÍA"],
                              label_visibility="collapsed")
    df = pd.DataFrame(st.session_state.history)
    if filtro != "Todos":
        df = df[df["Diagnóstico"] == filtro]
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No hay análisis previos.")

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="footer">PNEUMOAI · VISION TRANSFORMER (ViT) · dima806/chest_xray_pneumonia_detection · UPAO 2026</p>',
    unsafe_allow_html=True
)
