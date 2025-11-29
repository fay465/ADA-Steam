import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Steam Success Predictor", page_icon="🎮", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('steam_success_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Error: No se encuentran los archivos .pkl. Asegúrate de exportarlos del Colab y ponerlos aquí.")
    st.stop()

st.title("🎮 Steam Success Predictor")
st.markdown("""
Esta herramienta utiliza un modelo de **Machine Learning (Random Forest)** para estimar la probabilidad de que un videojuego tenga una recepción positiva en Steam.
*Basado en análisis de metadatos técnicos y comerciales.*
""")

st.divider()

st.subheader("Define las características del juego:")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Precio del Juego (USD)", min_value=0.0, max_value=200.0, value=19.99, step=1.0)
    is_indie = st.checkbox("¿Es un estudio Indie?", value=True, help="Marca si el Desarrollador es el mismo que el Editor")

with col2:

    available_genres = [c.replace('Gen_', '') for c in model_columns if c.startswith('Gen_')]
    available_cats = [c.replace('Cat_', '') for c in model_columns if c.startswith('Cat_')]
    
    genres_selected = st.multiselect("Géneros", available_genres, default=["Indie", "Action"])
    cats_selected = st.multiselect("Características", available_cats, default=["Single-player"])

if st.button("🔮 Predecir Recepción", type="primary"):
    
    input_data = {col: 0 for col in model_columns}
    
    input_data['price_usd'] = price
    input_data['is_indie'] = 1 if is_indie else 0
    
    for gen in genres_selected:
        col_name = f"Gen_{gen}"
        if col_name in input_data:
            input_data[col_name] = 1
            
    for cat in cats_selected:
        col_name = f"Cat_{cat}"
        if col_name in input_data:
            input_data[col_name] = 1
            
    df_input = pd.DataFrame([input_data])
    
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1] # Probabilidad de ser clase 1
    
    st.divider()
    st.subheader("Resultado del Análisis:")
    
    if prediction == 1:
        st.success(f"🌟 **ÉXITO PROBABLE** (Confianza: {probability:.1%})")
        st.markdown("El modelo predice que este juego tendrá reseñas **Positivas**.")
    else:
        st.error(f"⚠️ **RIESGO ALTO** (Confianza de éxito: {probability:.1%})")
        st.markdown("El modelo predice reseñas **Mixtas o Negativas**. Considera ajustar el precio o añadir funcionalidades.")

    st.progress(float(probability), text="Probabilidad de Aprobación de la Comunidad")

st.markdown("---")
st.caption("Proyecto Final de Análisis de Datos | TRL-3 Proof of Concept")