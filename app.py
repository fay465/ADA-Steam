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

st.title("Steam Success Predictor")
st.markdown("""
Esta herramienta predice la recepción de un juego basándose en sus metadatos.
*Modelo: Random Forest (84% Exactitud)*
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Precio (USD)", min_value=0.0, max_value=200.0, value=19.99, step=1.0)
    is_indie = st.checkbox("¿Es Indie?", value=True)

with col2:
    available_genres = [c for c in model_columns if c not in ['price_usd', 'is_indie']]

    if not available_genres:
        st.error("Error: No se detectaron géneros en el modelo. Verifica model_columns.pkl")
        st.stop()

    default_genres = ["Indie", "Action"]
    default_valid = [g for g in default_genres if g in available_genres]

    genres_selected = st.multiselect("Géneros", available_genres, default=default_valid)

if st.button("Predecir", type="primary"):

    input_data = {col: 0 for col in model_columns}

    input_data['price_usd'] = price
    input_data['is_indie'] = 1 if is_indie else 0
)
    for gen in genres_selected:
        if gen in input_data:
            input_data[gen] = 1

    df_input = pd.DataFrame([input_data])
    
    try:
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"**ÉXITO PROBABLE** (Confianza: {probability:.1%})")
        else:
            st.error(f"**ALTO RIESGO** (Confianza de éxito: {probability:.1%})")
            
        st.progress(float(probability))
        
    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {e}")