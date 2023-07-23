from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('modelo_001')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df
    return predictions

def run():

    #CARGANDO IMAGENES PARA EL FROND
    from PIL import Image
    image = Image.open('logo.png')
    image_empresa = Image.open('foto.jpg')

    st.image(image,use_column_width=False)

    #SE AGREGARA A LA IZQUIERDA MEDIANTE UN SLIDER
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('Este es un sitio web para predecir el monto de facturación de energía eléctrica')
    st.sidebar.success('https://www.pycaret.org')
    
    #IMAGEN A LA IZQUI
    st.sidebar.image(image_empresa)

    st.title("Sitio de prediccion")

    #LOGICA 
    if add_selectbox == 'Online':

        depar = st.text_input('Departamento', 'Ica')
        prov = st.text_input('Provincia', 'Pisco')
        distri = st.text_input('Distrito', 'Pisco')
        potencia = st.number_input('Potencia(KW)')
        consumo = st.number_input('Consumo')
        localidad = st.text_input('Localidad', '...')

        output=""

        #LA APP LE PASA LOS DATOS DEL FROND MEDIANTE UN DICCIONARIO
        input_dict = {'DEPARTAMENTO' : depar, 'PROVINCIA' : prov, 'DISTRITO' : distri, 'LOCALIDAD':localidad ,'POTENCIA_CONTRATADA' : potencia, 'CONSUMO_KW' : consumo }
        input_df = pd.DataFrame([input_dict])
        
        #BOTON
        if st.button("Procesar"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
