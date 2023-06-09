import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Home',
    page_icon='🛵')

# image_path = "C:/Users/Acer/Documents/Comunidade_DS/ftc_python"
image = Image.open('bicicleta-de-entrega.png')

st.sidebar.image(image, width=120)
st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('''---''')

st.write('# Cury Company General Dashboard')

st.markdown(
    '''
    
    Dashboard para acompanhamento das métricas de crescimento e performance da empresa

    ### Como utilizar?

    - VISÃO EMPRESA
        - Visão Estratégica: Métricas gerais
        - Visão Operacional: Acompanhamento semanal
        - Visão Geográfica: Insights de geolocalização

    - VISÃO ENTREGADOR
        - Indicadores de crescimento

    - VISÃO RESTAURANTE
        - Indicadores de acompanhamento dos restaurantes
    
    ''')
