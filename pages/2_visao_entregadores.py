# ========== Bibliotecas ========== #

from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Entregadores', page_icon='🛴', layout='wide')

# ========== Funções ========== #

def limpar_df(df):
    
    '''
    
    Essa função tem por objetivo limpar o dataframe em questão. São feitas as seguintes etapas:

    1. Remoção de NaN
    2. Mudança de datatype de algumas colunas
    3. Remoção de espaços nas variáveis de texto
    4. Formatação da coluna de data
    5. Limpeza da coluna de tempo (remover texto)

    Input: Dataframe
    Output: Dataframe
    
    '''
    
    df = df.loc[df['Delivery_person_Age'] != 'NaN ', :].copy()
    df = df.loc[df['Weatherconditions'] != 'conditions NaN', :].copy()
    df = df.loc[df['City'] != 'NaN ', :].copy()
    df = df.loc[df['Festival'] != 'NaN ', :].copy()
    df = df.loc[df['Road_traffic_density'] != 'NaN', :].copy()
    df = df.loc[df['multiple_deliveries'] != 'NaN ', :].copy()
    
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)
    
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()
    
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)

    return df

def avaliacao_media(df, col, num_op):     
    
    '''
    
    Essa função calcula a avaliação média agregada por 'col'. Caso num_op seja 1, então calcula apenas a média. Senão, calcula média e desvio padrão
    
    Input: Dataframe, coluna de agregação, num_op
    Output: Dataframe
    
    '''
    
    cols = ['Delivery_person_Ratings']
    cols.append(col)
            
    if num_op == 1:
        df_avg = df.loc[:, cols].groupby(cols[1]).mean().reset_index()
        return df_avg
        
    else:
        df_avg = df.loc[:, cols].groupby(cols[1]).agg({'Delivery_person_Ratings': ['mean', 'std']})
        df_avg.columns = ['delivery_mean', 'delivery_std']
        df_avg.reset_index(inplace=True)
        return df_avg


def top_entregadores(df, op):
    
    '''
    
    Essa função calcula o top 10 entregadores em 'df'. Caso op seja 1, calcula-se os mais rápidos. Senão, os mais lentos
    
    Input: Dataframe, op
    Output: Dataframe
    
    '''
    
    cols = ['Delivery_person_ID', 'City', 'Time_taken(min)']

    if op == 1:
        df_rapidos = df.loc[:, cols].groupby(['City', 'Delivery_person_ID']).min() \
                    .sort_values(['City', 'Time_taken(min)']).reset_index()
    
        df5_aux = df_rapidos.loc[df_rapidos['City'] == 'Metropolitian', :].head(10)
        df5_aux2 = df_rapidos.loc[df_rapidos['City'] == 'Urban', :].head(10)
        df5_aux3 = df_rapidos.loc[df_rapidos['City'] == 'Semi-Urban', :].head(10)
        
        df_rapidos = pd.concat([df5_aux, df5_aux2, df5_aux3]).reset_index(drop=True)
        return df_rapidos

    else:   
        df_lentos = df.loc[:, cols].groupby(['City', 'Delivery_person_ID']).max() \
                    .sort_values(['City', 'Time_taken(min)']).reset_index()
        
        df5_aux = df_lentos.loc[df_lentos['City'] == 'Metropolitian', :].head(10)
        df5_aux2 = df_lentos.loc[df_lentos['City'] == 'Urban', :].head(10)
        df5_aux3 = df_lentos.loc[df_lentos['City'] == 'Semi-Urban', :].head(10)
        
        df_lentos = pd.concat([df5_aux, df5_aux2, df5_aux3]).reset_index(drop=True)
        return df_lentos

        
# ========== Leitura ========== #

raw_df = pd.read_csv('Datasets/train.csv')

# ========== Limpeza ========== #

df = limpar_df(raw_df)


# ========== Construção da Página no Streamlit ========== #

st.header('Dashboard Delivery Person | Cury Company')


# ==== Sidebar ==== #


image_path = 'bicicleta-de-entrega.png'
image = Image.open(image_path)
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Cury Company')

st.sidebar.markdown('''---''')

st.sidebar.markdown('### Selecione uma data')
date_slider = st.sidebar.slider(
    'Qual valor limite?',
    value = datetime.datetime(2022, 4, 13),
    min_value = datetime.datetime(2022, 2, 11),
    max_value = datetime.datetime(2022, 4, 6),
    format='DD-MM-YYYY')

st.sidebar.markdown('''---''')

traffic = st.sidebar.multiselect(
    'Selecione as opções de condições do trânsito:',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low', 'Medium', 'High', 'Jam'])

st.sidebar.markdown('''---''')

weather = st.sidebar.multiselect(
    'Selecione as opções de condições do trânsito:',
    ['conditions Sunny', 'conditions Stormy', 'conditions Sandstorms', 'conditions Cloudy', 
    'conditions Fog', 'conditions Windy'],
    default= ['conditions Sunny', 'conditions Stormy', 'conditions Sandstorms', 'conditions Cloudy', 
    'conditions Fog', 'conditions Windy'])

# Aplicação dos filtros do usuário no dataframe

# Filtro de Data

linhas_filtro = df['Order_Date'] <= date_slider
df = df.loc[linhas_filtro, :]

# Filtro de Transito

linhas_filtro_2 = df['Road_traffic_density'].isin(traffic)
df = df.loc[linhas_filtro_2, :]

# Filtro de Clima

linhas_filtro_3 = df['Weatherconditions'].isin(weather)
df = df.loc[linhas_filtro_3, :]


# ==== Visão Entregadores ==== #

with st.container():

    # Título
    st.markdown('## Principais Métricas')

    # Construção das colunas do dash

    col1, col2, col3, col4 = st.columns(4, gap='large')

    # Maior Idade dos Entregadores
    with col1:
        max=df.loc[:, 'Delivery_person_Age'].max()
        col1.metric('Maior Idade', max)

    # Menor Idade dos Entregadores
    with col2:
        min=df.loc[:, 'Delivery_person_Age'].min()
        col2.metric('Menor Idade', min)
        
    # Melhor Condição Veículo
    with col3:
        melhor=df.loc[:, 'Vehicle_condition'].max()
        col3.metric('Melhor Condição', melhor)
        
    # Pior Condição Veículo
    with col4:
        pior=df.loc[:, 'Vehicle_condition'].min()
        col4.metric('Pior Condição', pior)

with st.container():

    # Título
    st.markdown('---')
    st.markdown('### Avaliações')

    # Construção das colunas do dash

    col1, col2 = st.columns(2)

    # Avaliação Média por Entregador
    with col1:
        st.markdown('#### Avaliações Médias por Entregador')
        df_avg = avaliacao_media(df, 'Delivery_person_ID', 1)
        st.dataframe(df_avg)

    # Avaliação Média por tipo de trânsito e por clima
    with col2:
        st.markdown('#### Avaliações Médias por Tipo de Trânsito')
        df_avg = avaliacao_media(df, 'Road_traffic_density', 2)
        st.dataframe(df_avg)

        st.markdown('#### Avaliações Médias por Clima')
        df_avg = avaliacao_media(df, 'Weatherconditions', 2)
        st.dataframe(df_avg)

with st.container():
    
    # Título
    st.markdown('---')
    st.markdown('### Velocidade de Entrega')

    # Construção das colunas do dash

    col1, col2 = st.columns(2)

    # Top 10 Entregadores mais rápidos
    with col1:
        st.markdown('#### Entregadores mais rápidos')
        df_rapidos = top_entregadores(df, 1)
        st.dataframe(data=df_rapidos)

    # Top 10 Entregadores mais lentos
    with col2:
        st.markdown('#### Entregadores mais lentos')
        df_lentos = top_entregadores(df, 2)
        st.dataframe(data=df_lentos)
        







