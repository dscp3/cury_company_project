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
import numpy as np

st.set_page_config(page_title='Visão Restaurantes', page_icon='🍕', layout='wide')

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

def calculo_dist_media(df, op):
    
    '''
    
    Essa função tem por objetivo calcular distância média de entrega e agrupar por cidade. Se op = 'city', agrupa distância por cidade. Senão, devolve apenas a distância média.

    Input: Dataframe
    Output: Número (op != 'city') ou Figura (op == 'city')
    
    '''

    df_dist = df.copy()
    
    df_dist['distancia'] = df_dist.apply(lambda x: haversine(
            (x['Restaurant_latitude'], 
            x['Restaurant_longitude']), 
            (x['Delivery_location_latitude'],
            x['Delivery_location_longitude'])), axis=1)

    if op != 'city':
        dist_media = np.round(df_dist['distancia'].mean(), 2)
        return dist_media
        
    else:
        dist_media = df_dist.loc[:, ['City', 'distancia']].groupby(['City']).mean().reset_index()
        fig = go.Figure(data=[go.Pie(labels=dist_media['City'], values=dist_media['distancia'], pull=[0, 0.1, 0])])
        return fig

def an_festival(df, festival, op):

    '''
    
    Essa função tem por objetivo analisar dados de média e desvio padrão em relação à realização (ou não) de festival

    Input: Dataframe, festival ('yes' quando teve festival e 'no' quando não teve) e op ('avg' para média e 'std' para Desv. Pad)
    Output: Número
    
    '''
    
    df_fest = df.copy()
    lista_cols = ['Time_taken(min)', 'Festival']
    df_fest = df_fest.loc[:, lista_cols].groupby(['Festival']).agg({'Time_taken(min)': ['mean', 'std']})
    df_fest.columns = ['time_taken_mean', 'time_taken_std']
    df_fest.reset_index(inplace=True)

    if (festival == 'yes') & (op == 'avg'):     
        df_tempo_fest = df_fest.loc[df_fest['Festival'] == 'Yes', :]
        resultado = np.round(df_tempo_fest['time_taken_mean'], 2)
        return resultado

    elif (festival == 'yes') & (op == 'std'):
        df_tempo_fest = df_fest.loc[df_fest['Festival'] == 'Yes', :]
        resultado = np.round(df_tempo_fest['time_taken_std'], 2)
        return resultado

    elif (festival == 'no') & (op == 'avg'):     
        df_tempo_fest = df_fest.loc[df_fest['Festival'] == 'No', :]
        resultado = np.round(df_tempo_fest['time_taken_mean'], 2)
        return resultado

    elif (festival == 'no') & (op == 'std'):
        df_tempo_fest = df_fest.loc[df_fest['Festival'] == 'No', :]
        resultado = np.round(df_tempo_fest['time_taken_std'], 2)
        return resultado

def estudos_cidade(df, op, op2):

    '''
    
    Essa função tem por objetivo analisar dados de média e desvio padrão em relação à tipos de ordem e tráfego por cidade

    Input: Dataframe, op ('1' agrupando por cidade e '2' para estudos mais detalhados) e op2 (funciona apenas quando op for 2: se op2 == 1, então agrupa por tipo de ordem. Senão, agrupa por tipo de tráfego)
    Output: Dataframe (op == 2 e op2 == 1) ou Figura
    
     '''
    
    df_aux = df.copy()
    lista_cols = ['Time_taken(min)', 'City']
    
    if op == 1:      
        df_aux = df_aux.loc[:, lista_cols].groupby(['City']).agg({'Time_taken(min)': ['mean', 'std']})
        df_aux.reset_index(inplace=True)
        df_aux.columns = ['city', 'time_taken_mean', 'time_taken_std']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Controle', x=df_aux['city'],
                                 y=df_aux['time_taken_mean'],
                                 error_y = dict(type='data', array=df_aux['time_taken_std'])))
        fig.update_layout(barmode='group')
        return fig
        
    elif op == 2:
        if op2 == 1:
            lista_cols.append('Type_of_order')
            df_aux = df_aux.loc[:, lista_cols].groupby(['City', 'Type_of_order']).agg({'Time_taken(min)': ['mean', 'std']})
            df_aux.columns = ['time_taken_mean', 'time_taken_std']
            df_aux.reset_index(inplace=True)
            return df_aux

        if op2 == 2:
            lista_cols.append('Road_traffic_density')        
            df_aux = df_aux.loc[:, lista_cols].groupby(['City', 'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})
            df_aux.columns = ['time_taken_mean', 'time_taken_std']
            df_aux.reset_index(inplace=True)
        
            fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='time_taken_mean',
                          color='time_taken_std', color_continuous_scale='RdBu',
                          color_continuous_midpoint=np.average(df_aux['time_taken_std']))        
            return fig     

            
# ========== Leitura ========== #

raw_df = pd.read_csv('Datasets/train.csv')

# ========== Limpeza ========== #

df = limpar_df(raw_df)


# ========== Construção da Página no Streamlit ========== #

st.header('Dashboard Restaurants | Cury Company')

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

orders = st.sidebar.multiselect(
    'Selecione os tipos de pedido:',
    ['Buffet', 'Drinks', 'Meal', 'Snack'], 
    default= ['Buffet', 'Drinks', 'Meal', 'Snack'])

# Aplicação dos filtros do usuário no dataframe

# Filtro de Data

linhas_filtro = df['Order_Date'] <= date_slider
df = df.loc[linhas_filtro, :]

# Filtro de Transito

linhas_filtro_2 = df['Road_traffic_density'].isin(traffic)
df = df.loc[linhas_filtro_2, :]

# Filtro de Pedido

linhas_filtro_3 = df['Type_of_order'].isin(orders)
df = df.loc[linhas_filtro_3, :]

# ==== Visão Entregadores ==== #

with st.container():

    # Título
    st.markdown('## Principais Métricas')

    # Construção das colunas do dash

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Entregadores Únicos
    with col1:
        delivery_unique = df.loc[:, 'Delivery_person_ID'].nunique()
        col1.metric('Entregadores', delivery_unique)

    # Distância Média
    with col2:
        dist_media = calculo_dist_media(df, 'nao_city')
        col2.metric('Dist Média', dist_media)

        
    # Tempo Médio Festival
    with col3:
        col3.metric('Tempo Médio Fest', an_festival(df, 'yes', 'avg'))
        
    # Desvio Padrão Festival
    with col4:
        col4.metric('Desv. Pad Fest', an_festival(df, 'yes', 'std'))

    # Tempo Médio sem Festival
    with col5:
        col5.metric('Tempo Médio sem Fest', an_festival(df, 'no', 'avg'))

    # Desvio Padrão sem Festival
    with col6:
        col6.metric('Desv. Pad sem Fest', an_festival(df, 'no', 'std'))

with st.container():

    col1, col2 = st.columns(2)
    

    with col1:

        st.markdown('### Tempo Médio por Cidade')
        fig = estudos_cidade(df, 1, 1)
        st.plotly_chart(fig)

    with col2:
    
        st.markdown('### Distribuição de Distâncias e Pedidos por Cidade')
        df_aux = estudos_cidade(df, 2, 1)
        st.dataframe(df_aux)

with st.container():
    
    # Título
    st.markdown('## Detalhamento por Cidade')

    # Construção das colunas do dash

    col1, col2 = st.columns(2)

    # Tempo por Cidade
    with col1:

        st.markdown('##### Distância Média Cidade')
        fig = calculo_dist_media(df, 'city')
        st.plotly_chart(fig)
                
        
    # Tempo por Cidade e Tipo de Tráfego
    with col2:
        st.markdown('##### Cidade x Tipo Tráfego')
        fig = df_aux = estudos_cidade(df, 2, 2)
        st.plotly_chart(fig)







