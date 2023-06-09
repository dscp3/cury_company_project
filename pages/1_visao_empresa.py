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

st.set_page_config(page_title='Visão Empresa', page_icon='📊', layout='wide')

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

def ordens_dia(df):
    
    '''

    Construção do gráfico de barra para mostrar a visão de entregas por dia

    Input: Dataframe
    Output: Gráfico

    '''
    
    cols = ['ID', 'Order_Date']
    df_plot_dia = df.loc[:, cols].groupby('Order_Date').count().reset_index()
    df_plot_dia.columns = ['order_date', 'qtd_entregas']
    fig1 = px.bar(df_plot_dia, x='order_date', y='qtd_entregas')
    
    return fig1

def trafego_entregas(df):

    '''

    Construção do gráfico de barra horizontal para mostrar a visão de entregas por tipo de tráfego

    Input: Dataframe
    Output: Gráfico

    '''
    
    cols = ['ID', 'Road_traffic_density']
    df_plot = df.loc[:, cols].groupby('Road_traffic_density').count().reset_index().sort_values(by='ID')
    df_plot = df_plot.loc[df_plot['Road_traffic_density'] != "NaN", :]
    df_plot.columns=['road_traffic', 'qtd_entregas']
    fig2 = px.bar(df_plot, x='qtd_entregas', y='road_traffic', orientation='h')
    
    return fig2

def trafego_entregas_city(df):

    '''

    Construção do gráfico de bolhas para mostrar a visão de cidade  por tipo de tráfego

    Input: Dataframe
    Output: Gráfico

    '''
    
    cols = ['ID', 'City', 'Road_traffic_density']
    df_plot = df.loc[:, cols].groupby(['City', 'Road_traffic_density']).count().reset_index()
    fig3 = px.scatter(df_plot, x='City', y='Road_traffic_density', size='ID', color='City')
    
    return fig3

def ordens_semana(df):

    '''

    Construção do gráfico de linha para mostrar a visão de pedidos por semana

    Input: Dataframe
    Output: Gráfico

    '''
    
    cols = ['ID', 'Order_Date']
    df_plot_dia = df.loc[:, cols].groupby('Order_Date').count().reset_index()
    df_plot_dia.columns = ['order_date', 'qtd_entregas']
        
    df_plot_dia['week_num'] = df_plot_dia['order_date'].dt.strftime('%U')
    cols = ['qtd_entregas', 'week_num']
        
    df_plot_semana = df_plot_dia.loc[:, cols].groupby('week_num').count().reset_index()
    fig4 = px.line(df_plot_semana, x='week_num', y='qtd_entregas')
            
    return fig4

def pedidos_entregador_semana(df):

    '''

    Construção do gráfico de linha para mostrar a visão de pedidos por semana por entregador

    Input: Dataframe
    Output: Gráfico

    '''
    
    df_plot_entreg_semana = df.copy()
    df_plot_entreg_semana['week_num'] = df['Order_Date'].dt.strftime('%U')
    cols = ['ID', 'week_num', 'Delivery_person_ID']
        
    df_plot_entreg_semana = df_plot_entreg_semana.loc[:, cols].groupby(['week_num']).agg({'ID': ['count'], 'Delivery_person_ID': ['nunique']}).reset_index()
    df_plot_entreg_semana.columns = ["_".join(col).strip() for col in df_plot_entreg_semana.columns.values]
    df_plot_entreg_semana['order_by_delivery'] = df_plot_entreg_semana['ID_count'] / df_plot_entreg_semana['Delivery_person_ID_nunique']
    fig5 = px.line(df_plot_entreg_semana, x='week_num_', y='order_by_delivery')

    return fig5
    
def desenhar_mapa(df):
        
    '''

    Construção do mapa com a relação das cidades

    Input: Dataframe
    Output: Mapa

    '''
    
    cols = ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']
    df_plot_3 = df.copy()
    df_plot_3 = df_plot_3.loc[:, cols].groupby(['City', 'Road_traffic_density']).median().reset_index()
    
    map_x = folium.Map(zoom_start=33)
    
    for i, location_info in df_plot_3.iterrows():
      folium.Marker([location_info['Delivery_location_latitude'],
                     location_info['Delivery_location_longitude']],
                    popup=location_info[['City', 'Road_traffic_density']]).add_to(map_x)
    
    return map_x
    
# ========== Leitura ========== #

raw_df = pd.read_csv('Datasets/train.csv')

# ========== Limpeza ========== #

df = limpar_df(raw_df)


# ========== Construção da Página no Streamlit ========== #

st.header('Dashboard | Cury Company')


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

# Aplicação dos filtros do usuário no dataframe

# Filtro de Data

linhas_filtro = df['Order_Date'] <= date_slider
df = df.loc[linhas_filtro, :]

# Filtro de Transito

linhas_filtro_2 = df['Road_traffic_density'].isin(traffic)
df = df.loc[linhas_filtro_2, :]


# ==== Abas ==== #

tab1, tab2, tab3 = st.tabs([
    'Visão Estratégica',
    'Visão Operacional',
    'Visão Geográfica'])


# ==== Visão Estratégica ==== #

with tab1:
    with st.container():
        
        # Título
        st.markdown('## Ordens por Dia')
        
        # Seleção de Linhas e Colunas
        fig1 = ordens_dia(df)
    
        # Construção do gráfico
        st.plotly_chart(fig1, use_container_width=True)
        

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            
            # Título
            st.markdown('## Distribuição por tipo de tráfego')

            # Seleção de Linhas e Colunas
            fig2 = trafego_entregas(df)
            
            # Construção do gráfico
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            
            # Título
            st.markdown('## Distribuição por tipo de tráfego e cidade')

            # Seleção de Linhas e Colunas
            fig3 = trafego_entregas_city(df)
            
            # Construção do gráfico
            st.plotly_chart(fig3, use_container_width=True)
            
        
# ==== Visão Operacional ==== #

with tab2:
    with st.container():
        
        # Título
        st.markdown('## Pedidos por semana')
    
        # Seleção de Linhas e Colunas
        fig4 = ordens_semana(df)

        # Construção do Gráfico   
        st.plotly_chart(fig4, use_container_width=True)

    with st.container():

        # Títuto
        st.markdown('## Pedidos por entregador por semana')

        # Seleção de Linhas e Colunas
        fig5 = pedidos_entregador_semana(df)

        # Construção do Gráfico
        st.plotly_chart(fig5, use_container_width=True)        


# ==== Visão Geográfica ==== #

with tab3:
    
    # Título
    st.markdown('## Principais Cidades')

    # Seleção de Linhas e Colunas
    map_x = desenhar_mapa(df)

    # Construção do Mapa
    folium_static(map_x, width=1024, height=600)

    
    

















