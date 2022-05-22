import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

color_blue = '#35C4D7'
color_orange = '#FCA311'
color_vert = '#CCE03D'
##############################################
#            Fonctions des onglets           #
##############################################
def intro():
  st.image("images/banniere_fixe.png", use_column_width=True)
  st.image("images/Publicis.png", use_column_width=True)
  st.image("images/Publicis2.png", use_column_width=True)
  st.image("images/Epsilon.png", use_column_width=True)
  st.image("images/Epsilon2.png", use_column_width=True)

def projet():
  st.image("images/Data4good.png", use_column_width=True)
  st.image("images/Data4good2.png", use_column_width=True)
  
  st.markdown(
    """
      # Petite histoire du son
      ### Caractéristiques
     """
  )
  col1, col2 = st.columns((1, 2))
  freq = col1.slider(
      "Fréquence : nombre d’oscillations par seconde (Hz)"
      , min_value=220
      , max_value=2000
      , value=450, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  amp = col1.slider(
      "Amplitude : intensité des oscillations (dB)"
      , min_value = 0
      , max_value = 10
      , value = 1
      , format = "%d"
  )
  fig = demo_freq_amplitude(freq, amp)
  col2.plotly_chart(fig)
  
  st.markdown(
    """
      ### Signal analogique VS numérique
    """
  )
  col1, col2 = st.columns((1, 2))
  sampling = col1.slider(
      "Sampling rate : nombre d’échantillon par seconde"
      , min_value=1000
      , max_value=30000
      , value=20000, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  precision = col1.slider(
      "Sample depth : précision d’un échantillon"
      , min_value = 0
      , max_value = 5
      , value = 1
      , format = "%d"
  )
  fig = demo_sampling_precision(sampling, precision)
  col2.plotly_chart(fig)
  st.markdown(
    """
      # Description des données
      ### Global
     """
  )
  # Statistique générale
  col1, col2 = st.columns((1, 2))
  fig1 = globale_stat()
  col1.plotly_chart(fig1,use_container_width=True)
  fig2 = globale_sunburst()
  col2.plotly_chart(fig2,use_container_width=True)

  # Boxplot max frequency
  fig = globale_boxplot()
  st.plotly_chart(fig,use_container_width=True)

  
def use_case():
  st.image("images/banniere_fixe.png", use_column_width=True)
  st.image("images/UC.png", use_column_width=True)

  
############################################## 
#             Fonctions annexes              #
##############################################
def demo_freq_amplitude(freq, amp,t=0.01):
  S_rate = 44100
  T = 1/S_rate
  N = S_rate * t
  omega = 2 * np.pi * freq
  x = np.arange(N)*T
  y = np.sin(omega * x)*amp
  dataframe = pd.DataFrame({"x": x, "y": y})
  fig = px.line(dataframe, x="x", y="y")
  fig.update(layout_yaxis_range = [-10,10])
  fig.update_layout(
      autosize=True
      , margin=dict(l=100, r=0, t=0, b=0)
      , width=600
      , height = 300
      , xaxis_title="Temps (en seconde)"
      , yaxis_title="Amplitude"
  )
  fig.update_traces(line_color=color_blue)
  return(fig)

def demo_sampling_precision(sampling, bites):
  freq = 500
  amp = 1
  T = 1/sampling
  t = 0.001
  omega = 2 * np.pi * freq
  x = np.arange(0,t,T)
  y = np.round(np.sin(omega * x)*amp,bites)
  dataframe = pd.DataFrame({"x": x, "y": y})
  fig1 = demo_freq_amplitude(freq,amp,t)
  fig2 = px.line(dataframe, x="x", y="y", color_discrete_sequence=[color_orange], markers = True)
  fig3 = go.Figure(data=fig1.data + fig2.data)
  fig3.update_layout(
      autosize=True
      , margin=dict(l=100, r=0, t=0, b=0)
      , width=600
      , height = 300
      , xaxis_title="Temps (en seconde)"
      , yaxis_title="Amplitude"
  )
  return(fig3)

def globale_stat():
  base_sun = pd.read_csv('data/base_sun.csv')
  base_histo = pd.read_csv('data/base_histo.csv')
  trace1 = go.Indicator(
          mode = "number",
          value = base_sun['nb audios'].sum(),
          title = {'text': "Nombre de fichiers audio"},
          domain={'x': [0.0, 1], 'y': [0.60, 1]}
      )
  trace2 = go.Indicator(
          mode = "number",
          value = base_histo['framerate'].mean(),
          title = {'text': "Sampling rate moyen"},
          domain={'x': [0.0, 1], 'y': [0.3, 0.5]}
      )
  trace3 = go.Indicator(
          mode = "number",
          value = base_histo['sampwidth'].mean(),
          title = {'text': "Sample depth moyen par sample"},
          domain={'x': [0.0, 1], 'y': [0.0, 0.2]}
      )
  fig = go.Figure(data = [trace1, trace2, trace3])
  return(fig)  

def globale_sunburst():
  code_couleur = {
      'apprentissage' : color_orange
      , 'validation' : color_blue
  }
  base_sun = pd.read_csv('data/base_sun.csv')
  fig = px.sunburst(
      base_sun
      , path=["base", "hasbird"]
      , values='nb audios'
      , color='base', color_discrete_map = code_couleur
      , title= "Répartition des audios selon leur base et la présence d'oiseau ou non"
  )
  fig.update_traces(textinfo="label+percent entry", textfont=dict(family="Arial Black"))
  return(fig)

def globale_boxplot():
  base_box = pd.read_csv('data/base_boxplot.csv')
  fig = px.box(base_box
      , x='hasbird'
      , y="Frequence max"
      , color="hasbird"
      , color_discrete_map = {0 : color_orange, 1 : color_blue}
      , title = "Distribution de la fréquence maximale selon la présence d'oiseau ou non"
      , labels = {'hasbird' : ""}
  )
  return(fig)
