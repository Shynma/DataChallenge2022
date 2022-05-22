import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

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
      ## Caractéristiques
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
      ## Signal analogique VS numérique
    """
  )
  col1, col2 = st.columns((1, 3))
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
#         , width=600
#         , height = 300
        , xaxis_title="Temps (en seconde)"
        , yaxis_title="Amplitude"
    )
    fig.update_traces(line_color='#35C4D7')
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
    fig2 = px.line(dataframe, x="x", y="y", color_discrete_sequence=['red'], markers = True)
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
