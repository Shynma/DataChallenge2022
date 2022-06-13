import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import fft
import wave
import base64
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor
import pywt
from scipy.signal import resample

color_black = '#000000'
color_blue0 = '#317087'
color_blue = '#00A8FF'
color_blue2 = '#30D1D4'
color_green = '#00E6A2'
color_orange = '#FFBA00'
color_red = '#E93B23'

height_carac = 250




##############################################
#            Fonctions des onglets           #
##############################################
def intro():
  st.image("images/01-Epsilon.png", use_column_width=True)
  st.image("images/02-ExpertisesDsc.png", use_column_width=True)

def projet():
  st.image("images/03-ProjectDFG.png", use_column_width=True)
  st.image("images/04-TitreAudio1.png", use_column_width=True)
  
  st.markdown(
    """
    <center><font size='+1'><b> QUATRE PARAMÈTRES </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1, col2, col3, col4 = st.columns((3, 1, 1, 3))
  freq = col2.slider(
      "Fréquence (Hz)" # nombre d’oscillations par seconde 
      , min_value=220
      , max_value=2000
      , value=450, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  amp = col2.slider(
      "Amplitude (dB)" # intensité des oscillations 
      , min_value = 0
      , max_value = 10
      , value = 1
      , format = "%d"
  )
  fig = demo_freq_amplitude(freq, amp)
  col1.plotly_chart(fig, use_container_width = True)
  
  sampling = col3.slider(
      "Sampling rate" #  : nombre d’échantillon par seconde
      , min_value=1000
      , max_value=30000
      , value=20000, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  
  precision = col3.slider(
      "Sample depth" # : précision d’un échantillon
      , min_value = 0
      , max_value = 5
      , value = 1
      , format = "%d"
  )
  fig = demo_sampling_precision(sampling, precision)
  col4.plotly_chart(fig, use_container_width = True)
  
  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> UNE FONCTION </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1,col2,col3 = st.columns((1, 2.5, 1))
  col2.image("images/Continuous_wavelet_transform.gif", use_column_width=True)


  
  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> UN RÉSEAU DE NEURONES </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1,col2,col3 = st.columns((1, 2, 1))
  col2.image("images/feed-forward-neural-network.gif", use_column_width=True)

  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> COMPOSÉ DE ... </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1,col2 = st.columns((1.5, 1))
  col1.markdown("<center><b> 3 COUCHES DE CONVOLUTION </b></center>", unsafe_allow_html=True)
  col1.image("images/cnn_plus_pool.gif", use_column_width=True)
  col2.markdown("<center><b> 2 COUCHES DE RÉCURRENCE </b></center>", unsafe_allow_html=True)
  col2.image("images/rnn.gif", use_column_width=True)



  st.markdown("<br><br><br>", unsafe_allow_html=True)
  st.image("images/05-TitreAudio2.png", use_column_width=True)
  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> DES DONNÉES AUDIO </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  # Statistique générale
  col1, col2 = st.columns((1, 1))
  fig1 = globale_stat()
  col1.plotly_chart(fig1,use_container_width=True)
  fig2 = globale_sunburst()
  col2.plotly_chart(fig2,use_container_width=True)

  # Boxplot max frequency
  col1, col2, col3 = st.columns((1, 5, 1))
  fig = globale_boxplot()
  col2.plotly_chart(fig,use_container_width=True)

  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> UN MODÈLE ENTRAINÉE</b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1, col2, col3 = st.columns((1, 2, 1))
  cm = [[0.44, 0.55],
      [0.2, 0.8]]
  fig = confusion_matrix(cm)
  col2.plotly_chart(fig,use_container_width=True)

  st.markdown(
    """
    <br><br>
    <center><font size='+1'><b> 1 UTILISATION EN TEMPS RÉEL  </b></font></center>
    <br>
    """
    , unsafe_allow_html=True
  )
  col1, col2, col3 = st.columns((1, 1, 1))
  option = col2.selectbox("",('Audio 1', 'Audio 2', 'Audio 3'))
  if option == 'Audio 1': 
    file_name = "data/00cc9afb-40da-4ca3-a4fe.wav"
  elif option == 'Audio 2':
    file_name = "data/0a0b783d-f9a3-4652-a01d.wav"
  elif option == 'Audio 3' :
    file_name = "data/0a4e8000-574c-46b8-a847.wav"
  else :
    file_name = ""
  
  if file_name != "" :
    col2.audio(file_name, format="audio/wav", start_time=0)
    fp = load_audio(file_name)
#     fp = load_audio(test_file)
    sampling_rate = fp.getframerate()
    samples = fp.readframes(-1)
    samples = np.frombuffer(samples, dtype='int16')
   
    # Affichage des stats
    col1, col2, col3 = st.columns((1, 1, 1))
    fig1 = ind_stat_sampling(sampling_rate)
    col1.plotly_chart(fig1,use_container_width=True)
    fig2 = ind_stat_nbbits(fp.getsampwidth())
    col2.plotly_chart(fig2,use_container_width=True)
    fig3 = ind_stat_freq(sampling_rate, samples)
    col3.plotly_chart(fig3,use_container_width=True)
    
    # Application du modèle
    p = apply_model(samples)
    resultat = model_output(p)
    st.image(resultat, use_column_width=True)
  
  
def use_case():
  st.image("images/08-Usages.png", use_column_width=True)
  st.image("images/merci.png", use_column_width=True)

  
############################################################################################################################################ 
#                                                  Fonctions annexes                                                                       #
############################################################################################################################################
def load_audio(file_name) :
  return(wave.open(file_name, 'r'))

@st.cache(ttl=1*3600,allow_output_mutation=True)
def load_model_nlp() :
  return(load_model("modele/model_70_epochs_over_70_14-12-2021-14-31-33.h5"))

@st.cache(ttl=1*3600)
def load_data_base_sun() :
  return(pd.read_csv('data/base_sun.csv'))

@st.cache(ttl=1*3600)
def load_data_base_histo() :
  return(pd.read_csv('data/base_histo.csv'))

@st.cache(ttl=1*3600)
def load_data_boxplot() :
  return(pd.read_csv('data/base_boxplot.csv'))


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
      , plot_bgcolor = 'rgba(0, 0, 0, 0)'
      , paper_bgcolor = 'rgba(0, 0, 0, 0)'
#       , yaxis_title="Amplitude"
#       , hovermode='x unified'
  )
  fig.update_traces(line_color=color_blue, hovertemplate='Temps : %{x:.4f} s <br> Amplitude : %{y:.2f} dB')
#   fig.data[0]['name'] = 'Amplitude'
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
      , hovermode='x unified'
      , plot_bgcolor = 'rgba(0, 0, 0, 0)'
      , paper_bgcolor = 'rgba(0, 0, 0, 0)'
  )
  fig3.update_traces(hovertemplate='%{y:.3f} dB')
  fig3.update_layout(legend=dict(
      yanchor="bottom",
      y=0.01,
      xanchor="center",
      x=0.5
  ))
  fig3.data[0]['showlegend'] = True
  fig3.data[0]['name'] = 'Son analogique'
  fig3.data[1]['showlegend'] = True
  fig3.data[1]['name'] = 'Son numerique'
  return(fig3)

def globale_stat():
  base_sun = load_data_base_sun()
  base_histo = load_data_base_histo()
  trace1 = go.Indicator(
          mode = "number",
          value = base_sun['nb audios'].sum(),
          title = {'text': "<span style='font-size:0.8em;color:gray'>Nombre d'audio</span>"},
          domain={'x': [0.0, 1], 'y': [0.60, 1]},
          number = {"font":{"size":100}}
      )
  trace2 = go.Indicator(
          mode = "number",
          value = base_histo['framerate'].mean(),
          title = {'text': "<span style='font-size:0.8em;color:gray'>Sampling rate moyen</span>"},
          domain={'x': [0.0, 0.4], 'y': [0, 0.4]},
          number = {"font":{"size":100}}
      )
  trace3 = go.Indicator(
          mode = "number",
          value = base_histo['sampwidth'].mean(),
          title = {'text': "<span style='font-size:0.8em;color:gray'>Sample depth moyen</span>"},
          domain={'x': [0.7, 1], 'y': [0.0, 0.4]},
          number = {"font":{"size":100}}
      )
  fig = go.Figure(data = [trace1, trace2, trace3])
  return(fig)  

def globale_sunburst():
  code_couleur = {
      'apprentissage' : color_blue2
      , 'validation' : color_orange
  }
  base_sun = load_data_base_sun()
  fig = px.sunburst(
      base_sun
      , path=["base", "hasbird"]
      , values='nb audios'
      , color='base', color_discrete_map = code_couleur
      #, title= "Répartition des audios selon leur base et la présence d'oiseau ou non"
  )
  fig.update_traces(textinfo="label+percent entry", textfont=dict(family="Arial Black"), hovertemplate="Donnees %{id} (%{value} audios)")
  return(fig)

def globale_boxplot():
  base_box = load_data_boxplot()
  fig = px.box(base_box
      , x='hasbird'
      , y="Frequence max"
      , color="hasbird"
      , color_discrete_map = {'sans oiseau' : color_blue, 'avec oiseau' : color_green}
      , title = "Distribution de la fréquence maximale"
      , labels = {'hasbird' : ""}
  )
  fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)'
    , 'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    , 'showlegend':False
    , 'yaxis_visible' : False
    , 'yaxis_showticklabels' : False
    , 'title_x' : 0.5
    #, 'title_font_family' : "Arial Black"
    , 'title_font_color' : 'gray'
    , 'title_font_size' : 20
  })
  return(fig)

def ind_stat_sampling(sampling_rate):
  fig = go.Figure(
      go.Indicator(
          mode = "number",
          value = round(sampling_rate/1000,1),
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "<span style='font-size:0.8em;color:gray'>Sampling Rate</span>"},
          number={'suffix': "kHz"}
      )
  )
  fig.update_layout(height=height_carac)
  return(fig)

def ind_stat_nbbits(sampwidth):
  fig = go.Figure(
      go.Indicator(
          mode = "gauge+number",
          value = sampwidth,
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "<span style='font-size:0.8em;color:gray'>Sample depth</span>"},
          gauge = {
              'axis': {
                'range': [None,32]
                , 'dtick' : 8
                },
              'bar': {'color': "#989898"},
              'bordercolor': "white",
              'steps': [
                  {'range': [0, 1], 'color': color_red},
                  {'range': [1, 4], 'color': color_orange},
                  {'range': [4,8], 'color': color_green},
                  {'range': [8,16], 'color': color_blue2},
                  {'range': [16,32], 'color': color_blue}
                  ]
          }
      )
  )
  fig.update_layout(height=height_carac)
  return(fig)

def ind_stat_freq(sampling_rate, samples) :
  n = len(samples)
  T = 1/sampling_rate
  normalize_samples = samples/max(samples)
  yf = fft.fft(normalize_samples)
  xf = np.linspace(0, int(1/(2*T)), int(n/2))
  final_y = 2.0/n * np.abs(yf[:n//2])
  test_max = final_y >= 0.001
  frequency_max_amplitude = xf[max(np.where(test_max==max(test_max))[0])]

  fig = go.Figure(
      go.Indicator(
          mode = "number",
          value = frequency_max_amplitude,
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "<span style='font-size:0.8em;color:gray'>Fréquence maximale</span>"},
          number={'suffix': "Hz"}
      )
  )
  fig.update_layout(height=height_carac)
  return(fig)

def plot_perf(filename):
  perf_df = pd.read_pickle(filename).reset_index()
  fig = px.line(perf_df, x='epoch', y=["loss", "accuracy"],color_discrete_sequence=[color_orange, color_blue2])
  fig.update_traces(hovertemplate='%{y:.2f}')
  fig.update_layout(hovermode='x unified', legend=dict(title='Metrics'))
  return(fig)

def confusion_matrix(cm):
  labels = ['sans oiseau', 'avec oiseau']
  title = 'Matrice de confusion'
  
  colorscale_epsilon = [
    [0,color_red]
    ,[0.2,color_orange]
    ,[0.4,color_green]
    ,[0.6,color_blue2]
    ,[0.8,color_blue]
    ,[1,color_blue0]
  ]
  
  data = go.Heatmap(z=cm, x = labels, y=labels, colorscale=colorscale_epsilon)
  # 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'YlGnBu'
  annotations = []
  for i, row in enumerate(cm):
      for j, value in enumerate(row):
          annotations.append(
              {
                  "x": labels[j],
                  "y": labels[i],
                  "font": {"color": '#454545','size':20},
                  "text": "{:.0f}%".format(value * 100),
                  "xref": "x1",
                  "yref": "y1",
                  "showarrow": False
              }
          )
  layout = {
        # "title": title
       "xaxis": {"title": "Valeur prédite"}
      , "yaxis": {"title": "Valeur réelle"}
      , "annotations": annotations
      # , 'title_x' : 0.5
      # , 'title_font_color' : 'gray'
      # , 'title_font_size' : 20

  }
  fig = go.Figure(data=data, layout=layout)
  fig.update_traces(hovertemplate='Classe réelle : %{y}<br>Classe prédite : %{x}')
  fig.data[0].update(zmin=0, zmax=1)
  return(fig)

def scalogram(data) :
  wavelet = 'morl' # wavelet type: morlet
  sr = 15000 # sampling frequency: 8KHz
  widths = np.arange(1, 128) # scale
  dt = 1/sr # timestep difference

  # Create a filter to select frequencies between 1.5kHz and max
  frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
  lower = ([x for x in range(len(widths)) if frequencies[x] < 1500])[0]
  widths = widths[:lower] # Select scales in this frequency range
  y = resample(data/max(data), sr) # Normalize + downsample
  wavelet_coeffs, freqs = pywt.cwt(y, widths, wavelet = wavelet, sampling_period=dt)
  return(wavelet_coeffs)

def apply_model(data):
  model = load_model_nlp()
  wavelet_coeffs = convert_to_tensor([scalogram(data)]) ## correspond à nd_array (8,15000)
  return model.predict(wavelet_coeffs)[0][0]

def model_output(p):
  res = ""
  if (p < 0.1) : 
    res = "images/Resultat-10.png"
  elif (p >= 0.1) & (p < 0.4) :
    res = "images/Resultat-10a40.png"
  elif (p >= 0.4) & (p < 0.5) : 
    res = "images/Resultat-40a50.png"
  elif (p >= 0.5) & (p < 0.6) :
    res = "images/Resultat-50a60.png"
  elif (p >= 0.6) & (p < 0.9) :
    res = "images/Resultat-60a90.png"
  elif (p >= 0.9) :
    res = "images/Resultat-90.png"
  return(res)
