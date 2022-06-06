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

height_carac = 300

##############################################
#            Fonctions des onglets           #
##############################################
def intro():
  st.image("images/banniere_vivatech.png", use_column_width=True)
  st.image("images/Epsilon0.png", use_column_width=True)
  st.markdown('#')
  st.markdown('#')
  st.markdown('#')
  st.image("images/data_science.png", use_column_width=True)

def projet():
  st.image("images/banniere_vivatech.png", use_column_width=True)
  st.image("images/contexte.png", use_column_width=True)
  
  st.markdown(
    """
      # ANALYSE D'UN AUDIO : DE LA THÉORIE ...
      ### PETITE HISTOIRE DU SON
     """
  )
  col1, col2 = st.columns((1, 2))
  col1.markdown("**Fréquence** : nombre d’oscillations par seconde (Hz)")
  freq = col1.slider(
      ""
      , min_value=220
      , max_value=2000
      , value=450, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  col1.markdown("**Amplitude** : intensité des oscillations (dB)")
  amp = col1.slider(
      ""
      , min_value = 0
      , max_value = 10
      , value = 1
      , format = "%d"
  )
  fig = demo_freq_amplitude(freq, amp)
  col2.plotly_chart(fig)
  
  st.markdown(
    """
      ### RÉALITÉ VS INFORMATIQUE
    """
  )
  col1, col2 = st.columns((1, 2))
  col1.markdown("**Sampling rate** : nombre d’échantillon par seconde")
  sampling = col1.slider(
      ""
      , min_value=1000
      , max_value=30000
      , value=20000, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
  )
  col1.markdown("**Sample depth** : précision d’un échantillon")
  precision = col1.slider(
      ""
      , min_value = 0
      , max_value = 5
      , value = 1
      , format = "%d"
  )
  fig = demo_sampling_precision(sampling, precision)
  col2.plotly_chart(fig)
  
  st.markdown(
    """
      ### CONVERSION VERS UNE IMAGE
    """
  )
  col1, col2 = st.columns((2, 1))
  file_ = open("images/Continuous_wavelet_transform.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  col1.markdown(
      f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
      unsafe_allow_html=True
  )
  col2.write("""
      1. **Application de la transformée en ondelettes :**
        - Inspiré de la transformée de Fourier
        - Utilisation d'une petite ondulation (ici, Morlet) pour obtenir la fréquence et l'amplitude à chaque instant t
      2. **Création du scalogramme :**
        - Utilisation du temps (en X), de la fréquence (en Y) et de l'amplitude (en Z - couleur) pour obtenir une image en 2D
        - Normalisation de la taille de l'image : 
          - discrétisation de la fréquence pour Y
          - downsampling et resizing pour X
  """)
  
  st.markdown(
    """
      ### UTILISATION D'UN MODÈLE D'IA
    """
  )
  col1, col2 = st.columns((1, 1))
  col1.write("""
    ##### RÉSEAU DE NEURONES CLASSIQUE
      - S'inspire du fonctionnement du cerveau humain
      - Possède 3 types de composants :
        - neurones externes qui envoient des informations
        - neurones externes qui reçoivent des information
        - neurones internes qui connectent les deux couches de neurones externes entre elles
      - Aggrège et transforme l'information à chaque étape
  """)
  file_ = open("images/feed-forward-neural-network.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  col2.markdown(
      f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
      unsafe_allow_html=True
  )

  col1, col2 = st.columns((2, 1))
  col2.write("""
    ##### RÉSEAU DE NEURONES À CONVOLUTION
      - Prise en compte de la forte corrélation entre un pixel et ceux qui l'entourent
      - Simplification de l'information en entrée en réduisant dimension et qualité tout en gardant les informations essentielles
  """)
  file_ = open("images/cnn_plus_pool.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  col1.markdown(
      f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="100%">',
      unsafe_allow_html=True
  )

  col1, col2 = st.columns((1, 2))
  file_ = open("images/rnn.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  col2.markdown(
      f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
      unsafe_allow_html=True
  )
  col1.write("""
  ##### RÉSEAU RÉCURRENT
    -	Adaptation du réseau de neurones aux données de taille variables (texte, audio)
    -	Division de l’information en entrée en portions de taille fixe
    - Prédiction sur une portion en utilisation ses données et le résultat sur les portions précédentes
  """)
 
  
  st.markdown(
    """
      # ANALYSE D'UN AUDIO : ... À LA PRATIQUE
      ### DESCRIPTION DES DONNÉES
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
  
  st.markdown(
    """
      ### MODÉLISATION
    """
  )
  st.markdown("""
  En combinant l'ensemble de ses structures, nous avons construit un **CRNN** pour notre tâche de prédiction.
  Le modèle prendra en entrée le scalogramme généré à partir de l'audio, puis doit prédire la présence ou l'absence
  de l'oiseau.
  """)
  
  filename = "modele/historique_entrainement_70_epochs_over_70_14-12-2021-14-31-33.pkl"
  fig = plot_perf(filename)
  col1, col2 = st.columns((1, 1))
  col1.plotly_chart(fig,use_container_width=True)
  cm = [[0.44, 0.55],
      [0.2, 0.8]]
  fig = confusion_matrix(cm)
  col2.plotly_chart(fig,use_container_width=True)
    
  st.markdown(
    """
      ### APPLICATION
    """
  )
  
  option = st.selectbox("Sélection d'un audio de test",('Audio 1', 'Audio 2', 'Audio 3'))
  if option == 'Audio 1': 
    test_file = "data/00cc9afb-40da-4ca3-a4fe.wav"
  elif option == 'Audio 2':
    test_file = "data/0a0b783d-f9a3-4652-a01d.wav"
  elif option == 'Audio 3' :
    test_file = "data/0a4e8000-574c-46b8-a847.wav"
  else :
    test_file = ""
  
  if test_file != "" :
    st.audio(test_file, format="audio/wav", start_time=0)
    fp = wave.open(test_file, 'r')
    sampling_rate = fp.getframerate()
    samples = fp.readframes(-1)
    samples = np.frombuffer(samples, dtype='int16')
   
    # Affichage des stats
    st.write("##### CARACTÉRISITIQUES DE L'AUDIO")
    col1, col2, col3 = st.columns((1, 1, 1))
    fig1 = ind_stat_sampling(sampling_rate)
    col1.plotly_chart(fig1,use_container_width=True)
    fig2 = ind_stat_nbbits(fp.getsampwidth())
    col2.plotly_chart(fig2,use_container_width=True)
    fig3 = ind_stat_freq(sampling_rate, samples)
    col3.plotly_chart(fig3,use_container_width=True)
    
    # Application du modèle
    mpath = "modele/model_70_epochs_over_70_14-12-2021-14-31-33.h5"
    p = apply_model(samples, mpath)
    resultat = model_output(p)
    st.write("##### PRÉDICTION")
    st.image("images/fleche.png")
    st.markdown(resultat,unsafe_allow_html=True)
  
#   uploaded_file = st.file_uploader("Choisir un fichier audio à analyser", type = ["WAV", "AIF", "MP3", "MID"])
#   if uploaded_file:
#     # Chargement de l'audio + ecoute
#     st.audio(uploaded_file, format="audio/wav", start_time=0)
#     fp = wave.open(uploaded_file, 'r')
#     sampling_rate = fp.getframerate()
#     samples = fp.readframes(-1)
#     samples = np.frombuffer(samples, dtype='int16')
   
#     # Affichage des stats
#     st.write("## Caractéristiques principales")
#     col1, col2, col3 = st.columns((1, 1, 1))
#     fig1 = ind_stat_sampling(sampling_rate)
#     col1.plotly_chart(fig1,use_container_width=True)
#     fig2 = ind_stat_nbbits(fp.getsampwidth())
#     col2.plotly_chart(fig2,use_container_width=True)
#     fig3 = ind_stat_freq(sampling_rate, samples)
#     col3.plotly_chart(fig3,use_container_width=True)
    
#     # Application du modèle
#     mpath = "modele/model_70_epochs_over_70_14-12-2021-14-31-33.h5"
#     p = apply_model(samples, mpath)
#     resultat = model_output(p)
#     print(p)
#     st.write(resultat)


def use_case():
  st.image("images/banniere_vivatech.png", use_column_width=True)
  st.image("images/use_case.png", use_column_width=True)
  st.image("images/merci.png")

  
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
  )
  fig3.update_traces(hovertemplate='%{y:.3f} dB')
  fig3.data[0]['showlegend'] = True
  fig3.data[0]['name'] = 'Son analogique'
  fig3.data[1]['showlegend'] = True
  fig3.data[1]['name'] = 'Son numerique'
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
      'apprentissage' : color_blue2
      , 'validation' : color_orange
  }
  base_sun = pd.read_csv('data/base_sun.csv')
  fig = px.sunburst(
      base_sun
      , path=["base", "hasbird"]
      , values='nb audios'
      , color='base', color_discrete_map = code_couleur
      , title= "Répartition des audios selon leur base et la présence d'oiseau ou non"
  )
  fig.update_traces(textinfo="label+percent entry", textfont=dict(family="Arial Black"), hovertemplate="Donnees %{id} (%{value} audios)")
  return(fig)

def globale_boxplot():
  base_box = pd.read_csv('data/base_boxplot.csv')
  fig = px.box(base_box
      , x='hasbird'
      , y="Frequence max"
      , color="hasbird"
      , color_discrete_map = {'sans oiseau' : color_blue, 'avec oiseau' : color_green}
      , title = "Distribution de la fréquence maximale selon la présence d'oiseau ou non"
      , labels = {'hasbird' : ""}
  )
  return(fig)

def ind_stat_sampling(sampling_rate):
  fig = go.Figure(
      go.Indicator(
          mode = "number",
          value = round(sampling_rate/1000,1),
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "Sampling Rate"},
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
          title = {'text': "Sample depth"},
          gauge = {
              'axis': {'range': [None,32]},
              'bar': {'color': "grey"},
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
          title = {'text': "Fréquence maximale"},
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
                  "font": {"color": color_black,'size':20},
                  "text": "{:.0f}%".format(value * 100),
                  "xref": "x1",
                  "yref": "y1",
                  "showarrow": False
              }
          )
  layout = {
      "title": title,
      "xaxis": {"title": "Valeur prédite"},
      "yaxis": {"title": "Valeur réelle"},
      "annotations": annotations

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


def apply_model(data, model_path):
  model = load_model(model_path)
  wavelet_coeffs = convert_to_tensor([scalogram(data)]) ## correspond à nd_array (8,15000)
  return model.predict(wavelet_coeffs)[0][0]

def model_output(p):
  res = """<div style="text-align: center">{}</div>"""
  if (p < 0.1) : 
      res = res.format("""Je peux dire avec une quasi certitude que je n'ai <span style="color:"""+color_red+"""">pas entendu d'oiseau.</span>""")
  elif (p >= 0.1) & (p < 0.4) :
      res = res.format("""Sans vouloir m'avancer, je dirais qu'il n'y a <span style="color:"""+color_orange+"""">pas d'oiseau dans cet audio.</span>""")
  elif (p >= 0.4) & (p < 0.5) : 
      res = res.format("""J'ai du mal à me decider. Mais il ... n'y a <span style="color:"""+color_green+"""">pas d'oiseau ?</span>""")
  elif (p >= 0.5) & (p < 0.6) :
      res = res.format("""J'ai du mal à me decider. Mais il ... <span style="color:"""+color_blue2+"""">y a un oiseau ?</span>""")
  elif (p >= 0.6) & (p < 0.9) :
      res = res.format("""Je dirais qu'il <span style="color:"""+color_blue+"""">y a un oiseau</span> dans cet audio. Dites-moi que j'ai raison, s'il-vous-plaît.""")
  elif (p >= 0.9) :
      res = res.format("""<span style="color:"""+color_blue0+"""">S'il n'y a pas d'oiseau dans cet audio, reinitialisez-moi complètement !</span>""")
  return(res)
