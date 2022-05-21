import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy
from math import log2, ceil
import pywt
import parselmouth
import wave
import base64
from PIL import Image
import plotly.figure_factory as ff
from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
from scipy.signal import resample

##############################################
#            Fonctions des onglets           #
##############################################
def intro():
    # Contexte et Objectif
    st.markdown(
        """
            # Présentation du Data Challenge 2021
            
            :dart: Détection de chants d'oiseaux dans des enregistrements audio  
            :bird: Utilisation de 8000 enregistrements smartphone réalisés par les utilisateurs de l’application Warblr  
            :crystal_ball: Apprentissage via des méthodes de Machine Learning  

            ## Petite histoire du son
            Le son se produit lorsque quelque chose ***bouge***. Ce mouvement provoque la ***compression*** et la ***décompression*** de l'air autour de l'objet en une ***vague***.  
            Vos oreilles peuvent ***détecter ces ondes de compression*** et votre cerveau les ***interprète comme des sons***.
        """
    )
    st.image("https://docplayer.fr/docs-images/17/98421/images/3-0.png", width = 400)
    st.markdown(
        """
            Un son est caractérisé par 2 paramètres :
            - ***sa fréquence*** qui s'exprime en Hertz et correspond au nombre d’oscillations par seconde d’un phénomène. Plus la fréquence est basse, plus le son est grave; plus elle est haute et plus le son est aigu.
            - ***son amplitude*** qui s'exprime en Décibel et indique si le son est faible ou fort.
        """
    )

    # Démo du fonctionnement des paramètres
    col1, col2 = st.columns((1, 3))
    col1.markdown(
        """
            ##### Démonstration 
        """
    )
    freq = col1.slider(
        "Fréquence (Hz)"
        , min_value=220
        , max_value=2000
        , value=450, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
    )
    amp = col1.slider(
        "Amplitude (dB)"
        , min_value = 0
        , max_value = 10
        , value = 1
        , format = "%d"
    )
    fig = demo_freq_amplitude(freq, amp)
    col2.plotly_chart(fig)

    # Partie 2 : Signal analogique VS numerique
    st.markdown(
        """
            ## Signal analogique VS numérique
            Un signal analogique est ***continue***. Il est donc représenté par un nombre infini de points.  
            Tandis qu'un signal numérique se compose d'une série ***"d'instantannés"*** de l'amplitude du signal au cours du temps. Chaque instantané est appellé un ***sample***.  
            Un signal numérique n'est donc qu'une liste de nombres facilement stockable et utilisable qui pourra être très proche du signal analogue selon le choix des 2 paramètres suivants :
            - le sampling rate : c'est nombre d'instantannés pris en 1 seconde
            - le sample depth : c'est la précision avec laquelle on va estimer l'instantanné (i.e. le nombre de bits)

            Pour avoir une qualité d'audio idéal et proche de l'analogique, il faut que ces 2 paramètres soient suffisamment élevés.  
            Cependant, à noter que plus ils sont élevés, plus le fichier devient "lourd".
        """
    )

    ## Démo sampling et bit
    col1, col2 = st.columns((1, 3))
    col1.markdown(
        """
            ##### Démonstration 
        """
    )
    sampling = col1.slider(
        "Sampling rate"
        , min_value=1000
        , max_value=30000
        , value=20000, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None
    )
    precision = col1.slider(
        "Sample depth"
        , min_value = 0
        , max_value = 5
        , value = 1
        , format = "%d"
    )
    fig = demo_sampling_precision(sampling, precision)
    col2.plotly_chart(fig)

def globale():
    st.write("# Description des données 🔍")

    # Statistique générale
    col1, col2 = st.columns((1, 2))
    fig1 = globale_stat()
    col1.plotly_chart(fig1,use_container_width=True)
    fig2 = globale_sunburst()
    col2.plotly_chart(fig2,use_container_width=True)

    # Histogramme durée
    fig = globale_histogramme()
    st.plotly_chart(fig,use_container_width=True)

    # Boxplot max frequency
    fig = globale_boxplot()
    st.plotly_chart(fig,use_container_width=True)

def indiv():
    st.write("# Analyse d'un fichier audio 🎼")
    uploaded_file = st.file_uploader("Choisir un fichier audio à analyser", type = ["WAV", "AIF", "MP3", "MID"])
    if uploaded_file:
        # Chargement de l'audio + ecoute
        st.audio(uploaded_file, format="audio/wav", start_time=0)
        fp = wave.open(uploaded_file, 'r')
        sampling_rate = fp.getframerate()
        samples = fp.readframes(-1)
        samples = np.frombuffer(samples, dtype='int16')

        # Affichage des stats
        st.write("## Caractéristiques principales")
        col1, col2, col3 = st.columns((1, 1, 1))
        fig1 = ind_stat_sampling(sampling_rate)
        col1.plotly_chart(fig1,use_container_width=True)
        fig2 = ind_stat_nbbits(fp.getsampwidth())
        col2.plotly_chart(fig2,use_container_width=True)
        fig3 = ind_stat_freq(sampling_rate, samples)
        col3.plotly_chart(fig3,use_container_width=True)

        # Description Fourier + GIF
        st.write("## Transformée de Fourier")
        st.write('#### Théorie')
        col1, col2 = st.columns((1, 1))
        col1.markdown("""
            La transformée de Fourier permet de passer du domaine temporel *(Temps x Amplitude)* au domaine fréquentiel *(Fréquence x Amplitude)*.  
              
            Pour cela, elle considère une fonction d'entrée f (<span style="color:indianred">en rouge</span>) dans le domaine temporel et va chercher à la décomposer en plusieurs ondes sinusoïdales simples.  
            On peut ainsi pour chaque onde définir la fréquence ainsi que l'amplitude et obtenir la fonction de sortie $\hat{f}$ (<span style="color:royalblue">en bleu</span>) dans le domaine fréquentiel.
        """,unsafe_allow_html=True)
        file_ = open("Final_dataviz/images/Fourier_transform_time_and_frequency_domains.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        col2.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True
        )

        #Application sur nos données
        st.write("#### Application sur l'audio")
        col1, col2 = st.columns((1, 1))
        fig1 = ind_time_amplitude(samples, sampling_rate)
        col1.plotly_chart(fig1,use_container_width=True)
        fig2 = ind_freq_amplitude(samples, sampling_rate)
        col2.plotly_chart(fig2,use_container_width=True)

        st.write("""
            #### Création du spectrogramme  
            La transformée de Fourier permet d'obtenir le domaine fréquentiel avec une haute précision mais en perdant la notion de temps.  
            Pour la réintégrer, une possibilité est d'appliquer la transformée de Fourier sur une petite fenêtre temporelle et ensuite de la faire glisser.  
            On peut alors obtenir une approximation de la fréquence dans le domaine temporel et ainsi traçer le spectrogramme.
        """)
        fig = ind_spectrogram(samples, sampling_rate)
        st.plotly_chart(fig,use_container_width=True)

        # # Spectrogramme
        # st.write("## Transformée en ondelettes")
        # st.write("RAF : explication sur le wavelet transform + 3D plot + décomposition")
        # fig = ind_decomposition(samples, sampling_rate)
        # st.pyplot(fig,use_container_width=True)
        
        # image = Image.open('Final_dataviz/images/equipe.png')
        # st.image(image, width = 300)

def modelisation():
    st.write("""
        # Equilibre de la variable cible
        Pour rappel, les données mises à disposition ont les caractéristiques suivantes :
        """)
    # Statistique Avant Data Augmentation
    col1, col2 = st.columns((1, 2))
    fig1 = globale_stat()
    col1.plotly_chart(fig1,use_container_width=True)
    fig2 = globale_sunburst()
    col2.plotly_chart(fig2,use_container_width=True)
    st.write("""
        Afin d'avoir une base d'apprentissage avec une **variable cible plus équilibrée**, nous avons testé 2 méthodes :  

        1.  Création de **nouveaux audios sans oiseaux** en utilisant les méthodes suivantes :
            * **Mix d'audio** : on combine 2 audios en faisant la moyenne des valeurs
            * **Rotation** : on coupe l'audio à un point et on inverse les places des deux parties
            * **Ajout de bruits** : avec un factor très faible, on ajoute des valeurs aléatoires à chaque points de l'audio
            * **Mise de points à 0** : on remplace certaines valeurs par 0
        1. Création d'une **pondération** inversement proportionnelle à la répartition pour surpondérer les audios sans oiseaux
    """)
    fig = mod_stat_barchart()
    st.plotly_chart(fig,use_container_width=True)

    st.write("# Conversion des audios en images")
    # Transformée de Fourier
    col1, col2 = st.columns((1, 1))
    col1.markdown("""
        La **transformée de Fourier** permet de passer du **domaine temporel** *(Temps x Amplitude)* au **domaine fréquentiel** *(Fréquence x Amplitude)*.  
            
        Pour cela, elle considère une fonction d'entrée f (<span style="color:indianred">en rouge</span>) dans le domaine temporel et va chercher à la **décomposer en plusieurs ondes sinusoïdales**.  
        On peut ainsi **pour chaque onde** définir la **fréquence** ainsi que l'**amplitude** et obtenir la fonction de sortie $\hat{f}$ (<span style="color:royalblue">en bleu</span>) dans le domaine fréquentiel.  

        Cette transformée a cependant un **défaut majeur** : en décomposant en **sinusoïdes** constantes, on **perd la dimension temporelle**. Il n'est donc **pas possible** de savoir à **quel moment chaque fréquence est représentée** à moins de la faire sur une **petite fenêtre temporelle** que l'on fait ensuite glisser. On obtient alors un **spectrogramme**.       
    """,unsafe_allow_html=True)
    file_ = open("Final_dataviz/images/Fourier_transform_time_and_frequency_domains.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col2.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True
    )
      
    # Transformée en ondelettes
    col1, col2 = st.columns((2, 1))
    file_ = open("Final_dataviz/images/Continuous_wavelet_transform.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col1.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True
    )
    col2.write("""
        La **transformée en ondelettes** s'inspire de la précédente mais utilise cette fois une **petite ondulation** au lieu d'une sinusoïde pour la **décomposition**.
        Grâce à cela, il est possible de **récupérer précisément à quel moment une fréquence est présente et à quelle amplitude**.  
            
        Il existe une multitude de forme d'ondelettes mais nous utiliserons l'une des plus classiques : celle de **Morlet**.  

    """)
    st.write("""  
        Une fois la transformée appliquée, il faut désormais se ramener à une **image en 2D où x représente le temps, y la fréquence et z (la couleur) l'amplitude**.  
        Toutefois, pour ensuite pouvoir être intégrée dans un modèle, il est nécessaire que l'**échelle** de l'image soit **constante** pour tous les audios !  

        Pour la **fréquence**, celle-ci a été **discrétiser** en un nombre de fixe de classes ce qui assure une échelle constante sur l'axe Y. Un **filtre** a ensuite été appliqué pour ne conserver que les classes représentant une **fréquence supérieure à 1.5kHz**, le minimum pour un chant d'oiseau.  
        Pour le **temps**, il est **impossible de tronquer l'audio** pour fixer un nombre de secondes au risque de ne pas avoir le chant d'oiseau sur la partie conservée. Deux méthodes ont donc été utilisées simultanément :
        - **le downsampling** : on réduit le nombre d'instantannés constituant l'audio
        - **le resizing** : une fois l'image générée, on altère sa dimension pour fixer l'axe x

        De cette façon, l'ensemble des fichiers audios sont convertis en image, un **scalogramme**, que l'on va ensuite pouvoir utiliser pour la modélisation.
    """)


    st.write("""
        # Description du modèle
        ### Réseau de neurones classique
    """)
    col1, col2 = st.columns((1, 1))
    col1.write("""
    Un réseau de neurones s'inspire du fonctionnement de notre cerveau.  

    C'est un ensemble composé de trois types de neurones:
    - des neurones externes qui reçoivent des informations
    - des neurones externes qui envoient des informations
    - des neurones internes qui connectent les deux couches de neurones externes entre elles

    Chaque neurones agrège l'information qu'il reçoit et la transforme avant de l'envoyer.
    """)
    file_ = open("Final_dataviz/images/feed-forward-neural-network.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col2.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True
    )
    st.write("### Réseau de neurones à convolution")
    col1, col2 = st.columns((2, 1))
    col2.write("""
    Lorsque l'input est une image, il est nécessaire d'ajouter une contrainte supplémentaire : la forte corrélation entre un pixel et ceux qui l'entourent.  

    Il faut pour ça introduire les convolutions. Celles-ci prennent une image en entrée et vont la simplifier en réduisant ses dimensions, baissant sa qualité et gardant que les informations importantes de l'image (forme, couleur, etc.).
    """)
    file_ = open("Final_dataviz/images/cnn_plus_pool.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col1.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="100%">',
        unsafe_allow_html=True
    )
    st.write("### Réseau récurrent")
    col1, col2 = st.columns((1, 2))
    file_ = open("Final_dataviz/images/rnn.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col2.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True
    )
    col1.write("""
    Lorsque l'input est une donnée séquentielle (un texte, un audio, etc.), deux nouvelles contraintes viennent s'ajouter : la taille peut varier d'un input à l'autre et l'ensemble peut être très volumineux.  

    Le réseau récurrent va alors s'entraîner sur une portion fixe de la séquence en basant ses calculs sur les données de la séquence et sur les calculs effectués sur les séquences précédentes.  
    """)
    st.write("""
    En combinant l'ensemble de ses structures, nous avons construit un CRNN pour notre tâche de prédiction.
    Le modèle prendra en entrée le scalogramme généré à partir de l'audio, puis doit prédire la présence ou l'absence
    de l'oiseau.
    """)
    st.image("Final_dataviz/images/archi_reseau.png", width = 1000)
    st.write("# Performance du modèle")
    col1, col2 = st.columns((1, 1))
    col1.write("Performance avec Data Augmentation sur l'apprentissage")
    # Add plotly + check données MC
    filename = "Modele\historique_entrainement_70_epochs_v1.pkl"
    fig = plot_perf(filename)
    col1.plotly_chart(fig,use_container_width=True)
    cm = [[0.3, 0.7],
        [0.12, 0.88]]
    fig = confusion_matrix(cm)
    col1.plotly_chart(fig,use_container_width=True)

    col2.write("Performance avec la pondération sur l'apprentissage")
    # Add plotly + check données MC
    filename = "Modele\historique_entrainement_70_epochs_over_70_14-12-2021-14-31-33.pkl"
    fig = plot_perf(filename)
    col2.plotly_chart(fig,use_container_width=True)
    cm = [[0.44, 0.55],
        [0.2, 0.8]]
    fig = confusion_matrix(cm)
    col2.plotly_chart(fig,use_container_width=True)

    st.write("# Application du modèle")
    uploaded_file = st.file_uploader("Choisir un fichier audio à prédire", type = ["WAV", "AIF", "MP3", "MID"])
    if uploaded_file:
        # Chargement de l'audio + ecoute
        st.audio(uploaded_file, format="audio/wav", start_time=0)
        fp = wave.open(uploaded_file, 'r')
        sampling_rate = fp.getframerate()
        samples = fp.readframes(-1)
        samples = np.frombuffer(samples, dtype='int16')
        mpath = "Modele\model_70_epochs_over_70_14-12-2021-14-31-33.h5"
        p = apply_model(samples, mpath)
        resultat = model_output(p)
        print(p)
        st.write(resultat)
        # Puis appliquer la fonction avec le paramètre samples + path du modèle 
        # en retirant la lecture du fichier dans la fonction
        col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
        col2.image("Final_dataviz/images/equipe.png", use_column_width=True)
############################################## 
#             Fonctions annexes              #
##############################################
## Home Page

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
    fig2 = px.line(dataframe, x="x", y="y", color_discrete_sequence=['red'], markers=True)
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


## Globale
def globale_stat():
    base_sun = pd.read_csv('Final_dataviz/intermediaire/base_sun.csv')
    base_histo = pd.read_csv('Final_dataviz/intermediaire/base_histo.csv')
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
        'apprentissage' : '#7BEBFF'
        , 'validation' : '#9183F5'
    }
    base_sun = pd.read_csv('Final_dataviz/intermediaire/base_sun.csv')
    fig = px.sunburst(
        base_sun
        , path=["base", "hasbird"]
        , values='nb audios'
        , color='base', color_discrete_map = code_couleur
        , title= "Répartition des audios selon leur base et la présence d'oiseau ou non"
    )
    fig.update_traces(textinfo="label+percent entry", textfont=dict(family="Arial Black"))
    return(fig)

def globale_histogramme():
    base_histo = pd.read_csv('Final_dataviz/intermediaire/base_histo.csv')
    fig = px.histogram(
        data_frame=base_histo
        , x="duration_class"
        , histnorm = 'percent'
        , color_discrete_sequence=['indianred']
        , labels = {'duration_class' : 'Classe de temps (secondes)'}
        , title = "Répartition des audios selon leur durée"
    )
    fig.update_xaxes(categoryorder='array', categoryarray= ["inf 9.9","]9.9, 10.0]","]10.0, 10.1]","]10.1, 10.2]","]10.2, 10.3]","]10.3, 10.4]","]10.4, 10.5]", "sup 10.5"])
    fig.update_layout(yaxis_title="Pourcentage")
    return(fig)

def globale_boxplot():
    base_box = pd.read_csv('Final_dataviz/intermediaire/base_boxplot.csv')
    fig = px.box(base_box
        , x='hasbird'
        , y="Frequence max"
        , color="hasbird"
        , title = "Distribution de la fréquence maximale selon la présence d'oiseau ou non"
        , labels = {'hasbird' : ""}
    )
    return(fig)


## Individual
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
                    {'range': [0, 1], 'color': 'red'},
                    {'range': [1, 4], 'color': 'orange'},
                    {'range': [4,8], 'color': 'yellow'},
                    {'range': [8,16], 'color': 'lightgreen'},
                    {'range': [16,32], 'color': 'green'}
                    ]
            }
        )
    )
    return(fig)

def ind_stat_freq(sampling_rate, samples) :
    n = len(samples)
    T = 1/sampling_rate
    normalize_samples = samples/max(samples)
    yf = scipy.fft.fft(normalize_samples)
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
    return(fig)

def ind_time_amplitude(samples, sampling_rate):
    duration_of_sounds = len(samples)/sampling_rate
    x = np.linspace(0,duration_of_sounds, len(samples))
    fig = px.line(x=x, y=samples, title="Domaine temporel", labels = {'x' : "Temps (secondes)", 'y' : "Amplitude"}, color_discrete_sequence  = ['indianred'])
    return(fig)  

def ind_freq_amplitude(samples, sampling_rate):
    normalize_samples = samples/max(samples)
    n = len(normalize_samples)
    T = 1/sampling_rate
    yf = scipy.fft.fft(normalize_samples)
    xf = np.linspace(0, int(1/(2*T)), int(n/2))
    fig,ax = plt.subplots()
    final_y = 2.0/n * np.abs(yf[:n//2])
    fig = px.line(x=xf, y=final_y, title = "Domaine fréquentiel", labels = {'x' : "Fréquence", 'y' : "Amplitude"}, color_discrete_sequence = ['royalblue'])
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'y', y0= 0.001, y1= 0.001,   # adding a horizontal line at Y = 1
        xref= 'paper', x0= 0, x1= 1,
        line=dict(color='Red')
            ) 
        ])
    return(fig)    

def ind_decomposition(samples, sampling_rate):
    nb_level = ceil(log2(sampling_rate)-3)
    coeffs = pywt.wavedec(samples, 'bior6.8', level= nb_level, mode = 'per')
    fig = plt.figure(figsize = (30,50))
    for i in range(nb_level+1):
        plt.subplot(nb_level+1,1,i+1)
        plt.plot(coeffs[nb_level-i])
        plt.ylabel('cD'+str(i+1))
        plt.title('Approx coeff '+str(i+1))   
    return(fig)

def ind_spectrogram(samples, sampling_rate):
    sound = parselmouth.Sound(values=samples,sampling_frequency=sampling_rate)
    sound.pre_emphasize()
    spectrogram = sound.to_spectrogram()
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    data = [go.Heatmap(x=X, y=Y, z=sg_db, zmin = -100, zmax = 0, colorscale = 'turbo')]
    layout = go.Layout(
        yaxis=dict(title='Fréquence (Hz)'),
        xaxis=dict(title='Temps (s)')
    )
    fig = go.Figure(data=data, layout=layout)
    return(fig)

## Modelisation
def mod_stat_barchart():
    base_barchart = pd.read_csv('Final_dataviz/intermediaire/base_barchart.csv')
    code_couleur = {
        'sans oiseau' : '#7BEBFF'
        , 'avec oiseau' : '#9183F5'
    }
    fig = px.bar(
        base_barchart, x="Apprentissage", y="nb audios"
        , color="hasbird", color_discrete_map = code_couleur
        , barmode="group"
        , text = "nb audios")
    fig.update_traces(textfont=dict(family="Arial Black"))
    return(fig)

def confusion_matrix(cm):
    labels = ['sans oiseau', 'avec oiseau']
    title = 'Matrice de confusion'
    data = go.Heatmap(z=cm, x = labels, y=labels, colorscale='YlGnBu')
    # 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds'
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "font": {"color": "grey",'size':20},
                    "text": str(value),
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
    return(fig)

def plot_perf(filename):
    perf_df = pd.read_pickle(filename).reset_index()
    fig = px.line(perf_df, x='epoch', y=["loss", "accuracy"])
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
    res = ""
    if (p < 0.1) : 
        res = "Je peux dire avec une quasi certitude que je n'ai pas entendu d'oiseau."
    elif (p >= 0.1) & (p < 0.4) :
        res = "Sans vouloir m'avancer, je dirais qu'il n'y a pas d'oiseau dans cet audio."
    elif (p >= 0.4) & (p < 0.5) : 
        res = "J'ai du mal à me decider. Mais il ... n'y a pas d'oiseau ?"
    elif (p >= 0.5) & (p < 0.6) :
        res = "J'ai du mal à me decider. Mais il ... y a un oiseau ?"
    elif (p >= 0.6) & (p < 0.9) :
        res = "Je dirais qu'il y a un oiseau dans cet audio. Dites-moi que j'ai raison, s'il-vous-plaît."
    elif (p >= 0.9) :
        res = "S'il n'y a pas d'oiseau dans cet audio, reinitialisez-moi complètement !"
    return(res)
