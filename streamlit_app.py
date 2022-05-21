# Import Packages
import streamlit as st
from streamlit.logger import get_logger
import functions
from collections import OrderedDict
import streamlit.components.v1 as components

LOGGER = get_logger(__name__)

# Dictionaires des onglets
DEMOS = OrderedDict(
    [
        ("Introduction", functions.intro),
        ("Description des données", functions.globale),
        ("Analyse d'un fichier audio", functions.indiv),
        ("Modélisation", functions.modelisation)
    ]
)

# Fonctionnement de l'application
def run():
    st.set_page_config(page_title="Data Challenge - Cigogne", page_icon="https://img.icons8.com/color/48/000000/stork.png", layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.image("images/banniere_fixe.png", use_column_width=True)
    demo_name = st.sidebar.selectbox("Choisir un onglet", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name]
    demo()


# Lancement de l'application
if __name__ == "__main__":
    run()
