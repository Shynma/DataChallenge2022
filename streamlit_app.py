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
        ("Qui sommes-nous ?", functions.intro),
        ("Notre projet", functions.projet),
        ("Pour aller plus loin", functions.use_case)
    ]
)

# Fonctionnement de l'application
def run():
    #st.set_page_config(page_title="Vivatch - Epsilon", page_icon="https://img.icons8.com/color/48/000000/stork.png", layout="wide", initial_sidebar_state="auto", menu_items=None)
    #demo_name = st.sidebar.selectbox("Menu", list(DEMOS.keys()), 0)
    #demo = DEMOS[demo_name]
    #demo()
    st.image("images/banniere_vivatech.png", use_column_width=True)
    functions.intro()
    functions.projet()
    functions.use_case()


# Lancement de l'application
if __name__ == "__main__":
    st.markdown(
        """
        <style>
        @font-face {
          font-family: 'Tangerine';
          font-style: normal;
          font-weight: 400;
          src: url(https://fonts.gstatic.com/s/tangerine/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
          unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
        }

          h1  {
            font-family: 'Tangerine';
            font-size: 48px;
          }
        </style>
        """
        , unsafe_allow_html=True
    )
    run()
