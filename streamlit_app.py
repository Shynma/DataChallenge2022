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
            font-family: 'trashbold';
            src: url('https://raw.githubusercontent.com/Shynma/DataChallenge2022/blob/master/trash-bold-webfont.woff2') format('woff2'),
                 url('https://raw.githubusercontent.com/Shynma/DataChallenge2022/blob/master/trash-bold-webfont.woff') format('woff');
            font-weight: normal;
            font-style: normal;

        }

          h1  {
            font-family: 'trashbold';
            font-size: 48px;
          }
        </style>
        """
        , unsafe_allow_html=True
    )
    run()
