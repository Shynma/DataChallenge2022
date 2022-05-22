import streamlit as st

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
  
  st.markdown("# Petite histoire du son")
  col1,col2 = st.columns((3,1))
  col1.markdown(
    """
      Deux caractéristiques du son :
      -	La ***fréquence*** (nombre d’oscillations par seconde) 
      -	L’***amplitude*** (intensité des oscillations)
    """
  )
  col2.image("https://docplayer.fr/docs-images/17/98421/images/3-0.png", width = 400)

def use_case():
  st.image("images/banniere_fixe.png", use_column_width=True)
  st.image("images/UC.png", use_column_width=True)
