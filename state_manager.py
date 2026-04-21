import streamlit as st

def init_state():
    if "results" not in st.session_state:
        st.session_state.results = []

def reset_case():
    st.session_state.results = []
