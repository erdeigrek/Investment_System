import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Learning Monitor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Learning Monitor — System inwestycyjny")

root = Path(__file__).resolve().parent


st.markdown(
    """
### Jak korzystać
- Wejdź w **Plan nauki** i wybierz punkt.
- Potem przejdź do **Quiz** i sprawdź zrozumienie.
"""
)
