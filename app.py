import streamlit as st
import os
from streamlit_option_menu import option_menu

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Import module
from modules import bootle_classification, extraction_sentiment, time_series_prediction,dashboard

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilih Module",
        options=["Dashboard", "Klasifikasi Botol", "Ekstraksi Sentimen", "Prediksi Time Series"],
        icons=["box", "chat-left-text", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# Routing sesuai pilihan
if selected == "Dashboard":
    dashboard.run()
elif selected == "Klasifikasi Botol":
    bootle_classification.run()
elif selected == "Ekstraksi Sentimen":
    extraction_sentiment.run()
elif selected == "Prediksi Time Series":
    time_series_prediction.run()
