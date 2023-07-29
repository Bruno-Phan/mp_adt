import streamlit as st
from streamlit_option_menu import option_menu
from pycaret.anomaly import *
from streamlit_extras.switch_page_button import switch_page
from preprocess import DataProcessingTool

st.set_page_config(page_title="MP_ADT", layout="wide",)

st.markdown(f'<h1 style="text-align:center;">Anomaly Detection Tool</h1>', unsafe_allow_html=True)

selected=option_menu(menu_title=None,
# options=["Analyze data", "Train new model", "Use own model"], 
options=["Home", "Supervised AD", "Unsupervised AD", "Use own model"], 
orientation="horizontal")

dprocess=DataProcessingTool()

if "data" not in st.session_state:
    st.session_state.data="none"

def main():
    def change_data_state():
        st.session_state.data="yes"

    # opt = st.radio("Dataset Type:",["Unlabeled", "Labeled"])
    # uploaded = st.file_uploader("Upload your Labeled CSV data", type=["csv"], on_change=change_data_state)

    # if selected=="Analyze data" and uploaded is not None:
    #     data = pd.read_csv(uploaded)
    #     pyg.walk(data,env='Streamlit', dark="light")
    #     st.cache_data.clear()
    #     st.session_state.data="none"
    #     dprocess.reset()

    if selected=="Unsupervised AD":
    #  st.runtime.legacy_caching.clear_cache()
        switch_page("unsupervised ad")
        dprocess.reset()

    elif selected=="Supervised AD":
    #  st.runtime.legacy_caching.clear_cache()
        switch_page("supervised ad")
        dprocess.reset()

    elif selected=="Use own model":
    #  st.runtime.legacy_caching.clear_cache()
        switch_page("use own model")
        dprocess.reset()
       
if __name__ == "__main__":
    main()

