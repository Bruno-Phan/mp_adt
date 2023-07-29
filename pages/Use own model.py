import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pycaret.anomaly import *
import joblib
from preprocess import DataProcessingTool

dprocess=DataProcessingTool()

st.set_page_config(page_title="Anomaly_Detection_Tool",
                    layout="wide",)

st.markdown(f'<h1 style="text-align:center;">Use own model</h1>', unsafe_allow_html=True)

class ReuseModel:
    def __init__(_self):
        _self.loaded_model = None
        _self.uploaded_file = None

    # Function to load the trained model
    def load_trained_model(_self, model_filename):
        _self.loaded_model = joblib.load(model_filename)
      
    def use_own_model(_self, data, X, y,l):
        st.write("Upload your trained model (Pickle file)")
        uploaded_model_file = st.file_uploader("Upload your trained model (Pickle file)", type=["pkl"], key=l)
        if uploaded_model_file!=None:
            ttool.load_trained_model(uploaded_model_file)
            # Detect anomalies using the loaded model
            y_pred = ttool.loaded_model.predict(X)
            # Visualize anomalies using a scatter plot
            data["Anomaly"] = np.where(y_pred == 1, True, False)
            dprocess.plot_anomaly(data)
            dprocess.diagnostic_report(y, y_pred)

    def use_own_model_unsup(_self, data,ss):
        st.write("Upload your trained model (Pickle file)")
        # Provide unique key arguments for each file uploader
        uploaded_model_file = st.file_uploader("Upload your trained model (Pickle file)", type=["pkl"],key=ss)
        
        if uploaded_model_file!=None:
            ttool.load_trained_model(uploaded_model_file)
            # Detect anomalies using the loaded model
            loaded_model_results = assign_model(ttool.loaded_model)
            # Visualize anomalies using a scatter plot
            st.subheader("Unsupervised-Visualization of Anomalies")
            dprocess.plot_anomaly(loaded_model_results)

   
ttool=ReuseModel()
def main():
    
    def change_data_state():
        st.session_state.data="done"

    if "data" not in st.session_state:
        st.session_state.data="not done"

    uploaded = st.file_uploader("Upload your CSV data", type=["csv"], on_change=change_data_state)
    opt = st.radio("Choose Dataset Type:",["Labeled", "Unlabeled"])

    if opt=="Labeled" and st.session_state.data=="done" and uploaded is not None:
        data,X,y,_,_, _, _=dprocess.preprocess_sup_data(uploaded)
        st.subheader("Choose the algoritm:")
        if st.checkbox("KNN"):
           ttool.use_own_model(data, X, y,'KNN')
        if st.checkbox("SVC"):
           ttool.use_own_model(data, X, y,'SVC')
        if st.checkbox("ETC"):
           ttool.use_own_model(data, X, y,'ETC')

    if opt=="Unlabeled" and st.session_state.data=="done" and uploaded is not None:
        data=dprocess.preprocess_unsup_data(uploaded)
        st.subheader("Choose the algoritm:")
        if st.checkbox("KNN"):
            ttool.use_own_model_unsup(data,'s')
        if st.checkbox("PCA"):
            ttool.use_own_model_unsup(data,'q')
        if st.checkbox("SVM"):
            ttool.use_own_model_unsup(data,'m')

if __name__ == "__main__":
    main()