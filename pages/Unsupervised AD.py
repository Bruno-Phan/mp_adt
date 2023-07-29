import streamlit as st
from pycaret.anomaly import *
import joblib
from io import BytesIO
from preprocess import DataProcessingTool

dprocess=DataProcessingTool()

st.set_page_config(page_title="Anomaly_Detection_Tool", layout="wide", initial_sidebar_state="expanded")

st.markdown(f'<h1 style="text-align:center;">Unsupervised Anomaly Detection</h1>', unsafe_allow_html=True)

class TrainTool:
    def __init__(_self):
        _self.loaded_model = None
        _self.uploaded_file = None
        _self.new_data = None
        _self.unsupervised_algorithm = None
        _self.knn = None
        _self.pca = None
        _self.svm = None
        _self.show_save_button = False
        _self.saved_model_filename = None

        _self.knn_n_neighbors = 5
        _self.knn_fraction = 0.05
        _self.show_knn_params = False

        _self.pca_standardization = False
        _self.pca_fraction = 0.05
        _self.show_pca_params = False

        _self.svm_fraction = 0.05
        _self.svm_kernel = 'rbf'
        _self.svm_gamma = 'auto'
        _self.svm_nu = 0.5
        _self.show_svm_params = False

    # Function to save the trained model
    def save_trained_model(self, model, model_name):
        model_filename = f"{model_name}.pkl"
        joblib.dump(model, model_filename)
        return model_filename 
    

    @st.cache_resource(experimental_allow_widgets=True)
    def knn_train(_self, data, new_fraction, new_n_neighbors):

        # Create the k-Nearest Neighbors (KNN) AD model
        _self.knn = create_model('knn', fraction = new_fraction, n_neighbors = new_n_neighbors)

        # Assign anomaly scores to data points
        knn_results = assign_model(_self.knn)
        st.subheader("Visualization of Anomalies - KNN")
        dprocess.plot_anomaly(knn_results)

        _self.show_save_button = True

        # Save and downlaod model
        dtrain.saved_model_filename = dtrain.save_trained_model(dtrain.knn, "usup_knn_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")
        
        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")

    @st.cache_resource(experimental_allow_widgets=True)
    def pca_train(_self, data, new_fraction, new_standardization):

        # Create the Principal Component Analysis (PCA) AD model
        _self.pca = create_model('pca', fraction = new_fraction, standardization = new_standardization)

        # Assign anomaly scores to data points
        pca_results = assign_model(_self.pca)
        st.subheader("Visualization of Anomalies - PCA")
        dprocess.plot_anomaly(pca_results)

        _self.show_save_button = True

        # Save and downlaod model
        dtrain.saved_model_filename = dtrain.save_trained_model(dtrain.pca, "usup_pca_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")

        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")

    @st.cache_resource(experimental_allow_widgets=True)
    def svm_train(_self, data, new_fraction, new_kernel, new_gamma, new_nu):

        # Create the One-Class Support Vector Machine (SVM) AD model
        _self.svm = create_model('svm', fraction = new_fraction, kernel = new_kernel, gamma = new_gamma, nu = new_nu)

        # Assign anomaly scores to data points
        svm_results = assign_model(_self.svm)
        st.subheader("Visualization of Anomalies - SVM")
        dprocess.plot_anomaly(svm_results)

        _self.show_save_button = True

        # Save and downlaod model
        dtrain.saved_model_filename = dtrain.save_trained_model(dtrain.svm, "usup_svm_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")

        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")

dtrain = TrainTool()

def main():

    def change_data_state():
        st.session_state.data="OK"

    if "data" not in st.session_state:
        st.session_state.data="not OK"

    if "svm_gamma_changed" not in st.session_state:
        st.session_state.svm_gamma_changed = False

    uploaded = st.file_uploader("Upload your CSV data", type=["csv"], on_change=change_data_state)

    if st.session_state.data=="OK":
        data=dprocess.preprocess_unsup_data(uploaded)

        st.subheader("Choose an algorithm:")

        selected_algorithm = st.selectbox('Select Algorithm', ['KNN', 'PCA', 'SVM'])
        
        if selected_algorithm == "KNN":
            dtrain.show_knn_params = st.checkbox("Expert mode for KNN")

            if dtrain.show_knn_params:
                dtrain.knn_n_neighbors = st.slider('n_neighbors', 1, 10, dtrain.knn_n_neighbors)
                dtrain.knn_fraction = st.slider('contamination', 0.01, 0.5, dtrain.knn_fraction)
                                                
                if st.button("Train again"):
                    dtrain.knn_train(data,  dtrain.knn_fraction, dtrain.knn_n_neighbors)
            else:
                dtrain.knn_train(data,  dtrain.knn_fraction, dtrain.knn_n_neighbors)


        if selected_algorithm == "PCA":
            dtrain.show_pca_params = st.checkbox("Expert mode for PCA")

            if dtrain.show_pca_params:
                dtrain.pca_standardization = st.selectbox('Standardization', [False, True], index=0)
                dtrain.pca_fraction = st.slider('contamination', 0.01, 0.5, dtrain.pca_fraction)
                                                
                if st.button("Train again"):
                    dtrain.pca_train(data,  dtrain.pca_fraction, dtrain.pca_standardization)
            else:
                dtrain.pca_train(data,  dtrain.pca_fraction, dtrain.pca_standardization)


        if selected_algorithm == "SVM":
            dtrain.show_svm_params = st.checkbox("Expert mode for SVM")

            if dtrain.show_svm_params:

                dtrain.svm_fraction = st.slider('contamination', 0.01, 0.5, dtrain.svm_fraction)
                dtrain.svm_kernel = st.selectbox('kernel', ['poly', 'rbf', 'sigmoid'], index=['poly', 'rbf', 'sigmoid'].index(dtrain.svm_kernel))
                # dtrain.svm_gamma = st.number_input('gamma', min_value=0.0, step=0.1)
                dtrain.svm_nu = st.slider('nu', 0.01, 1.00, dtrain.svm_nu)
                
                temp_svm_gamma = st.number_input('gamma', value=0.0, min_value=0.0, step=0.1)
        
                # Check if the user has input any value other than 0.0
                if temp_svm_gamma != 0.0:
                    dtrain.svm_gamma = temp_svm_gamma
                    st.session_state.svm_gamma_changed = True
                elif st.session_state.svm_gamma_changed == False:
                    dtrain.svm_gamma = 'auto'
                                                
                if st.button("Train again"):
                    dtrain.svm_train(data,  dtrain.svm_fraction, dtrain.svm_kernel, dtrain.svm_gamma, dtrain.svm_nu)
            else:
                dtrain.svm_train(data,  dtrain.svm_fraction, dtrain.svm_kernel, dtrain.svm_gamma,  dtrain.svm_nu)

if __name__ == "__main__":
    main()