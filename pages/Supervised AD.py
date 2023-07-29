import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from pycaret.anomaly import *
import joblib
from io import BytesIO
from preprocess import DataProcessingTool

dprocess=DataProcessingTool()

st.set_page_config(page_title="Anomaly_Detection_Tool", layout="wide", initial_sidebar_state="expanded")

st.markdown(f'<h1 style="text-align:center;">Supervised Anomaly Detection</h1>', unsafe_allow_html=True)

class TrainTool:
    def __init__(_self):
        _self.loaded_model = None
        _self.uploaded_file = None
        _self.new_data = None
        _self.supervised_algorithm = None
        _self.knn=None
        _self.svc=None
        _self.etc=None
        _self.show_save_button = False
        _self.saved_model_filename = None

        _self.knn_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'n_jobs': -1
        }
        _self.show_knn_params = False

        _self.svc_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': False,
            'tol': 1e-3,
            'max_iter': -1,
        }
        _self.show_svc_params = False

        _self.etc_params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        }
        _self.show_etc_params = False

    # Save the trained model
    def save_trained_sup_model(self, model, model_name):
        model_filename = f"{model_name}.pkl"
        joblib.dump(model, model_filename)
        return model_filename
    
    # Supervised Algs
    @st.cache_resource(experimental_allow_widgets=True)
    def knn_train(_self, data,x, y, x_train, y_train, x_test, y_test):

        # Create KNN model
        _self.knn = KNeighborsClassifier(**_self.knn_params)
        _self.knn.fit(x_train, y_train)

        y_pred = _self.knn.predict(x)
        data["Anomaly"] = np.where(y_pred == 1, True, False)

        # Visualize anomalies
        st.subheader("KNN-Visualization of Anomalies")
        dprocess.plot_anomaly(data)

        # Predict anomalies
        y_pre = _self.knn.predict(x_test)
        dprocess.diagnostic_report(y_test, y_pre)

        _self.show_save_button = True

        # if st.button("Save model"):
        dtrain.saved_model_filename = dtrain.save_trained_sup_model(dtrain.knn, "sup_KNN_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")

        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")
        
    @st.cache_resource(experimental_allow_widgets=True)
    def svc_train(_self, data, x, y, x_train, y_train, x_test, y_test):

        # Create the SVC model
        _self.svc = SVC(**_self.svc_params)
        _self.svc.fit(x_train, y_train)

        y_pred = _self.svc.predict(x)
        data["Anomaly"] = np.where(y_pred == 1, True, False)

        # Visualize anomalies 
        st.subheader("SVC-Visualization of Anomalies")
        dprocess.plot_anomaly(data)

        # Predict anomalies
        y_pre = _self.svc.predict(x_test)
        dprocess.diagnostic_report(y_test, y_pre)
        
        _self.show_save_button = True

        # if st.button("Save model"):
        dtrain.saved_model_filename = dtrain.save_trained_sup_model(dtrain.svc, "sup_svc_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")

        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")

    @st.cache_resource(experimental_allow_widgets=True)
    def etc_train(_self, data, x, y, x_train, y_train, x_test, y_test):

        # Create the Extra Trees Classifier model 
        _self.etc = ExtraTreesClassifier(**_self.etc_params)
        _self.etc.fit(x_train, y_train)

        # Model Evaluation
        # Predict anomalies
        y_pred = _self.etc.predict(x)
        data["Anomaly"] = np.where(y_pred == 1, True, False)
        # Visualize anomalies
        st.subheader("ETC-Visualization of Anomalies")
        dprocess.plot_anomaly(data)

        # Predict anomalies 
        y_pre = _self.etc.predict(x_test)
        dprocess.diagnostic_report(y_test, y_pre)

        _self.show_save_button=True

        # if st.button("Save model"):
        dtrain.saved_model_filename = dtrain.save_trained_sup_model(dtrain.etc, "sup_etc_model")
        st.success(f"Model saved as '{dtrain.saved_model_filename}'")
        # Provide download button for the saved model
        with open(dtrain.saved_model_filename, "rb") as f:
            model_bytes = BytesIO(f.read())
        st.download_button("Download the saved model", model_bytes, file_name=dtrain.saved_model_filename, mime="application/octet-stream")

dtrain = TrainTool()

def main():

    def change_data_state():
        st.session_state.data="OK SUP"
    if "data" not in st.session_state:
        st.session_state.data="not OK SUP"

    uploaded = st.file_uploader("Upload your CSV data", type=["csv"], on_change=change_data_state)


    if st.session_state.data=="OK SUP":
        data, x, y, x_train, x_test, y_train, y_test=dprocess.preprocess_sup_data(uploaded)

        st.subheader("Choose an algorithm:")

        selected_algorithm = st.selectbox('Select Algorithm', ['KNN', 'ETC', 'SVC'])


        # KNN
        if selected_algorithm == 'KNN':
            dtrain.show_knn_params = st.checkbox("Expert mode for KNN")
           
            if dtrain.show_knn_params:
                dtrain.knn_params['n_neighbors'] = st.slider('n_neighbors', 1, 10, dtrain.knn_params['n_neighbors'])
                dtrain.knn_params['weights'] = st.selectbox('weights', ['uniform', 'distance'], index=['uniform', 'distance'].index(dtrain.knn_params['weights']))
                dtrain.knn_params['algorithm'] = st.selectbox('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(dtrain.knn_params['algorithm']))
                dtrain.knn_params['leaf_size'] = st.slider('leaf_size', 1, 50, dtrain.knn_params['leaf_size'])
                dtrain.knn_params['p'] = st.slider('p (Power parameter for the metric)', 1, 5, dtrain.knn_params['p'])
                dtrain.knn_params['metric'] = st.selectbox('metric', ['minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean'], index=['minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean'].index(dtrain.knn_params['metric']))
                dtrain.knn_params['n_jobs'] = st.number_input('n_jobs (number of parallel jobs)', -1, 100, dtrain.knn_params['n_jobs'])
                
                if st.button("Train and evaluate again"):
                    dtrain.knn_train(data, x, y, x_train, y_train, x_test ,y_test)
            else:
                dtrain.knn_train(data, x, y, x_train, y_train, x_test ,y_test)
        
        # SVC
        elif selected_algorithm == 'ETC':
            dtrain.show_svc_params = st.checkbox("Expert mode for SVC")
            
            if dtrain.show_svc_params:

                dtrain.svc_params['C'] = st.number_input('C', 0.1, 10.0, dtrain.svc_params['C'])
                dtrain.svc_params['degree'] = st.number_input('degree', 1, 10, dtrain.svc_params['degree'])
                dtrain.svc_params['gamma'] = st.text_input('gamma', dtrain.svc_params['gamma'])
                dtrain.svc_params['coef0'] = st.number_input('coef0', 0.0, 1.0, dtrain.svc_params['coef0'])
                dtrain.svc_params['tol'] = st.number_input('tol', 1e-5, 1e-1, dtrain.svc_params['tol'])
                dtrain.svc_params['max_iter'] = st.number_input('max_iter', -1, 10000, dtrain.svc_params['max_iter'])

                if st.button("Train and evaluate again"):
                    dtrain.svc_train(data, x, y, x_train, y_train, x_test ,y_test)
            else:
                dtrain.svc_train(data, x, y, x_train, y_train, x_test ,y_test)
        
        # ETC
        elif selected_algorithm == 'SVC':
           dtrain.show_etc_params = st.checkbox("Expert mode ETC")
           
           if dtrain.show_etc_params:
               dtrain.etc_params['n_estimators'] = st.slider('n_estimators', 10, 200, dtrain.etc_params['n_estimators'])
               dtrain.etc_params['criterion'] = st.selectbox('criterion', ['gini', 'entropy'], index=['gini', 'entropy'].index(dtrain.etc_params['criterion']))
               dtrain.etc_params['max_depth'] = st.number_input('max_depth', None, 100, dtrain.etc_params['max_depth'] if dtrain.etc_params['max_depth'] else 50)
               dtrain.etc_params['min_samples_split'] = st.slider('min_samples_split', 2, 10, dtrain.etc_params['min_samples_split'])
               dtrain.etc_params['min_samples_leaf'] = st.slider('min_samples_leaf', 1, 10, dtrain.etc_params['min_samples_leaf'])
               dtrain.etc_params['max_features'] = st.selectbox('max_features', ['auto', 'sqrt', 'log2'], index=['auto', 'sqrt', 'log2'].index(dtrain.etc_params['max_features']))
               
               if st.button("Train and evaluate again"):
                   dtrain.etc_train(data, x, y, x_train, y_train, x_test ,y_test)
           else:
               dtrain.etc_train(data, x, y, x_train, y_train, x_test ,y_test)

if __name__ == "__main__":
    main()