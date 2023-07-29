import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from pycaret.anomaly import *
import plotly.graph_objects as go
import plotly.express as px

class DataProcessingTool:

    # Supervised Data Preprocessing
    def preprocess_sup_data(_self, data):
        data = pd.read_csv(data)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M')
        data.set_index('timestamp', drop=True, inplace=True)

        data['day'] = [i.day for i in data.index]
        data['day_of_year'] = [i.dayofyear for i in data.index]
        data['week_of_year'] = [i.weekofyear for i in data.index]
        data['hour'] = [i.hour for i in data.index]
        data['minute'] = [i.minute for i in data.index]
        data['is_weekday'] = [i.isoweekday() for i in data.index]

        x = data.drop(columns=['label'])
        y = data['label']

        # split the data to training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # Feature Scaling
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)
        # x = scaler.transform(x)

        return data,x,y,x_train,x_test, y_train, y_test
    
    # Unsupervised Data Procees
    def preprocess_unsup_data(_self, data):
        data = pd.read_csv(data)

        # Preprocessing steps for the new data
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M:%S')

        data.set_index('timestamp', drop=True, inplace=True)

        data['day'] = [i.day for i in data.index]
        data['day_of_year'] = [i.dayofyear for i in data.index]
        data['week_of_year'] = [i.weekofyear for i in data.index]
        data['hour'] = [i.hour for i in data.index]
        data['minute'] = [i.minute for i in data.index]
        data['is_weekday'] = [i.isoweekday() for i in data.index]

        setup(data, session_id=123, verbose = False)

        return data
    
    @st.cache_resource(experimental_allow_widgets=True)
    def plot_anomaly(_self, anomaly_result):
        # plot value on y-axis and date on x-axis
        fig = px.line(anomaly_result, x=anomaly_result.index, y="value", title= 'DETECTED OUTLIERS', template='plotly_white')

        # create list of outlier_dates
        outlier_dates = anomaly_result[anomaly_result['Anomaly'] == 1].index

        # obtain y value of anomalies to plot
        y_values = [anomaly_result.loc[i]['value'] for i in outlier_dates]

        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                                 name='Anomaly',
                                 marker=dict(color='red', size=10)))
        st.plotly_chart(fig)

    @st.cache_resource(experimental_allow_widgets=True)
    def diagnostic_report(_self, y, y_pred):
        accuracy = accuracy_score(y, y_pred)
        st.write("Accuracy:", accuracy*100)

        # Print classification report and confusion matrix (optional)
        # Get the classification report and confusion matrix
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-5]:
            row_data = line.strip().split()
            class_name = row_data[0]
            precision, tpr, f1_score, support = [float(val) for val in row_data[1:]]
            report_data.append([class_name, precision, tpr, f1_score])

        # Create a DataFrame to store the extracted data
        data = pd.DataFrame(report_data, columns=['Class', 'Precision', 'True Positive Rate', 'F1-Score'] )
        data.set_index('Class', inplace=True)

        # Display the classification report as a table
        st.markdown("### Evaluation metrics")
        st.dataframe(data)

        # Display confusion matrix as a data frame
        st.markdown("### Confusion Matrix")
        st.dataframe(pd.DataFrame(matrix, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))


    def reset(self):
        self.loaded_model = None
        self.uploaded_file = None
        self.new_data = None
        self.unsupervised_algorithm = None
        self.supervised_algorithm = None
        self.knn = None
        self.pca = None
        self.svm = None
        self.iforest=None
        self.ocsvm=None
        self.Lof=None
        self.show_save_button = False
        self.saved_model_filename = None
        