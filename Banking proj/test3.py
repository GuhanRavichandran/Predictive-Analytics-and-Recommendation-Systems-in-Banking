import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title(":violet[***Predictive Analytics and Recommendation Systems in Banking***]:bank:")
st.subheader(":orange[Project demonstrates analysis of defaulters and recommend loans]:moneybag:")
import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
background_image_path = r'C:\Users\Hp\OneDrive\Desktop\Banking proj\loan5.jpg'
base64_image = get_base64_of_bin_file(background_image_path)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
df=pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\Banking proj\Data_set.csv")
Employment=df.groupby(['EmploymentType', 'LoanPurpose', 'Default']).size().reset_index(name='Count')
Loan=df.groupby("LoanPurpose")["Default"].value_counts()

tab1,tab2,tab3,tab4=st.tabs(["***Projectinfo***","***Datavisualisation***","***ModelTrained***","***Recommendation***"])
with tab1:
     st.subheader("***A loan is a sum of money that one or more individuals  borrow from banks  so as to financially manage planned or unplanned events. In doing so, the borrower incurs a debt, which he has to pay back with interest and within a given period of time.***")
     st.image(r"C:\Users\Hp\OneDrive\Desktop\Banking proj\loan.jpg", use_column_width=True)
     col1,col2=st.columns(2)
     with col1:
        st.header("***:red[Project Purpose]***")
        st.markdown(":green[Banking is financial Institution helps individuals for loans and deposits]") 
        st.write(":violet[classify Account holders]")
        st.write(":violet[Types of loans]")
        st.write(":violet[Grouping defaulters]")
     with col2:
         st.image(r"C:\Users\Hp\OneDrive\Desktop\Banking proj\loan2.jpeg", use_column_width=True)
with tab2:
    st.subheader("Employment and loan category")
    Emp= st.radio("Select the categories",["Loan_Defaulter","Term & Interest","Credit_score"])
    if Emp=="Loan_Defaulter":
        col1,col2=st.columns(2)
        with col1:
            occupation_counts = df['LoanPurpose'].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(occupation_counts, labels=occupation_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Loan')
            st.pyplot(plt.gcf())
        with col2:
            Employment_counts = df['EmploymentType'].value_counts()
            # Create a pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(Employment_counts, labels=Employment_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Employment')
            st.pyplot(plt.gcf())
            
        st.subheader("Defaulter and distribution")
        col1,col2=st.columns(2)
        with col1:      
            Defaulter = df['Default'].value_counts()
            # Create a pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(Defaulter, labels=Defaulter.index, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Defaulter')
            st.pyplot(plt.gcf())
        with col2:
            plt.figure(figsize=(10, 10))
            sns.barplot(data=df, x='EmploymentType', y='Default', hue='LoanPurpose')

            # Set plot title and labels
            plt.title('Default Rates by Employment Type and Loan Purpose')
            plt.xlabel('Employment Type')
            plt.ylabel('Default')

            # Display the plot in Streamlit
            st.pyplot(plt.gcf())
            
    if Emp=="Term & Interest":
            col1,col2=st.columns(2)
            with col1:
                plt.figure(figsize=(10, 10))
                sns.barplot(data=df, x='LoanPurpose', y='LoanTerm',hue='HasMortgage')

                # Set plot title and labels
                plt.title('Default Rates by Employment Type and Loan Term')
                plt.xlabel('Employment Type')
                plt.ylabel('LoanTerm')

                # Display the plot in Streamlit
                st.pyplot(plt.gcf())
            with col2:
                plt.figure(figsize=(10, 10))
                sns.barplot(data=df, x='LoanPurpose', y='InterestRate',hue='HasCoSigner')

                # Set plot title and labels
                plt.title('Default Rates by Employment Type and InterestRate')
                plt.xlabel('Employment Type')
                plt.ylabel('InterestRate')

                # Display the plot in Streamlit
                st.pyplot(plt.gcf())
                
    if Emp=="Credit_score":
                plt.figure(figsize=(10, 10))
                sns.barplot(data=df, x='EmploymentType', y='CreditScore',hue='Default')

                # Set plot title and labels
                plt.title('Default Rates by Employment Type and CreditScore')
                plt.xlabel('Employment Type')
                plt.ylabel('CreditScore')

                # Display the plot in Streamlit
                st.pyplot(plt.gcf())
        
                


with tab3:
    X = df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'InterestRate', 'LoanTerm']]
    y = df['Default']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the models
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    # Make predictions
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_rf = rf_clf.predict(X_test)

    # Evaluate the models
    st.subheader("Logistic Regression Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
    st.write(f"F1 Score: {classification_report(y_test, y_pred_log_reg, output_dict=True)['1']['f1-score']:.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]):.4f}")
    st.write()

    st.subheader("Random Forest Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
    st.write(f"F1 Score: {classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score']:.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]):.4f}")
        
    data = df

    # Feature selection and preprocessing pipeline
    categorical_features = ['EmploymentType', 'MaritalStatus']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    X = preprocessor.fit_transform(data[categorical_features])

    # Normalize the features, setting with_mean=False for sparse matrices
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means with chosen number of clusters (e.g., k=10)
    k = 10
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X_scaled)

    # Add K-Means cluster labels to the original dataset
    data['Cluster_KMeans'] = clusters_kmeans

    # Evaluate K-Means Clustering
    st.subheader("K-Means Clustering Evaluation Metrics:")
    st.write(f"Silhouette Score: {silhouette_score(X_scaled, clusters_kmeans):.4f}")

    
        
with tab4:
        loancategory=st.radio(":red[Select the loantype]",df["LoanPurpose"].unique())

        if loancategory=='Home':
            st.write(Loan['Home'])
            
        if loancategory=='Education':
            st.write(Loan['Education'])
            
        if loancategory=='Other':
            st.write(Loan['Other'])
            
        if loancategory=='Business':
            st.write(Loan['Business'])
            
        if loancategory=='Auto':
            st.write(Loan['Auto'])
        
        Employmenttype=st.selectbox(":red[Select the Occupation]",df["EmploymentType"].unique())

        if Employmenttype=='Full-time':
            st.write(Employment[Employment['EmploymentType'] == 'Full-time'])
            st.write(":green[***Full-time employee having a highest default on other loans,Education loan and Home loans***]")

        if Employmenttype=='Part-time':
            st.write(Employment[Employment['EmploymentType'] == 'Part-time'])
            st.write(":green[***Part-time employee having a highest default on Business loans,Education loan and Home loans***]")
            
        if Employmenttype=='Self-employed':
            st.write(Employment[Employment['EmploymentType'] == 'Self-employed'])
            st.write(":green[***Self-employed  having a highest default on Business loans,Education loan and Home loans***]")
            
        if Employmenttype=='Unemployed':
            st.write(Employment[Employment['EmploymentType'] == 'Unemployed'])
            st.write(":green[***Full-time employee having a highest default on Auto loans,Business loan and Home loans***]")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        

        

    


