# Predictive-Analytics-and-Recommendation-Systems-in-Banking

***A bank loan is a debt that a person, better known as the borrower, owes to a bank. It's basically an agreement between the borrower and the bank about a certain amount of money that the borrower will borrow and then pay back in specific increments at a specific interest rate.***

**Installation**
To run this project, you need to install the following packages:
```python
pip install pandas
pip install seaborn
pip install sklearn

Kindly import below library
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

