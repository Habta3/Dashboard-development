#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px
from Fetch_data import db_execute_fetch

st.set_page_config(page_title="Day 5", layout="wide")
import mysql.connector
import mysql

connection = mysql.connector.connect(host='localhost',
                                         database='Telecom',
                                         user='root',
                                         password='1234') 
import pickle

import pandas as pd
import streamlit as st
from PIL import Image


def load_model():
    pickle_in = open('E:\\10xAccademy_Practice\\Week 1\\User Experiance Analysis\\K-Means Model\\UserExperiance_K-Means_Model.pkl', 'rb')
    satisfaction_model = pickle.load(pickle_in)
    return satisfaction_model



def Predict_Experiance():
    st.title("Prdict User Experiance")

    eng_score1 = st.slider("avg_rtt_total Score", min_value=0.0,
                          max_value=100.0, step=10.0)
    eng_score2 = st.slider("avg_bearer_tp_total score", min_value=0.0,
                          max_value=100.0, step=10.0)
    eng_score3 = st.slider("avg_tcp_retrans_total", min_value=0.0,
                          max_value=100.0, step=10.0)
    eng_score4 = st.slider("avg_tp_total", min_value=0.0,
                          max_value=100.0, step=10.0)
    eng_score5 = st.slider("total_avg_tcp_tota", min_value=0.0,
                          max_value=100.0, step=10.0)
    

    if st.button("Predict"):
        model = load_model()
        result = model.fit_predict([[eng_score1,eng_score2,eng_score3,eng_score4,eng_score5]])

        st.success(f"The customer satisfaction is {result[0][1][2][3][4],result[0][1][2][3][4],result[0][1][2][3][4],result[0][1][2][3][4]}")

Predict_Experiance()
import os
# from matplotlib.pyplot import plt
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

#sys.path.append(os.path.abspath(os.path.join('./scripts')))


def overview_app():
    

    st.header("Top 10 handsets used by customers")
    top_df = pd.read_csv('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\data\\top_10_handset.csv')

    fig = px.bar(top_df, x='handset_type', y='count', height=500)
    st.plotly_chart(fig)

    st.header("Top 3 handsets Manufacturers")
    top_3_df = pd.read_csv('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\data\\top_3_manuf.csv')
    fig = px.bar(top_3_df, x='handset_manufacturer', y='count', height=500)
    st.plotly_chart(fig)

    st.header("Top 5 handsets type manufactured by apple")
    top_5_app = pd.read_csv('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\data\\top_5_apple.csv')
    fig = px.bar(top_5_app, x='Handset', y='count', height=500)
    st.plotly_chart(fig)

    st.header("User's with most sessions")
    top_5_session = pd.read_csv('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\data\\top_5_session.csv', nrows=5)

    # print(top_5_session)
    st.write(top_5_session)

    st.header("Duration Distribution")
    image = Image.open('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\assets\\Durationdist.png')
    st.image(image, caption="Duration Distribution", use_column_width=True)

    st.header("Top data usage per applications")
    image = Image.open('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\assets\\top_data_usage.png')
    st.image(image, caption="Applications Data usage", use_column_width=True)

    st.header("Application Duration distribution using deciles")
    image = Image.open('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\assets\\DurationDeciles.png')
    st.image(image, caption="Applications Duration Distribution",
             use_column_width=True)

    st.header("Clustering users based on their Engagement score")
    image = Image.open('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\assets\\userEngagCluster.png')
    st.image(image, caption="Users clustering into 3 groups based on Engagement score",
             use_column_width=True)

    st.header("TCP retransmissions")
    image = Image.open('./assets/tcpretransmission.png')
    st.image(image, caption="Top TCP retransmissions",
             use_column_width=True)

    st.header("Top throughputs")
    image = Image.open('./assets/TopTP.png')
    st.image(image, caption="Top 10 througputs",
             use_column_width=True)

    st.header("Experience Distribution")
    image = Image.open('C:\\Users\\HB\\Downloads\\Telecom-Data-Analysis-main\\Telecom-Data-Analysis-main\\assets//ClusterDist.png')
    st.image(image, caption="Experience distribution of user",
             use_column_width=True)





overview_app()










def DBConnect(dbName=None):
   
    conn = mysql.connector.connect(host='localhost', user='root', password="1234",
                         database=dbName, buffered=True)
    cur = conn.cursor()
    return conn, cur
def db_execute_fetch(*args, many=False, tablename='', rdf=True, **kwargs) -> pd.DataFrame:
    """

    Parameters
    ----------
    *args :

    many :
         (Default value = False)
    tablename :
         (Default value = '')
    rdf :
         (Default value = True)
    **kwargs :


    Returns
    -------

    """
    connection, cursor1 = DBConnect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if tablename:
        print(f"{nrow} recrods fetched from {tablename} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res

def loadData():
    query = "select * from satisfactionData2"
    df = db_execute_fetch(query, dbName="Telecom", rdf=True)
    return df

def selectHashTag():
    df = loadData()
    hashTags = st.multiselect("choose combaniation of hashtags", list(df['hashtags'].unique()))
    if hashTags:
        df = df[np.isin(df, hashTags).any(axis=1)]
        st.write(df)

def selectLocAndAuth():
    df = loadData()
    location = st.multiselect("choose Location of tweets", list(df['place_coordinate'].unique()))
    lang = st.multiselect("choose Language of tweets", list(df['language'].unique()))

    if location and not lang:
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    elif lang and not location:
        df = df[np.isin(df, lang).any(axis=1)]
        st.write(df)
    elif lang and location:
        location.extend(lang)
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    else:
        st.write(df)

def barChart(data, title, X, Y):
    title = title.title()
    st.title(f'{title} Chart')
    msgChart = (alt.Chart(data).mark_bar().encode(alt.X(f"{X}:N", sort=alt.EncodingSortField(field=f"{Y}", op="values",
                order='ascending')), y=f"{Y}:Q"))
    st.altair_chart(msgChart, use_container_width=True)

def wordCloud():
    df = loadData()
    cleanText = ''
    for text in df['clean_text']:
        tokens = str(text).lower().split()

        cleanText += " ".join(tokens) + " "

    wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
    st.title("Tweet Text Word Cloud")
    st.image(wc.to_array())

def stBarChart():
    df = loadData()
    dfCount = pd.DataFrame({'Tweet_count': df.groupby(['original_author'])['clean_text'].count()}).reset_index()
    dfCount["original_author"] = dfCount["original_author"].astype(str)
    dfCount = dfCount.sort_values("Tweet_count", ascending=False)

    num = st.slider("Select number of Rankings", 0, 50, 5)
    title = f"Top {num} Ranking By Number of tweets"
    barChart(dfCount.head(num), title, "original_author", "Tweet_count")


def langPie():
    df = loadData()
    dfLangCount = pd.DataFrame({'Tweet_count': df.groupby(['language'])['clean_text'].count()}).reset_index()
    dfLangCount["language"] = dfLangCount["language"].astype(str)
    dfLangCount = dfLangCount.sort_values("Tweet_count", ascending=False)
    dfLangCount.loc[dfLangCount['Tweet_count'] < 10, 'lang'] = 'Other languages'
    st.title(" Tweets Language pie chart")
    fig = px.pie(dfLangCount, values='Tweet_count', names='language', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')

    colB1, colB2 = st.beta_columns([2.5, 1])

    with colB1:
        st.plotly_chart(fig)
    with colB2:
        st.write(dfLangCount)


st.title("Data Display")
selectHashTag()
st.markdown("<p style='padding:10px; background-color:#000000;color:#00ECB9;font-size:16px;border-radius:10px;'>Section Break</p>", unsafe_allow_html=True)
selectLocAndAuth()
st.title("Data Visualizations")
wordCloud()
with st.beta_expander("Show More Graphs"):
    stBarChart()
    langPie()


# In[2]:





# In[ ]:







# In[ ]:




