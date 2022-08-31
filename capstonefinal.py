from random import Random
from unittest import load_tests

import pandas as pd
import streamlit as st
import hydralit_components as hc
from plotly.graph_objs import *
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier


   

#df = pd.read_csv("C:\\Users\\USER\\Downloads\\finaloriginal.csv")
#df.drop[["CUSTOMER_TRX_LINE_ID","BILL_TO_USTOMER_ID","SHIP_TO_CUSTOMER_ID","SHIP_SITE_USE_ID","CURRENCY CODE","EXCHANGE_RATE","LINE_TYPE","PRICE_ADJUSTMENT_ID","MODIFIER_ID","LINE_ATT2","PARENT_ORDER_TYPE_ID","PARENT_OE_HEADER_ID","PARENT_OE_LINE_ID","OE_LINE_ID","PARENT_TRANSACTION_SOURCE_ID","TRANSACTION_SOURCE_ID","SHIPMENT_DATE"]]

st.set_page_config(layout="wide",page_title=None)

#uploaded_file = st.sidebar.file_uploader(label="Upload your data",type=["csv"])

#global df
#if uploaded_file is not None:
#dff=pd.read_csv(uploaded_file)
 

    
#dff=pd.read_csv("CosmalineTransactions.csv")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
     # To read file as bytes:
      bytes_data = uploaded_file.getvalue()
      st.write(bytes_data)
     

dff = pd.read_csv(uploaded_file)





#dff = pd.read_csv("C:\\Users\\USER\\Downloads\\CosmalineTransactions.csv")
dff.drop(["CUSTOMER_TRX_LINE_ID","BILL_TO_CUSTOMER_ID","SHIP_TO_CUSTOMER_ID","SHIP_SITE_USE_ID","CURRENCY_CODE","EXCHANGE_RATE","LINE_TYPE","PRICE_ADJUSTMENT_ID","MODIFIER_ID","LINE_ATT2","PARENT_ORDER_TYPE_ID","PARENT_OE_HEADER_ID","PARENT_OE_LINE_ID","OE_LINE_ID","PARENT_TRANSACTION_SOURCE_ID","TRANSACTION_SOURCE_ID","SHIPMENT_DATE"],axis=1)

over_theme = {'menu_background': '#000080'}
menu_data = [{'label':'Sales Prediction'},{'label':'Next Purchase Prediction'},{'label':'Market Basket Analysis'},{'label':'Price Sensitivity Analysis'},{'label':'RFM Analysis'},{'label':'Recommendation System'},{'label':'Google Trends'},{'label':'Competition'}]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky',override_theme=over_theme)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import base64
import time
from streamlit_lottie import st_lottie
import requests
import plotly
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import datetime as dt


def load_lottieurl(url:str):
           r = requests.get(url)
           if r.status_code !=200:
             return None
           return r.json()


if menu_id == "Next Purchase Prediction":
    import datetime

    dff["TRX_DATE"] = pd.to_datetime(dff["TRX_DATE"])
    
    ctm_bhvr_dt = dff.loc[(dff['TRX_DATE'] >= '2021-01-01')
                     & (dff['TRX_DATE'] < '2021-07-20')]
    
    
    ctm_next_quarter =  dff.loc[(dff['TRX_DATE'] >= '2021-07-20')
                     & (dff['TRX_DATE'] < '2021-10-15')]

    ctm_dt = pd.DataFrame(ctm_bhvr_dt['CUSTOMER_ID'].unique())

    ctm_dt.columns = ['CUSTOMER_ID']
    ctm_1st_purchase_in_next_quarter = ctm_next_quarter.groupby('CUSTOMER_ID').TRX_DATE.min().reset_index()
    ctm_1st_purchase_in_next_quarter.columns = ['CUSTOMER_ID', 'MinPurchaseDate']

    ctm_last_purchase_bhvr_dt = ctm_bhvr_dt.groupby('CUSTOMER_ID').TRX_DATE.max().reset_index()
    ctm_last_purchase_bhvr_dt.columns = ['CUSTOMER_ID', 'MaxPurchaseDate']

    ctm_purchase_dates = pd.merge(ctm_last_purchase_bhvr_dt, ctm_1st_purchase_in_next_quarter, on='CUSTOMER_ID', how='left')

    ctm_purchase_dates['NextPurchaseDay'] = (ctm_purchase_dates['MinPurchaseDate'] - ctm_purchase_dates['MaxPurchaseDate']).dt.days

    ctm_dt = pd.merge(ctm_dt, ctm_purchase_dates[['CUSTOMER_ID', 'NextPurchaseDay']], on='CUSTOMER_ID', how='left')
    ctm_dt = ctm_dt.fillna(9999)

    ctm_max_purchase = ctm_bhvr_dt.groupby('CUSTOMER_ID').TRX_DATE.max().reset_index()
    ctm_max_purchase.columns = ['CUSTOMER_ID','MaxPurchaseDate']

    # Find the recency of each customer in days
    ctm_max_purchase['Recency'] = (ctm_max_purchase['MaxPurchaseDate'].max() - ctm_max_purchase['MaxPurchaseDate']).dt.days

   # Merge the dataframes ctm_dt and ctm_max_purchase[['CustomerID', 'Recency']] on the CustomerID column.
    ctm_dt = pd.merge(ctm_dt, ctm_max_purchase[['CUSTOMER_ID', 'Recency']], on='CUSTOMER_ID')   

    def order_cluster(df, target_field_name, cluster_field_name, ascending):
        new_cluster_field_name = "new_" + cluster_field_name
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        df_new["index"] = df_new.index
        df_final = pd.merge(df, df_new[[cluster_field_name, "index"]], on=cluster_field_name)
    
        df_final = df_final.drop([cluster_field_name], axis=1)
    
        df_final = df_final.rename(columns={"index": cluster_field_name})
    
        return df_final
   
   
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt[['Recency']])
    ctm_dt['RecencyCluster'] = kmeans.predict(ctm_dt[['Recency']])
    ctm_dt = order_cluster(ctm_dt, 'Recency', 'RecencyCluster', False)

    
    
    
    ctm_frequency = dff.groupby('CUSTOMER_ID').TRX_DATE.count().reset_index()
    ctm_frequency.columns = ['CUSTOMER_ID', 'Frequency']
    ctm_dt = pd.merge(ctm_dt, ctm_frequency, on='CUSTOMER_ID')
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt[['Frequency']])
    ctm_dt['FrequencyCluster'] = kmeans.predict(ctm_dt[['Frequency']])
    ctm_dt = order_cluster(ctm_dt, 'Frequency', 'FrequencyCluster', False)

    ctm_revenue = dff.groupby('CUSTOMER_ID').AR_AMOUNT_ALL.sum().reset_index()
    ctm_dt = pd.merge(ctm_dt, ctm_revenue, on='CUSTOMER_ID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt[['AR_AMOUNT_ALL']])
    ctm_dt['RevenueCluster'] = kmeans.predict(ctm_dt[['AR_AMOUNT_ALL']])
    ctm_dt = order_cluster(ctm_dt, 'AR_AMOUNT_ALL', 'RevenueCluster', True)
    
    ctm_dt['OverallScore'] = ctm_dt['RecencyCluster'] + ctm_dt['FrequencyCluster'] + ctm_dt['RevenueCluster']

    ctm_dt.groupby('OverallScore')['Recency', 'Frequency', 'AR_AMOUNT_ALL'].mean()

    ctm_dt['Segment'] = 'Low-Value'

    ctm_dt.loc[ctm_dt['OverallScore'] > 3, 'Segment'] = 'Mid-Value'

    ctm_dt.loc[ctm_dt['OverallScore'] > 5, 'Segment'] = 'High-Value'
    
    ctm_class = ctm_dt.copy()

    ctm_class = pd.get_dummies(ctm_class)

    ctm_class['NextPurchaseDayRange'] = 1 

    ctm_class.loc[ctm_class.NextPurchaseDay>85, 'NextPurchaseDayRange'] = 0

    ctm_class = ctm_class.drop('NextPurchaseDay', axis=1)
    X, y = ctm_class.drop('NextPurchaseDayRange', axis=1), ctm_class.NextPurchaseDayRange

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)
    import xgboost as xgb
    rfc = RandomForestClassifier()
    XG = xgb.XGBClassifier()
    k = KNeighborsClassifier()
    lr = LogisticRegression()
    XG.fit(X_train,y_train)
    y_pred = XG.predict(X_test)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


    
    
    dff = dff.sort_values(by="TRX_DATE",ascending=True)
    tod = dff["TRX_DATE"].max()
    d = datetime.timedelta(days = 213)
    a = tod - d

    last_6 = dff.loc[(dff['TRX_DATE'] >= a)
                     & (dff['TRX_DATE'] < tod)]
    
    ctm_dt1 = pd.DataFrame(last_6['CUSTOMER_ID'].unique())

    ctm_dt1.columns = ['CUSTOMER_ID']
    
    ctm_last_purchase_bhvr_dt1 = last_6.groupby('CUSTOMER_ID').TRX_DATE.max().reset_index()
    ctm_last_purchase_bhvr_dt1.columns = ['CUSTOMER_ID', 'MaxPurchaseDate']

    
    ctm_max_purchase1 = last_6.groupby('CUSTOMER_ID').TRX_DATE.max().reset_index()
    ctm_max_purchase1.columns = ['CUSTOMER_ID','MaxPurchaseDate']

    # Find the recency of each customer in days
    ctm_max_purchase1['Recency'] = (ctm_max_purchase1['MaxPurchaseDate'].max() - ctm_max_purchase1['MaxPurchaseDate']).dt.days

   # Merge the dataframes ctm_dt and ctm_max_purchase[['CustomerID', 'Recency']] on the CustomerID column.
    ctm_dt1 = pd.merge(ctm_dt1, ctm_max_purchase1[['CUSTOMER_ID', 'Recency']], on='CUSTOMER_ID') 
    


    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt1[['Recency']])
    ctm_dt1['RecencyCluster'] = kmeans.predict(ctm_dt1[['Recency']])
    ctm_dt1 = order_cluster(ctm_dt1, 'Recency', 'RecencyCluster', False)

    
    
    ctm_frequency1 = dff.groupby('CUSTOMER_ID').TRX_DATE.count().reset_index()
    ctm_frequency1.columns = ['CUSTOMER_ID', 'Frequency']
    ctm_dt1 = pd.merge(ctm_dt1, ctm_frequency1, on='CUSTOMER_ID')
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt1[['Frequency']])
    ctm_dt1['FrequencyCluster'] = kmeans.predict(ctm_dt1[['Frequency']])
    ctm_dt1 = order_cluster(ctm_dt1, 'Frequency', 'FrequencyCluster', False)

    ctm_revenue1 = dff.groupby('CUSTOMER_ID').AR_AMOUNT_ALL.sum().reset_index()
    ctm_dt1 = pd.merge(ctm_dt1, ctm_revenue1, on='CUSTOMER_ID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(ctm_dt1[['AR_AMOUNT_ALL']])
    ctm_dt1['RevenueCluster'] = kmeans.predict(ctm_dt1[['AR_AMOUNT_ALL']])
    ctm_dt1 = order_cluster(ctm_dt1, 'AR_AMOUNT_ALL', 'RevenueCluster', True)


    ctm_dt1['OverallScore'] = ctm_dt1['RecencyCluster'] + ctm_dt1['FrequencyCluster'] + ctm_dt1['RevenueCluster']

# Get mean of the components of OverallScore
    ctm_dt1.groupby('OverallScore')['Recency', 'Frequency', 'AR_AMOUNT_ALL'].mean()

    ctm_dt1['Segment'] = 'Low-Value'

    ctm_dt1.loc[ctm_dt['OverallScore'] > 3, 'Segment'] = 'Mid-Value'
    ctm_dt1.loc[ctm_dt['OverallScore'] > 5, 'Segment'] = 'High-Value'

    ctm_class1 = ctm_dt1.copy()

# Convert categorical data to numerical data using get_dummies
    ctm_class1 = pd.get_dummies(ctm_class1)
    

    predictions = XG.predict(ctm_class1)
    ctm_class1["pred"] = predictions

    only_0 = ctm_class1.loc[ctm_class1["pred"]==0]
    total_0 = len(only_0)

    
    only_1 = ctm_class1.loc[ctm_class1["pred"]==1]
    total_1 = len(only_1)

    col1,col2 = st.columns([4,1])
    with col1:
        st.subheader("Next Purchase Prediction")
        st.markdown("The following tool helps in estimating the number of customers that will perform another purchase in the coming 3 months to manage sales expectations.It can also estimate the number of customers that might churn.")
             
    with col2:
        anim = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jzaeejb5.json")
        st_lottie(anim, height=150,width=300)
    st.markdown("----")


    col3,col4,col5 = st.columns([3,1,3])

    with col3:
            
            st.subheader("Returning Customers")
            st.markdown("The customers that the model predicted to perform a purchase in the coming 3 months")
            theme_mon = {'bgcolor':'#90EE90 ','content_color':'white','progress_color':'white','title_color':'white','icon':'bi bi-check','icon_color':'white','font_size':'100',}
            hc.info_card(title="Total Returning Customers",content =total_1, theme_override=theme_mon)

            total_results = total_1 + total_0
            percentage_0 = round((total_0/ total_results) *100)       
            percentage_1 = round((total_1/  total_results) *100)

            percentage_0 = "{:.2f}".format(percentage_0)
            
            percentage_1 = "{:.2f}".format(percentage_1)
            
            values = [percentage_1,percentage_0]
            label = ["Returning",""]
            colors = ["Green","Grey"]
            fig1 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
            fig1.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
            fig1.update_layout(
              title="Percentage of Total Customers",
              width=280,
             height = 300,
        
             showlegend=False,
    
            annotations=[dict(text=percentage_1, x=0.5, y=0.5, font_size=20, showarrow=False)])
            fig1
            
            with st.expander("More Details"):
                st.markdown("Average Score")
                st.subheader("{:.2f}".format((only_1["OverallScore"].mean())))

                st.markdown("Average Recency")
                st.subheader("{:.2f}".format(only_1["Recency"].mean()))

                st.markdown("Average Frequency")
                st.subheader("{:.2f}".format(only_1["Frequency"].mean()))

                st.markdown("Average Monetary")
                st.subheader("{:.2f}".format(only_1["AR_AMOUNT_ALL"].mean()))
        
    st.markdown("----")
    st.subheader("Customer-Based Results")
        
    customer_select = st.selectbox("Choose Customer",ctm_class1["CUSTOMER_ID"].unique().tolist())
    ctm_class2 = ctm_class1.loc[(ctm_class1["CUSTOMER_ID"]==customer_select)]
    col6,col7,col8,col9,col10 = st.columns([2,2,2,2,2])
    with col6:
        st.markdown("Predicted as:")
        if ctm_class2.iloc[0,11] ==0:
            st.subheader("Churning Customer")

        if ctm_class2.iloc[0,11] ==1:
            st.subheader("Returning Customer")

    
    with col7:
        st.markdown("Recency")
        st.subheader("{:.2f}".format(ctm_class2.iloc[0,1]))

    
    with col8:
        st.markdown("Frequency")
        st.subheader("{:.2f}".format(ctm_class2.iloc[0,3]))


    
    with col9:
        st.markdown("Monetary")
        st.subheader("{:.2f}".format(ctm_class2.iloc[0,5]))
    
    
    with col10:
        st.markdown("Overall Score")
        st.subheader("{:.2f}".format(ctm_class2.iloc[0,7]))



        
        

        
        
                


                
                


            

    with col5:
            st.subheader("Churning Customers")
            st.markdown("The customers that the model predicted to restrain from purchasing in the coming 3 months")
            theme_mon = {'bgcolor':'#E6676B ','content_color':'white','progress_color':'white','title_color':'white','icon':'bi bi-x-square','icon_color':'white','font_size':'100',}
            hc.info_card(title="Total Churned Customers",content =total_0, theme_override=theme_mon)

            
            values = [percentage_0,percentage_1]
            label = ["Churning",""]
            colors = ["red","Grey"]
            fig2 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
            fig2.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
            fig2.update_layout(
              title="Percentage of Total Customers",
              width=280,
             height = 300,
        
             showlegend=False,
    
            annotations=[dict(text=percentage_0, x=0.5, y=0.5, font_size=20, showarrow=False)])
            fig2
            
            with st.expander("More Details"):
                st.markdown("Average Score")
                
                st.subheader("{:.2f}".format((only_0["OverallScore"].mean())))

                st.markdown("Average Recency")
                st.subheader("{:.2f}".format(only_0["Recency"].mean()))

                st.markdown("Average Frequency")
                st.subheader("{:.2f}".format(only_0["Frequency"].mean()))

                st.markdown("Average Monetary")
                st.subheader("{:.2f}".format(only_0["AR_AMOUNT_ALL"].mean()))
            



        
    

        
        
            
    


    
            



    
        
    
     




    


if menu_id == "Sales Prediction":
    col1,col2 = st.columns([4,1])
    col3,col4 = st.columns([3,1])
    col5,col6 = st.columns([3,1])
    with col1:
        st.subheader("Sales Forecasting")
        st.markdown("The following page acts as a Sales forecasting tool in timeseries. After numereous trials, the following model was chosen to be trained on the input data,and use it to forecast sales for the coming 4 days.")
    
    with col2:
        anim = load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_7gw8FJ.json")
        st_lottie(anim, height=150,width=300)
    

    with col3:
        
        dff["TRX_DATE"] = pd.to_datetime(dff["TRX_DATE"])
        df1 = dff[["TRX_DATE","AR_AMOUNT_ALL"]]
        df1['TRX_DATE'] = pd.to_datetime(df1['TRX_DATE'])

        weekly_data = df1.resample('W', on='TRX_DATE')['AR_AMOUNT_ALL'].sum()
        weekly_data = pd.DataFrame(weekly_data)
        weekly_data.reset_index(inplace=True)
        weekly_plot = [ 
        
        go.Scatter( 
        x=weekly_data.index, 
        y=weekly_data['AR_AMOUNT_ALL'], 
        name='actual',
        line=dict(color="#000080") 
    )  
     
] 
        plot_layout = go.Layout( 
        title='Weekly Trend of Sales',
        paper_bgcolor= 'rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)') 
    
        fig_weekly = go.Figure(data=weekly_plot, layout=plot_layout) 
        st.plotly_chart(fig_weekly)


        week_sales = weekly_data.sort_values(by="AR_AMOUNT_ALL",ascending=False)
        highest_week = week_sales.iloc[0,0]
        
    with col4:
      
        weekly_data = weekly_data.sort_values(by="AR_AMOUNT_ALL",ascending=False)
        weekly_data['TRX_DATE'] = weekly_data['TRX_DATE'].dt.date
        
        st.markdown("Starting Date of week with highest sales")
        st.subheader(weekly_data.iloc[0,0])
        st.markdown("Sales")
        highest_date = "{:.2f}".format(weekly_data.iloc[0,1])
        st.subheader(f'{str(highest_date)} L.L')
        with st.expander("Full data"):
            st.dataframe(weekly_data)


        


    with col5:    
        daily_data = df1.resample('D', on='TRX_DATE')['AR_AMOUNT_ALL'].sum()
        daily_data = pd.DataFrame(daily_data)
        daily_data.reset_index(inplace=True)
        daily_data["TRX_DATE"] = pd.to_datetime(daily_data["TRX_DATE"])
        daily_data["Day_of_week"] = daily_data["TRX_DATE"].dt.strftime('%A')
        daily_data["Num_of_day"] = daily_data["TRX_DATE"].dt.dayofweek
        

        
        daily_plot = [ 
        
        go.Scatter( 
        x=daily_data.index, 
        y=daily_data['AR_AMOUNT_ALL'], 
        name='actual',
        line=dict(color="#000080") 
    )  
     
] 
        plot_layout = go.Layout( 
        title='Daily Trend of Sales',
        paper_bgcolor= 'rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)') 
    
        fig_daily = go.Figure(data=daily_plot, layout=plot_layout) 
        st.plotly_chart(fig_daily)
    
    with col6:
        daily_data = daily_data.sort_values(by="AR_AMOUNT_ALL",ascending=False)
        daily_data['TRX_DATE'] = daily_data['TRX_DATE'].dt.date
       
        st.markdown("Day with highest sales")
        st.subheader(daily_data.iloc[0,0])
        st.markdown("Sales")
        highest_date_day = "{:.2f}".format(daily_data.iloc[0,1])
        st.subheader(f'{str(highest_date_day)} L.L')
        with st.expander("Full data"):
            st.dataframe(daily_data)
    with st.expander("Days of Week Sales"):
         for_weekday = daily_data.groupby(["Day_of_week","Num_of_day"],as_index=False)["AR_AMOUNT_ALL"].sum()
         for_weekday = for_weekday.sort_values(by="Num_of_day",ascending=True)
         fig7 = px.bar(for_weekday,x="Day_of_week",y="AR_AMOUNT_ALL",color_discrete_sequence =['navy']*len(for_weekday))
         plot_layout = go.Layout( 
         paper_bgcolor= 'rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)') 
    
         st.plotly_chart(fig7)

    st.markdown("----")


    col7,col8 = st.columns([3,2])
    
    dff["AR_AMOUNT_ALL"].abs()
    
    data = df1.resample('D', on='TRX_DATE')['AR_AMOUNT_ALL'].sum()
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    
    data2 = data.iloc[0:130,]
    data3 = data.tail(369)
    y = data3['AR_AMOUNT_ALL'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

# generate the input and output sequences
    n_lookback = 25 # length of input sequences (lookback period)
    n_forecast =  3# length of output sequences (forecast period)
    n_lookback = np.array(n_lookback)
    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
      X.append(y[i - n_lookback: i])
      Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

# fit the model

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))
   

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=-1)
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)
    st.markdown(Y_)
    df_past = data[['AR_AMOUNT_ALL']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'AR_AMOUNT_ALL'}, inplace=True)
    df_past['Date'] = pd.to_datetime(data['TRX_DATE'])
    df_past['Forecast'] = np.nan

    df_past['Forecast'].iloc[-1] = df_past['AR_AMOUNT_ALL'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    
   
    results = df_past.append(df_future).set_index('Date')
    col8,col9 = st.columns([3,2])
    with col7:
        st.subheader("Future Predictions")
        st.markdown("Please not that the blue line represents historical data, while the red line represents the predicted sales value for the coming 4 days")
    with col8:
        plot_data = [ 
            go.Scatter( 
            x=results.index, 
            y=results['AR_AMOUNT_ALL'], 
            name='actual' 
    ), 
        go.Scatter( 
        x=results.index, 
        y=results['Forecast'], 
        name='predicted' 
    ) 
] 
        plot_layout = go.Layout( 
        title='Sales Prediction',paper_bgcolor= 'rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)') 
    
        fig = go.Figure(data=plot_data, layout=plot_layout) 
        st.plotly_chart(fig)

    with col9:
    
        results = results.sort_values(by="Forecast")
        st.table(results["Forecast"].head(4))
        


dff["AR_QTY_ALL"] = dff["AR_QTY_ALL"].abs()





    



    
 

    

    

            

        


        



        





if menu_id =="Price Sensitivity Analysis":
    col1,col2 = st.columns([4,1])
    with col1:
        st.subheader("Price Sensitivity Analysis")
        st.markdown("In order to optimize sales, the elasticity of each product is crucial. Nevertheless, detecting how each customer reacts to the price changes in each product is key for tailored prices")      
        
        
        Customer = st.selectbox("Please Choose Customer",dff["CUSTOMER_ID"].unique().tolist())
        df4 = dff.loc[dff["CUSTOMER_ID"]==Customer]
        
        
        
    with col2:
        anim5 = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_OOvaPt.json")
        st_lottie(anim5, height=150,width=300)
    

     
    df_sen = df4[["CUSTOMER_ID","TRX_DATE","Category","Sub_Brand","UNIT_PRICE","AR_QTY_ALL"]]
    df_sen["TRX_DATE"] =  pd.to_datetime(df_sen['TRX_DATE'])
    df_laptop = df_sen.groupby(['TRX_DATE','Sub_Brand']).agg({'UNIT_PRICE':'mean','AR_QTY_ALL': 'mean' }).reset_index()
    x_pivot = df_laptop.pivot(index= 'TRX_DATE' ,columns='Sub_Brand' ,values='UNIT_PRICE')
    x_values = pd.DataFrame(x_pivot.to_records())
    y_pivot = df_laptop.pivot( index = 'TRX_DATE',columns='Sub_Brand', values='AR_QTY_ALL')
    y_values = pd.DataFrame(y_pivot.to_records())

    
    df_laptop["Revenue"] = df_laptop["UNIT_PRICE"]*df_laptop["AR_QTY_ALL"]
    
    df_laptop = df_laptop.sort_values(by="UNIT_PRICE",ascending=False)


    points = []
    results_values = {
        "Sub_Brand": [],
        "price_elasticity": [],
        "price_mean": [],
        "quantity_mean": [],
        "intercept": [],
        "t_score":[],
        "slope": [],
        "coefficient_pvalue" : [],
        "rsquared": []
}
#Append x_values with y_values per same product name
    for column in x_values.columns[1:]:
        column_points = []
        for i in range(len(x_values[column])):
            if not np.isnan(x_values[column][i]) and not np.isnan(y_values[column][i]):
                column_points.append((x_values[column][i], y_values[column][i]))
        df = pd.DataFrame(list(column_points), columns= ['x_value', 'y_value'])
    

    #Linear Regression Model
        import statsmodels.api as sm
        x_value = df['x_value']
        y_value = df['y_value']
        X = sm.add_constant(x_value)
        model = sm.OLS(y_value, X)
        result = model.fit()
    
    
    #(Null Hypothesis test) Coefficient with a p value less than 0.05
        if result.f_pvalue < 0.05:
        
            rsquared = result.rsquared
            coefficient_pvalue = result.f_pvalue
            intercept, slope = result.params
            mean_price = np.mean(x_value)
            mean_quantity = np.mean(y_value)
            tintercept, t_score = result.tvalues
     
        #Price elasticity Formula
            price_elasticity = (slope)*(mean_price/mean_quantity)    
            
        #Append results into dictionary for dataframe
            results_values["Sub_Brand"].append(column)
            results_values["price_elasticity"].append(price_elasticity)
            results_values["price_mean"].append(mean_price)
            results_values["quantity_mean"].append(mean_quantity)
            results_values["intercept"].append(intercept)
            results_values['t_score'].append(t_score)
            results_values["slope"].append(slope)
            results_values["coefficient_pvalue"].append(coefficient_pvalue)
            results_values["rsquared"].append(rsquared)
        
        
    final_df = pd.DataFrame.from_dict(results_values)
    df_elasticity =final_df[['Sub_Brand','price_elasticity','t_score','coefficient_pvalue','slope','price_mean','quantity_mean','intercept',"rsquared"]]
    
    if len(df_elasticity) == 0:
        st.subheader("Error")
        st.markdown("The following customer didnt perform enough transactions for an adequate price sensitivity analysis,please choose another one")

    if len(df_elasticity) >=1:
        col3,col4 = st.columns([4,0.25])
        col5,col6,col7,col8 = st.columns([3,3,0.25,3])
        with col3:
            st.subheader("Success!")
            st.markdown("The analysis for the chosen customer is ready. The following products are the ones purchased by the customer, choose one to view an analysis")
            product = st.selectbox("The following products came back with a result, please choose one to inspect more",df_elasticity["Sub_Brand"])
            df_elasticity = df_elasticity.loc[df_elasticity["Sub_Brand"]==product]
            st.markdown("----")
            

        
        
        with col5:
         st.markdown("Elasticity of demand")
         st.subheader("{:.2f}".format(df_elasticity.iloc[0,1]))
        st.markdown("Explanation:")
        if df_elasticity.iloc[0,1] <0 and df_elasticity.iloc[0,1] >=-0.5:
            with col6:
              st.markdown("Explanation")
              st.markdown("""For the chosen customer, this product is inelastic.
              Meaning that the change in demand was less than the change of price that occured!""")

            with col8:
             st.markdown("Reccomendations:")
             st.markdown("""A margin of profit still exists,therefore increasing the price should be considered.
                         This situation is best for optimizing revenue without losing traffic""")
             
        if df_elasticity.iloc[0,1] <-0.5 and df_elasticity.iloc[0,1] >-1:
            with col6:
              st.markdown("Explanation")
              st.markdown("""For the chosen customer, this product is inelastic.
              Meaning that the change in demand was less than the change of price that occured!""")

            with col8:
             st.markdown("Reccomendations:")
             st.markdown("""A margin of profit still exists,therefore increasing the price is a possibility in this case
                          However, the elasticity isnt so far from 1, meaning that price should be increased carefully in this case.""")
             
            
        if df_elasticity.iloc[0,1] ==-1:
            with col6:
              st.markdown("Explanation")
              st.markdown("""For the chosen customer, this product is perfectly inelastic.
              Meaning that the change in demand is equal to the change of price that occured!""")

            with col8:
             st.markdown("Reccomendations:")
             st.markdown("""Changing the price wouldnt highly effect the demand level of the customer.
             Should consider other factors and act accordingly""")
             


        if df_elasticity.iloc[0,1] <-1:
            with col6:
              st.markdown("Explanation")
              st.markdown("""For the chosen customer, this product is elastic.
              Meaning that the change in demand was greater than the change of price that occured implying a sensitivity to price""")

            with col8:
             st.markdown("Reccomendations:")
             st.markdown("""Decreasing the price a little bit might be the ideal situation in this case.
                         That would prevent the loss of the customer""")
             



        
        if df_elasticity.iloc[0,1] >0:
            with col6:
              st.markdown("Explanation")
              st.markdown("""In this rare case, elasticity is positive. Meaning that demand increased even though price increased! This either indicates that the product is seasonal and in-demand no matter the price, or has very low competitors in the market.""")
            with col8:
             st.markdown("Reccomendations:")
             st.markdown("""Increasing the price only means a higher profit margin in this case and would be ideal""")
        st.markdown("----")

             


        


        col9,col10,col11,col12= st.columns([4,2,2,1])
        with col9:
            st.markdown("Price and Quantity")
            option = {
            'xAxis': {
            'type': 'category',
            'boundaryGap': 'false',
            'data': df_laptop["UNIT_PRICE"].tolist()
  },
            'yAxis': {
            "splitLine":{'show':False},
            'type': 'value'
  },
            'series': [
    {
            'data': df_laptop["AR_QTY_ALL"].tolist(),
            'type': 'line',
            'areaStyle': {}
    }
  ]
}

            st_echarts(options=option)
        
        with col10:
            st.markdown("Average Price")
            avg_price = "{:.2f}".format(df_elasticity.iloc[0,5])
            avg_price = float(avg_price)
            avg_price_rem = 100000 - avg_price
            values = [avg_price,avg_price_rem]
            label = ["Correlation","Nothing"]
            colors = ["Yellow","Grey"]
            fig10 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
            fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
            fig10.update_layout(
            width=280,
            height = 300,
        
            showlegend=False,
    
            annotations=[dict(text=avg_price, x=0.5, y=0.5, font_size=20, showarrow=False)])


            fig10

        with col11:
            st.markdown("Average Quantity")
            avg_quan = "{:.2f}".format(df_elasticity.iloc[0,6])
            avg_quan = float(avg_quan)
            avg_quan_rem = 100 - avg_quan
            values = [avg_quan,avg_quan_rem]
            label = ["Correlation","Nothing"]
            colors = ["Green","Grey"]
            fig10 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
            fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
            fig10.update_layout(
            width=280,
            height = 300,
        
            showlegend=False,
    
            annotations=[dict(text=avg_quan, x=0.5, y=0.5, font_size=20, showarrow=False)])

            fig10
    
            revenue = df_elasticity.iloc[0,6]*df_elasticity.iloc[0,5]
            revenue =  "{:.2f}".format(revenue)
            
            st.metric("Revenue",revenue)
            

        st.markdown("Estimate Demand")
        with st.form("Assigning Weights",clear_on_submit=True):
          col1,col2,col3 = st.columns([3,3,3])
          with col1:
            st.markdown("Please enter your desired price for this product to estimate how much the customer would purchase based on their demand elasticity")
            price_input = st.number_input("Choose Price",avg_price)
            result = ((price_input - df_elasticity.iloc[0,5]) ) / price_input

            
            change = result * df_elasticity.iloc[0,1]
            

            demand = df_elasticity.iloc[0,6] - (-1*(df_elasticity.iloc[0,6]*change))
            demand = "{:.2f}".format(demand)
            st.form_submit_button()
        with col2:
            st.markdown("The demand would be:")
            st.subheader(demand)
            
            demand = float(demand)
            avg_quan = float(avg_quan)
            demand_diff = demand - avg_quan
            demand_diff = float(demand_diff)
            demand_diff = "{:.2f}".format(demand_diff)
            if demand> avg_quan:
                 st.markdown(f'The demand would change by {str(demand_diff)} units')

            
            if demand < avg_quan:
                
                st.markdown(f'The demand would change by {str(demand_diff)} units')

            

            
        with col3:
            
            st.markdown("The revenue would be:")
            rev1 = df_elasticity.iloc[0,5] * df_elasticity.iloc[0,6]
            rev1 = "{:.2f}".format(rev1)
            
            price_input = float(price_input)
            rev2 = (df_elasticity.iloc[0,6] - (-1*(df_elasticity.iloc[0,6]*change))) * price_input
            rev2 = "{:.2f}".format(rev2)
            st.subheader(rev2)
           
            if rev2 > rev1:
                revv = float(rev2)-float(rev1)
                revv = "{:.2f}".format(revv)
                st.markdown(f'The revenue would change by {str(revv)}L.L')
            
            if rev2 < rev1:
                revv = float(rev1)-float(rev2)
                revv = "{:.2f}".format(revv)
                
                st.markdown(f'The revenue would change by {str(revv)}L.L')

                
        st.markdown("Please not that the above numbers are estimations that dont consider other factors that might effect the demand")
            


if menu_id == "Market Basket Analysis":
    col1,col2 = st.columns([4,1])
    with col1:
        st.subheader("Market Basket Analysis")
        st.markdown("The following is a market basket analysis, which studies which products get bought together the most and hence aids in allocating products and offers better.")
        Area= st.selectbox("Area",dff["Area"].unique().tolist())
        st.markdown("Please specify the support and confidence you want")
    with st.form("Support",clear_on_submit=True):
            col4,col5 = st.columns([2,2])
            with col4:
                supp = st.number_input("Support",0.05)
                supp = np.asarray(supp, dtype='float64')
                with st.expander("Explanation"):
                    st.markdown("Support: the percentage of transactions in which product A appeared in.")
         
        
           
            with col5:
                con = st.number_input("Confidence",0.6)
                con = np.asarray(con, dtype='float64')
                with st.expander("Explanation"):
                    st.markdown("Confidence: the proportion that those who bought product A bought product B.")
    
            st.form_submit_button()
        
            

        
            
    with col2: 
        anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5kr5npck.json")
        st_lottie(anim, height=150,width=300)
        
  
    df_mba = dff[["CUSTOMER_TRX_ID","AR_QTY_ALL","Sub_Brand","UNIT_PRICE","Area"]] 
    df_mba["AR_QTY_ALL"] = df_mba["AR_QTY_ALL"].abs()
    def hot_encode(x):
        if(x<= 0):
            return 0
        if(x>= 1):
            return 1       
    basket = (df_mba[df_mba['Area'] ==Area]
          .groupby(['CUSTOMER_TRX_ID', 'Sub_Brand'])['AR_QTY_ALL']
          .sum().unstack().reset_index().fillna(0)
          .set_index('CUSTOMER_TRX_ID'))
    
    most_freq = basket.sum(axis = 0, skipna = True)
    most_freq = most_freq.to_frame().reset_index()
    most_freq= most_freq.rename(columns= {0: 'count'})
    most_freq.index.name = 'Item'
    most_freq = most_freq.sort_values(by="count",ascending=False)
     
    
    
    basket_encoded = basket.applymap(hot_encode)
    from mlxtend.frequent_patterns import apriori, association_rules
    frq_items = apriori(basket_encoded, min_support = supp, use_colnames = True)
    

    rules = association_rules(frq_items, metric="confidence", min_threshold = con)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    rules = pd.DataFrame(rules)


  
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    
    
    d = st.selectbox("*The products combination that have been found to have a consequent are stored in the below selectbox",rules["antecedents"].unique().tolist())
    
    
    
    reg = rules.loc[(rules["antecedents"]==d)]
    
    
    col6,col7,col8,col9 = st.columns([0.25,3,1,3])
    col11,col12 = st.columns([3,2])
    
    with col6:
        
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: #000080;
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: #FFFFFF;
        overflow-wrap: break-word;
}

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: white;
}
        </style>
"""
,        unsafe_allow_html=True)

        st.metric(label = "", value="1")
        
    with col11:    
        val1 = reg.iloc[0,5]
        val1 =  "{:.2f}".format(val1)
            
        val1 = float(val1)*100
            
            
        val2 = 100-val1

        
        values = [val1,val2]
        label = ["",""]
        colors = ["navy","grey"]
        fig10 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig10.update_layout(
        title = "Confidence",
        width = 280,
        height = 300,
        showlegend=False,
    
        annotations=[dict(name = "Confidence" ,text=val1, x=0.5, y=0.5, font_size=20, showarrow=False)])
        layout = go.Layout(
        margin=go.layout.Margin(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=0, #top margin
    )
)

        fig10 = dict(data=fig10, layout=layout)
            
        fig10


    
        with col7:
            st.markdown("Precedent")
            st.subheader(reg.iloc[0,1])
        
        with col9:
             st.markdown("Lift")
             st.subheader("{:.2f}".format(reg.iloc[0,6]))

        with col12:
            st.markdown("Interpretation")
            st.markdown("""This means that when the item-set chosen in the selectbox is bought,the precedent product is bought with them.
                         The proportion of customers who did so is determined by the confidence receieved.This aids in allocating offers, as an itemset with a high confidence wont require offering high discounts.
                         The Lift represents how likely it is that buying the initial item set would lead to buying the resulted product. Knowing this likelyhood would allow for better targeted campaigns with the products in a single bundle with a competitive price""")
            with st.expander("Recommendations"):
                st.markdown("High Confidence and Lift --> Offer the products together")
                st.markdown("Confidence between 40%-80%, a discount on a collective bundle would be ideal")
                st.markdown("Confidence lower than 40% with a lift higher than 1 might be worth it to look into the purchases done and by which customer")           
            
        if len(reg) >1:
            with st.expander("More Products Found"):
                st.dataframe(reg)

    st.markdown("----")  
    col13,col14 = st.columns([4,2])
    with col13:
            most_freqq = most_freq.head()
            most_freqq = most_freqq.sort_values(by="count",ascending=False)
            st.subheader("Most Occuring Products-(Key Products)")
            fig1 = px.bar(most_freqq,x="count",y="Sub_Brand",color_discrete_sequence =['navy']*len(most_freqq))
            fig1.update_layout(paper_bgcolor= 'rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1)
       
    with col14:
            num_transactions = len(basket)
            st.subheader("Total Transactions in Area")
            st.subheader(num_transactions)
            st.markdown("""These products,called key products, are the most re-occuring products among the transactions, implying that they are bought with several products.
                        Therefore, keeping the price of these products low might encourage people to purchase more, and hence purchase other products along""")


            with st.expander("View Frequency of all Products"):
                st.dataframe(most_freqq)

        
    
        

           


    




    




if menu_id == "RFM Analysis":
    st.subheader("RFM Analysis")
    dff["DATE_KEY2"] = pd.to_datetime(dff["DATE_KEY2"])
    
    col1,col2 = st.columns([4,1])
    
    with col1:
        st.markdown("""Segmenting Customers based on RFM score allows for better dynamic targetting and tailoring treatments.
        Please proceed to assigning weights for each aspect based on their significance in your case.""")
        with st.form("Assigning Weights",clear_on_submit=True):
            
                freq_input = st.number_input("Frequency")
                freq_input = np.asarray(freq_input, dtype='float64')
            
                recency_input = st.number_input("Recency")
                recency_input = np.asarray(recency_input, dtype='float64')
            
                mon_input = st.number_input("Monetary")
                mon_input = np.asarray(mon_input, dtype='float64')
                st.form_submit_button()
  
    with col2:
        
        anim = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_d4zpfpou.json")
        st_lottie(anim, height=150,width=300)

   
   
    rfm_box = st.selectbox("Choose Customer",dff["CUSTOMER_ID"].unique().tolist())


    
    df_recency = dff.groupby(by='CUSTOMER_ID',
                        as_index=False)['DATE_KEY2'].max()
    df_recency.columns = ['CUSTOMER_ID', 'LastPurchaseDate']
    recent_date = df_recency['LastPurchaseDate'].max()
    df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(
    lambda x: (recent_date - x).days)
    df_recency.head()

    frequency_df = dff.drop_duplicates().groupby(
    by=['CUSTOMER_ID'], as_index=False)['DATE_KEY2'].count()
    frequency_df.columns = ['CUSTOMER_ID', 'Frequency']
    frequency_df.head()
    frequency_df.sort_values("CUSTOMER_ID")
    
    monetary_df = dff.groupby(by='CUSTOMER_ID', as_index=False)['AR_AMOUNT_ALL'].sum()
    monetary_df.columns = ['CUSTOMER_ID', 'Monetary']
    monetary_df.head()

    rf_df = df_recency.merge(frequency_df, on='CUSTOMER_ID')
    rfm_df = rf_df.merge(monetary_df, on='CUSTOMER_ID').drop(
    columns='LastPurchaseDate')
    rfm_df.head()

    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)
 
# normalizing the rank of the customers
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['M_rank']/rfm_df['M_rank'].max())*100
 
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
 
    rfm_df.head()


    rfm_df['RFM_Score'] = recency_input*rfm_df['R_rank_norm']+freq_input * \
    rfm_df['F_rank_norm']+mon_input*rfm_df['M_rank_norm']
    rfm_df['RFM_Score'] *= 0.05
    rfm_df = rfm_df.round(2)
    rfm_df[['CUSTOMER_ID', 'RFM_Score']].head(7)

    rfm_df["Recency_Score"] = rfm_df["R_rank_norm"]*0.05
    rfm_df["Frequency_Score"] = rfm_df["F_rank_norm"]*0.05
    rfm_df["Monetary_Score"] = rfm_df["M_rank_norm"]*0.05
    rfm_df.sort_values("RFM_Score",ascending=True)
    
    df1= dff.merge(rfm_df,on="CUSTOMER_ID")
    df2 = df1.loc[(df1["CUSTOMER_ID"]==rfm_box)]
    
   
    
    
    st.markdown("----")



    rfm_score = "{:.2f}".format(df2["RFM_Score"].mean())
    rfm_score = float(rfm_score)
    
    
    col6,col7,col8 = st.columns([3,3,3])
    with col6:
        st.markdown("RFM Score:")
        st.subheader(rfm_score)
    with col7:
        st.markdown("Area")
        st.subheader(df2.iloc[0,61])
        
        
    with col8:
        st.markdown("Customer Type")
        
       
        if rfm_score >=0 and rfm_score < 1.5:
            type = "Lost Customer"
            st.subheader(type)
            
        if rfm_score >= 1.5 and rfm_score <3:
            type = "Almost Lost Customer"
            st.subheader(type)
           
        if rfm_score >=3 and rfm_score < 4:
            type = "Required Attention"
            st.subheader(type)
        if rfm_score >=4 and rfm_score < 4.5:
            type = "Potential Champion"
            st.subheader(type)
        if rfm_score >=4.5 and rfm_score < 5:
            type = "Champion"
            st.subheader(type)
        
    st.markdown("----")

    col9,col10,col11,col12,col13 = st.columns([2,0.25,2,0.25,2])
    with col9:
        st.markdown("Recency Score:")
        theme_rec = {'bgcolor':'#E6676B','content_color':'white','progress_color':'white','title_color':'white','icon':'bi bi-hourglass','icon_color':'white','font_size':'100',}
        hc.info_card(title="Recency Score",content = "{:.2f}".format(df2["Recency_Score"].mean()),theme_override=theme_rec)
        

    with col11:
        st.markdown("Frequency Score:")
        theme_freq = {'bgcolor':'#ADD8E6','content_color':'white','progress_color':'white','title_color':'white','icon':'bi bi-bar-chart-line','icon_color':'white','font_size':'100',}
        hc.info_card(title="Frequency Score",content = "{:.2f}".format(df2["Frequency_Score"].mean()),theme_override=theme_freq)
        
        

        

    with col13:
        st.markdown("Monetary Score")
        theme_mon = {'bgcolor':'#DAF7A6 ','content_color':'white','progress_color':'white','title_color':'white','icon':'bi bi-cash-stack','icon_color':'white','font_size':'100',}
        hc.info_card(title="Monetary Score",content = "{:.2f}".format(df2["Monetary_Score"].mean()),theme_override=theme_mon)
    
    col14,col15,col16 = st.columns([3,3,3])
    
    with col14:
        Rec_score = "{:.2f}".format(df2["R_rank_norm"].mean())
        Rec1_score = float(Rec_score)
        Rec2_score = 100 - Rec1_score
        Rec2_score = "{:.2f}".format(Rec2_score)
        Rec2_score = float(Rec2_score)
        Rec_score_rem = 100 - Rec2_score
        
        
        values = [Rec2_score,Rec_score_rem]
        label = ["",""]
        colors = ["Green","grey"]
        fig8 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig8.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig8.update_layout(title="Rank within the top%:",
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=Rec2_score, x=0.5, y=0.5, font_size=20, showarrow=False)])

        fig8

        st.markdown("Recency Recommendations:")
        if df2["Recency_Score"].mean()>=0 and df2["Recency_Score"].mean()<2:
            st.markdown("In terms of recency, this customer is probably a lost customer.")
            st.markdown("However, a high frequency and monetary score might reflect that it is worth contacting the customer.")
            st.markdown("Offering a discount and a good deal, or checking what went wrong are possible communication topics")
            with st.expander("Details"):
                df_details = df2[["TRX_DATE","Sub_Brand","SALESREP_ID"]]
                df_details = df_details.sort_values(by="TRX_DATE",ascending=False)
                st.markdown("Last day of purchase with product and salesperson dealt with")
                st.table(df_details.head(1))

        if df2["Recency_Score"].mean()>=2 and df2["Recency_Score"].mean()<3.5:
            st.markdown("This customer isnt a lost customer yet, but their recency score requires attention.")
            st.markdown("Offering them a tailored package and optimal price to retain them would be recommended.")
            with st.expander("Details"):
                df_details = df2[["TRX_DATE","Sub_Brand","SALESREP_ID"]]
                df_details = df_details.sort_values(by="TRX_DATE",ascending=False)
                st.markdown("Last day of purchase with product and salesperson dealt with")
                st.table(df_details.head(1))
                

        
        if df2["Recency_Score"].mean()>=3.5 and df2["Recency_Score"].mean()<4:
            st.markdown("Recency Score seems well!") 
            st.markdown("Might require an offer or a bundle from time to time!")

        
        if df2["Recency_Score"].mean()>=4 and df2["Recency_Score"].mean()<=5:
            st.markdown("Recency Score is optimal!") 
        
                




    with col15:
        Freq_score = "{:.2f}".format(df2["F_rank_norm"].mean())
        Freq1_score = float(Freq_score)
        Freq2_score = 100 - Freq1_score
        Freq2_score = "{:.2f}".format(Freq2_score)
        Freq2_score = float(Freq2_score)
        Freq_score_rem = 100 - Freq2_score
        values = [Freq2_score,Freq_score_rem]
        label = ["",""]
        colors = ["Green","grey"]
        fig8 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig8.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig8.update_layout(
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=Freq2_score, x=0.5, y=0.5, font_size=20, showarrow=False)])

        fig8
        st.markdown("Frequency Recommendations:")
        if df2["Frequency_Score"].mean()>=0 and df2["Frequency_Score"].mean()<2:
            st.markdown("This customer isnt a frequent customer.")
            st.markdown("If they bring high monetary value, perhaphs understanding the problem requires using the MBA tool and the Price sensitivity analysis tool to understand their behavior")
            
            

        if df2["Frequency_Score"].mean()>=2 and df2["Frequency_Score"].mean()<3.5:
            st.markdown("This customer still bounces back from time to time, but they might be dealing with another distributor or competitors.")
            st.markdown("In this case, shwoing them great customer care such as tailored prices and discounts might omit competition.")


        
        if df2["Frequency_Score"].mean()>=3.5 and df2["Frequency_Score"].mean()<4:
            st.markdown("Frequency Score seems well!") 
            st.markdown("Might require an offer or a bundle from time to time!")

        
        if df2["Frequency_Score"].mean()>=4 and df2["Frequency_Score"].mean()<=5:
            st.markdown("The customer is loyal!") 
            st.markdown("However, this means giving them a special treatment, and making sure they know it!")


        


    with col16:
        
        Mon_score = "{:.2f}".format(df2["M_rank_norm"].mean())
        Mon_score = float(Mon_score)
        Mon_score = 100 - Mon_score
        Mon_score = "{:.2f}".format(Mon_score)
        Mon_score = float(Mon_score)
        Mon_score_rem = 100 - Mon_score

        values = [Mon_score,Mon_score_rem]
        label = ["",""]
        colors = ["Green","grey"]
        fig10 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig10.update_layout(
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=Mon_score, x=0.5, y=0.5, font_size=20, showarrow=False)])

        fig10  

        st.markdown("Monetary Recommendations:")
        if df2["Monetary_Score"].mean()>=0 and df2["Monetary_Score"].mean()<2:
            st.markdown("This customer isnt of a high monetary value for the company.")
            st.markdown("Please consider that if the other two scores are low as well, this might mean that the customer is new")
            
            

        if df2["Monetary_Score"].mean()>=2 and df2["Monetary_Score"].mean()<3.5:
            st.markdown("This customer might be a good investment of the company's time and effort to retain them.")
            st.markdown("Consider incentives and offers based on their price sensitivity and items preferences.")


        
        if df2["Monetary_Score"].mean()>=3.5 and df2["Monetary_Score"].mean()<4:
            st.markdown("Monetary Score seems well!") 
            st.markdown("Might require an offer or a bundle from time to time to show appreciation!")

        
        if df2["Monetary_Score"].mean()>=4 and df2["Monetary_Score"].mean()<=5:
            st.markdown("The customer is of great monetary importance!") 
            st.markdown("However, this means giving them a special treatment, and making sure they know it!")     

        
        
        
   
    
    

if menu_id == "Recommendation System":

    st.subheader("Product Recommendation System")
    col11,col22 = st.columns([3,1])
    with col11:

     st.markdown("Product Recommendation system based on previous purchases to enhance and increase product sales")
     D = st.selectbox("Select Product",("Koleston 2000","Koleston Foam","Cosmaline Sanitizer","Cosmaline Baby","Koleston Kit","Bodyguard Mask","Softwave Kids Shower","Softwave Kids Haircare","Wella House","Koleston Tube","Symi","Basic","Cosmaline Facial","Soft Wave Shower","Soft Wave Hair Care","Soft Wave Liquid Soap","Cosmal Cure Professional","Cosmaline Generic","Koleston Naturals","Doucy","Koleston Natural","Cosmaline Gentlemen","Cosmaline Hair & Body Mist"))
     if D == "Koleston 2000":
        rs = "Koleston 2000"
     if D =="Koleston Foam":
        rs = "Koleston Foam"
    
     if D =="Cosmaline Sanitizer":
        rs = "Cosmaline Sanitizer"
    
     if D =="Cosmaline Baby":
        rs = "Cosmaline Baby"
    
     if D =="Koleston Kit":
        rs = "Koleston Kit"
    
     if D =="Bodyguard Mask":
        rs = "Bodyguard Mask"

    
     if D =="Softwave Kids Shower":
        rs = "Softwave Kids Shower"

    
     if D =="Softwave Kids Haircare":
        rs = "Softwave Kids Haircare"

    
     if D =="Wella House":
        rs = "Wella House"

    
     if D =="Koleston Tube":
        rs = "Koleston Tube"

    
     if D =="Symi":
        rs = "Symi"

    
     if D =="Basic":
        rs = "Basic"
    
     if D =="Cosmaline Facial":
        rs = "Cosmaline Facial"

     if D =="Soft Wave Shower":
        rs = "Soft Wave Shower"
    
     if D =="Soft Wave Hair Care":
        rs = "Soft Wave Hair Care"

     if D =="Soft Wave Liquid Soap":
        rs = "Soft Wave Liquid Soap"

    
     if D =="Cosmal Cure Professional":
        rs = "Cosmal Cure Professional"
    
     if D =="Cosmaline Generic":
        rs = "Cosmaline Generic"

     if D =="Koleston Naturals":
        rs = "Koleston Naturals"
    
     if D =="Koleston Natural":
        rs = "Koleston Natural"

     if D =="Cosmaline Gentlemen":
        rs = "Cosmaline Gentlemen"
     if D =="Cosmaline Hair & Body Mist":
        rs = "Cosmaline Hair & Body Mist"
    
     if D =="Koleston Kit":
        rs = "Koleston Kit"
    
    








    
    with col22:
        anim3 = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_zok6ip7g.json")
        st_lottie(anim3,key="hello", height=150,width=300)

    
    st.markdown("----")
    st.markdown("Correlation in %")
    
    col1,col2,col3,col4,col5,col6 = st.columns([0.25,2,0.25,2,0.25,2])
    
    with col1:
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: #000080;
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: #FFFFFF;
        overflow-wrap: break-word;
}

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: white;
}
        </style>
"""
,        unsafe_allow_html=True)

        st.metric(label = "", value="1")

    
    
    df6 = dff[["Sub_Brand","CUSTOMER_TRX_ID","AR_QTY_ALL"]]
    
    df_items = df6.pivot_table(index='CUSTOMER_TRX_ID', columns=['Sub_Brand'], values='AR_QTY_ALL').fillna(0)
    
    def get_recommendations(df, item):
    
     recommendations = df.corrwith(df[item])
     recommendations.dropna(inplace=True)
     recommendations = pd.DataFrame(recommendations, columns=['correlation']).reset_index()
     recommendations = recommendations.sort_values(by='correlation', ascending=False)
     return recommendations
     


    recommendations = get_recommendations(df_items, rs)
    
    with col2:
      st.markdown("Product Name")

      st.subheader(recommendations.iloc[1,0])

    with col3:
        st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: #000080;
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: #FFFFFF;
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: white;
}
</style>
"""
, unsafe_allow_html=True)

        st.metric(label = "", value="2")
    
    with col4:
          st.markdown("Product Name")

          st.subheader(recommendations.iloc[2,0])

    
    with col5:
        st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: #000080;
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: #FFFFFF;
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: white;
}
</style>
"""
, unsafe_allow_html=True)

        st.metric(label = "", value="3")


    with col6:
        st.markdown("Product Name")
        st.subheader(recommendations.iloc[3,0])



    
    


    
    

    col7,col8,col9 = st.columns([2.5,2.5,2.5])

    with col7:
        g1 = round(recommendations.iloc[1,1]*100)
        h1 = 100 - g1
        values = [g1,h1]
        label = ["Correlation","Nothing"]
        colors = ["Green","Grey"]
        fig8 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig8.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig8.update_layout(
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=g1, x=0.5, y=0.5, font_size=20, showarrow=False)])
        

        fig8

    with col8:
        g2 = round(recommendations.iloc[2,1]*100)
        h2 = 100-g2
        values = [g2,h2]
        label = ["Correlation","Nothing"]
        colors = ["Orange","Grey"]
        fig9 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig9.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig9.update_layout(
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=g2, x=0.5, y=0.5, font_size=20, showarrow=False)])

        fig9

    with col9:

        g3 = round(recommendations.iloc[3,1]*100)
        h3 = 100-g3
        
        values = [g3,h3]
        label = ["Correlation","Nothing"]
        colors = ["Navy","Grey"]
        fig10 = go.Figure(data=[go.Pie(labels=label, values=values, hole=.9)])
        fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
        fig10.update_layout(
        width=260,
        height = 300,
        
        showlegend=False,
    
        annotations=[dict(text=g3, x=0.5, y=0.5, font_size=20, showarrow=False)])

        fig10
     
    with st.expander("All Products"):
    
         ff = px.bar(recommendations,x='Sub_Brand',y="correlation",color_discrete_sequence =['navy']*len(recommendations))
         ff.update_layout(paper_bgcolor= 'rgba(0,0,0,0)',
         plot_bgcolor='rgba(0,0,0,0)')
         ff.update_traces(width=0.5)
         st.plotly_chart(ff)



    col44,col55,col66,col77= st.columns([4,0.25,0.25,0.25])
    with col44:
        st.subheader("Suitable Contact")
    

    with col77:
        from PIL import Image
        image = Image.open('greentick.jpg')

        st.image(image)
        

    col88,col99,col100,col101= st.columns([2,2,2,1])
    
    with col88:
        st.markdown("The customers that purchase product 1:")
        df_suitablecontact = dff.loc[dff["Sub_Brand"]==recommendations.iloc[1,0]]
        df_suitablecontact1 =  df_suitablecontact.groupby(["CUSTOMER_ID"],as_index=False)["AR_QTY_ALL"].sum()
        df_suitablecontact1 = df_suitablecontact1.sort_values(by="AR_QTY_ALL",ascending=False)
        st.table(df_suitablecontact1.head())

    with col99:
        
        st.markdown("The customers that purchase product 2:")
        df_suitablecontact = dff.loc[dff["Sub_Brand"]==recommendations.iloc[2,0]]
        df_suitablecontact1 =  df_suitablecontact.groupby(["CUSTOMER_ID"],as_index=False)["AR_QTY_ALL"].sum()
        df_suitablecontact1 = df_suitablecontact1.sort_values(by="AR_QTY_ALL",ascending=False)
        
        st.table(df_suitablecontact1.head())

    with col100:
        
        st.markdown("The customers that purchase product 3:")
        df_suitablecontact = dff.loc[dff["Sub_Brand"]==recommendations.iloc[1,0]]
        df_suitablecontact1 =  df_suitablecontact.groupby(["CUSTOMER_ID"],as_index=False)["AR_QTY_ALL"].sum()
        df_suitablecontact1 = df_suitablecontact1.sort_values(by="AR_QTY_ALL",ascending=False)
        
        st.table(df_suitablecontact1.head())




    
    
        

       
        
        
        
    
       
    

if menu_id == "Google Trends":
    import pytrends
    from pytrends.request import TrendReq

    st.subheader("Google Trends")
    col1,col2 = st.columns([3,1])
    with col1:
        st.markdown("Monitoring real time demand is crucial to allocate our products, and optimize our sales. Lets use Google trends to inspect each product's demand and push our sales per region")
        select = st.selectbox("Choose Topic",("Cosmaline","Wella","Hair Care","Koleston","Hair Color","Facial","skin care","baby care","cosmaline shampoo"))
    
    with col2:
        
        anim2 = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_dffnrlva.json")
        st_lottie(anim2,key="hello", height=150,width=300)


    pytrend = TrendReq()
    kw_list = []
   
    if select == "Cosmaline":
        kw_list = ["Cosmaline"]
    if select == "Wella":
        kw_list = ["Wella"]
    if select == "Hair Care":
        kw_list = ["Hair Care"]
    if select == "Koleston":
        kw_list = ["Koleston"]
    if select == "Hair Color":
        kw_list = ["Hair Color"]
    if select == "Facial":
        kw_list = ["Facial"]
    if select == "skin care":
        kw_list = ["skin care"]
    if select == "baby care":
        kw_list = ["baby care"]
    if select == "cosmaline shampoo":
        kw_list = ["cosmaline shampoo"]
        
        
        
        

        
        
        

    pytrend.build_payload(kw_list=kw_list, timeframe='today 12-m')
    interest_over_time_df = pytrend.interest_over_time().drop(columns='isPartial')
    col4,col5 = st.columns([4,2])
    with col4:
        pd.options.plotting.backend = "plotly"
    
        fig = interest_over_time_df[kw_list].plot()
        fig.update_layout(
        title_text='Search volume over time',
        legend_title_text='Search terms',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig) 

        pytrend.build_payload(kw_list, geo='LB')

# requesting data
        interest_by_region_df = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True)

# moving geoName from the index to the column
        interest_by_region_df.reset_index(inplace=True)
        interest_by_region_df = interest_by_region_df.sort_values(by = kw_list,ascending=True)
        fig1 = px.bar(interest_by_region_df,x=kw_list,y='geoName',color_discrete_sequence =['navy']*len(interest_by_region_df))
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            
        
        )
        st.plotly_chart(fig1)

# showing the DataFrame example
       
        

      
       
    with  col5:
        st.table(interest_over_time_df.head(10))
        interest_by_region_df = interest_by_region_df.sort_values(by= kw_list,ascending=False)
        
        st.table(interest_by_region_df.head(10))
    

if menu_id == "Competition":
    import pytrends
    from pytrends.request import TrendReq
    pytrend = TrendReq()

    st.subheader("Competitors Analysis")
    col1,col2 = st.columns([3,1])
    col11,col22,col33 = st.columns([3,3,3])
    with col1:
        st.markdown("Competitors' analysis is an important part in terms of understanding the market demand and where the customers are searching for similar products")
        
        e = st.selectbox("Default options",("Beesline","LOLITAS CLOSET"))
        if e == "Beesline":
            d="Beesline"
        
            
            
        if e == "LOLITAS CLOSET":
            d="LOLITAS CLOSET"
        kw_list1 = ["Cosmaline",d]
        kw_list3 = [d]

        if st.checkbox("Enable Custom Search"):
            d = st.text_input("Search for any competitor",value="Beesline") 
            
            kw_list1 = ["Cosmaline",d]
            kw_list3 = [d]

        


        
        
   

    with col2:
        animm = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_cv6rdeii.json")
        st_lottie(animm, height=150,width=300)
        
        
        
        
        
            
    st.markdown("----")
    
    
    
   

    

    
    

    
    st.subheader("Analysis and Comparison")
    col33,col44,col55= st.columns([3,3,2])
    with col33:
     t=st.selectbox("Choose Time Span",('12 months','5 years','3 months'))

     if t =='12 months':
        time_frame = 'today 12-m'

    
    
    
     if t =='3 months':
        time_frame = 'today 3-m'
    
    
     if t =='5 years':
        time_frame = 'today 5-y'

    with col33:
        pd.options.plotting.backend = "plotly"
        pytrend.build_payload(kw_list=kw_list1, timeframe=time_frame)
        interest_over_time_df1 = pytrend.interest_over_time().drop(columns='isPartial')
        fig2 = interest_over_time_df1[kw_list1].plot()
        fig2.update_layout(
        
        title_text = "Search Volume Over Time",
         
        legend_title_text='Search terms',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        
        width = 800)
         
     

        st.plotly_chart(fig2)
    


    

    with col55:
        st.dataframe(interest_over_time_df1)


    col3,col4 = st.columns([3,3])
    with col3:
         st.markdown("Search by Region")
        
         kw_list2 = ["Cosmaline"]

         pytrend.build_payload(kw_list2, geo='LB',timeframe=time_frame)


         interest_by_region_df2 = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True)

         interest_by_region_df2.reset_index(inplace=True)
         
         interest_by_region_df2 = interest_by_region_df2.sort_values(by = kw_list2,ascending=True)
         pytrend.build_payload(kw_list3, geo='LB',timeframe=time_frame)


         interest_by_region_df3 = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True)

         interest_by_region_df3.reset_index(inplace=True)
         interest_by_region_df3 = interest_by_region_df3.sort_values(by = kw_list3,ascending=True)
         interest_by_region_all = interest_by_region_df2.merge(interest_by_region_df3,on="geoName")

         

         fig3 = px.bar(interest_by_region_all,x=[interest_by_region_all.iloc[:,1],interest_by_region_all.iloc[:,2]],y='geoName',color_discrete_sequence =(['navy']*len(interest_by_region_all.iloc[:,1]),['red']*len(interest_by_region_all.iloc[:,2])))
         fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width = 600,
            showlegend=False
            )
         category_orders={"interest_by_region_all.iloc[:,1]": ["Cosmaline"],
                        "interest_by_region_all.iloc[:,2]": ["Competitor"]}
         
         st.plotly_chart(fig3)

         
    with col4: 
         st.markdown("Search Share")    
         mean_1 = interest_by_region_all.iloc[:,1].mean()
         mean_2 = interest_by_region_all.iloc[:,2].mean() 
          

         for_mean1 = mean_1/(mean_1 + mean_2)
         for_mean2 = mean_2/(mean_2 + mean_1)
         value1 = for_mean1
         value2 = for_mean2
         values = [value1,value2]
         label = ["Cosmaline","Competitor"]
         colors = ["navy","red"]
         fig10= go.Figure(data=[go.Pie(labels=label, values=values, hole=0)])
         fig10.update_traces(hoverinfo='label+percent+name', textinfo='none',marker=dict(colors=colors))
         fig10.update_layout(
         
         
        
         showlegend=False,
    
         annotations=[dict(text="", x=0.5, y=0.5, font_size=20, showarrow=False)])
        
         fig10
    st.markdown("Related Queries") 
    st.markdown("----")   
    col5,col6 = st.columns([3,3])
    
    with col5:
        
        st.markdown("Cosmaline")
        pytrend.build_payload(kw_list2, timeframe=time_frame)
        kw_list2_df = pd.DataFrame(kw_list2)
  
        rq = pytrend.related_queries()
        st.table(rq[kw_list2_df.iloc[0,0]]['top'])
    
    with col6:
        st.markdown("Competitor")
        pytrend.build_payload(kw_list3, timeframe=time_frame)
        kw_list3_df = pd.DataFrame(kw_list3)
         

        rq = pytrend.related_queries()
        rq.values()
        st.table(rq[kw_list3_df.iloc[0,0]]['top'])
    

         


  
 
         

    
        
         
    
   

         





       

      

    
