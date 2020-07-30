# import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = pickle.load(open('FPP_model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features =np.array(features)
    print(final_features)
    final_features=final_features.reshape(1,7)

    df_test=pd.DataFrame(data=final_features,columns=['Airline','Source','Destination','Dep_Time','Arrival_Time','Total_Stops','Additional_Info'])

    # print(df_test.head(1))

    input_list=[0]*30

    array2=np.array(input_list)
    array2=array2.reshape(1,30)

    df_predict=pd.DataFrame(data=array2,columns=['Total_Stops','Additional_Info','Airline_Air India','Airline_GoAir','Airline_IndiGo','Airline_Jet Airways',
                                     'Airline_Jet Airways Business','Airline_Multiple carriers','Airline_Multiple carriers Premium economy','Airline_SpiceJet','Airline_Trujet',
                                     'Airline_Vistara','Airline_Vistara Premium economy','Source_Chennai','Source_Delhi','Source_Kolkata','Source_Mumbai','Dest_Cochin',
                                     'Dest_Delhi','Dest_Hyderabad','Dest_Kolkata','Dest_New Delhi','Day_of_Week','Day_of_Journey','Month_of_Journey','Arrival_Hour','Arrival_Minute',
                                     'Departure_Hour','Departure_Minute','Total_Duration'])



    df_test.Total_Stops.replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)
    df_test["Total_Stops"] = df_test["Total_Stops"].astype(int)

    df_predict["Total_Stops"]= df_test["Total_Stops"]
    df_predict["Total_Stops"] = df_predict["Total_Stops"].astype(int)


    # print("**********Airlines**********************")

    df_predict['Airline_Air India'][0] = 0
    df_predict['Airline_GoAir'][0] = 0
    df_predict['Airline_IndiGo'][0] = 0
    df_predict['Airline_Jet Airways'][0] = 0
    df_predict['Airline_Jet Airways Business'][0] = 0
    df_predict['Airline_Multiple carriers'][0] = 0
    df_predict['Airline_Multiple carriers Premium economy'][0] = 0
    df_predict['Airline_SpiceJet'][0] = 0
    df_predict['Airline_Trujet'][0] = 0
    df_predict['Airline_Vistara'][0] = 0
    df_predict['Airline_Vistara Premium economy'][0] = 0

    if(df_test['Airline'][0]=='Air India'):
        df_predict['Airline_Air India'][0]=1


    if (df_test['Airline'][0]== 'GoAir'):
        df_predict['Airline_GoAir'][0] = 1

    if (df_test['Airline'][0] == 'IndiGo'):
        df_predict['Airline_IndiGo'][0]= 1

    if (df_test['Airline'][0]== 'Jet Airways'):
        df_predict['Airline_Jet Airways'][0]= 1

    if (df_test['Airline'][0]== 'Jet Airways Business'):
        df_predict['Airline_Jet Airways Business'][0] = 1

    if (df_test['Airline'][0]== 'Multiple carriers'):
        df_predict['Airline_Multiple carriers'][0] = 1



    if (df_test['Airline'][0] == 'Multiple carriers Premium economy'):
        df_predict['Airline_Multiple carriers Premium economy'][0] = 1

    if (df_test['Airline'][0] == 'SpiceJet'):
        df_predict['Airline_SpiceJet'][0]= 1

    if (df_test['Airline'][0] == 'Trujet'):
        df_predict['Airline_Trujet'][0] = 1

    if (df_test['Airline'][0] == 'Vistara'):
        df_predict['Airline_Vistara'][0] = 1

    if (df_test['Airline'][0] == 'Vistara Premium economy'):
        df_predict['Airline_Vistara Premium economy'][0]= 1

    # print("**********Sources**********************")

    df_predict['Source_Chennai'][0]= 0
    df_predict['Source_Delhi'][0] = 0
    df_predict['Source_Mumbai'][0] = 0
    df_predict['Source_Kolkata'][0]= 0

    if (df_test['Source'][0] == 'Chennai'):
        df_predict['Source_Chennai'][0] = 1

    if (df_test['Source'][0]== 'Delhi'):
        df_predict['Source_Delhi'][0]= 1

    if (df_test['Source'][0]== 'Mumbai'):
        df_predict['Source_Mumbai'][0] = 1

    if (df_test['Source'][0] == 'Kolkata'):
        df_predict['Source_Kolkata'][0] = 1

    df_predict['Dest_Cochin'][0] = 0
    df_predict['Dest_Delhi'][0] = 0
    df_predict['Dest_Hyderabad'][0] = 0
    df_predict['Dest_Kolkata'][0] = 0
    df_predict['Dest_New Delhi'][0] = 0

    if (df_test['Destination'][0] == 'Cochin'):
        df_predict['Dest_Cochin'][0] = 1

    if (df_test['Destination'][0] == 'Delhi'):
        df_predict['Dest_Delhi'][0] = 1

    if (df_test['Destination'][0] == 'Hyderabad'):
        df_predict['Dest_Hyderabad'][0] = 1
    if (df_test['Destination'][0]== 'Kolkata'):
        df_predict['Dest_Kolkata'][0] = 1

    if (df_test['Destination'][0]== 'New Delhi'):
        df_predict['Dest_New Delhi'][0] = 1



    departure_date,departure_time = df_test['Dep_Time'][0].split('T')
    arrival_date,arrival_time= df_test['Arrival_Time'][0].split('T')
    year_of_journey,month_of_journey,day_of_journey =departure_date.split('-')


    df_predict["Day_of_Journey"]=day_of_journey



    day_week=int(day_of_journey)

    df_predict["Day_of_Week"] =day_week%7

    df_predict["Month_of_Journey"]=month_of_journey



    df_predict['Arrival_Hour'],df_predict['Arrival_Minute'] = arrival_time.split(':')


    df_predict['Arrival_Hour'] = df_predict['Arrival_Hour'].astype(int)
    df_predict['Arrival_Minute'] =df_predict['Arrival_Minute'].astype(int)


    df_predict['Departure_Hour'], df_predict['Departure_Minute'] =departure_time.split(':')


    df_predict['Departure_Hour'] = df_predict['Departure_Hour'].astype(int)
    df_predict['Departure_Minute'] = df_predict['Departure_Minute'].astype(int)



    if (departure_date==arrival_date):
        df_predict['Total_Duration'] = (df_predict['Arrival_Hour']-df_predict['Departure_Hour'])*60 + abs(df_predict['Arrival_Minute']-df_predict['Departure_Minute'])
    else:
        df_predict['Total_Duration'] = (24-df_predict['Departure_Hour']+df_predict['Arrival_Hour']) * 60 + df_predict['Arrival_Minute']- df_predict['Departure_Minute']

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    df_predict["Additional_Info"] = encoder.fit_transform(df_test['Additional_Info'])

    prediction = model.predict(df_predict)

    output = round(prediction[0])

    print(output)

    return render_template('details.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)