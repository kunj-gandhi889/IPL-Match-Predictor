#### IPL WINNER PREDICTOR
import random
import numpy as np 
import streamlit as st  # for web app gui
import pickle,bz2
import pandas as pd
import matplotlib.pyplot as plt
import time
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='IPL Win Predictor',layout="wide")

color = ["#ebdd1c","#4835f2","#ff0000","#861bbf","#0c71ed","#e831d9","#bd0000","#bd4800"]  # color theme of teams
tagline = ["Whistle Podu","Ye hai nayi dilli","Live Punjabi Play Punjabi","Korbo Lorbo Jeetbo Re","Duniya hila Denge Hum","Halla Bol","Play Bold","Orange Army"]

with bz2.BZ2File("Teams.pbz2","rb") as f:
    teams = pickle.load(f)
with bz2.BZ2File("Cities.pbz2","rb") as f:
    cities = pickle.load(f)
with bz2.BZ2File("struct.pbz2","rb") as f:
    pipe_o = pickle.load(f)
with bz2.BZ2File("final_df.pbz2","rb") as f:
    final_data = pickle.load(f)

def getImage(team):   # for logo
    path = f"./LOGO/{team}.png"
    st.image(path)

def getColor(team):
    i = teams.index(team)
    color_team = color[i]
    return color_team

def getTagline(team):
    i = teams.index(team)
    tag = tagline[i]
    return tag

def match_prob(id,pipe):
    data = final_data[final_data["match_id"]==id]
    xyz = final_data.query(f"((balls_faced>=114 and balls_faced<=119) or balls_faced%6 ==0 ) and match_id == {id}")
    xyz.sort_values("balls_faced",inplace=True)
    wxy = xyz[xyz["balls_faced"]%6==0]
    wxy = pd.concat([wxy,xyz.iloc[-1].to_frame().T])
    data = wxy.iloc[:,1:-1]
    result = pipe.predict_proba(data)
    data["lose_prob"] = np.round(result.T[0]*100,1)
    data["win_prob"] = np.round(result.T[1]*100,1)
    target = data["target"].values[0]
    batting_team = data["batting_team"].values[0]
    bowling_team = data["bowling_team"].values[0]
    city = data["city"].values[0]
    new = data[["current_score","balls_faced","wickets_fallen","lose_prob","win_prob"]]
    new['end_of_over'] = range(1,new.shape[0]+1)
    new["runs_in_over"] = new["current_score"].diff()
    over1 = new["current_score"].values[0]
    wicket1 = new["wickets_fallen"].values[0]
    new["runs_in_over"] = new["runs_in_over"].fillna(over1) 
    new["runs_in_over"] = new["runs_in_over"].astype(int)
    new["wicket_in_over"] = new["wickets_fallen"].diff()
    new["wicket_in_over"] = new["wicket_in_over"].fillna(wicket1)
    new["wicket_in_over"] = new["wicket_in_over"].astype(int)
    return new,target,batting_team,bowling_team,city

select = st.sidebar.radio("Navigation Menu",["Prediction Model","Dataset","About Project"])

# predict control variables
make_false = 0  
ball_exceed = 0
inconsistent_input = 0
getResult = 0

if(select == "Prediction Model"):   
    st.markdown("<h1 style='text-align: center; color:red'>IPL MATCH PREDICTOR</h1>",unsafe_allow_html=True)

    col1 , col2 = st.columns(2)
    with col1:
        bat_team = st.selectbox("Choose Batting Team",teams)

    with col2:
        bowl_team = st.selectbox("Choose Bowling Team",teams) 

    if(bat_team == bowl_team):
        make_false = 1

    city = st.selectbox("Select City",cities)
    target = st.number_input("Target",min_value=0,max_value=400)
    runs = st.number_input("Current Score",min_value=0,max_value=target+5)

    if(target==0 and runs==0):
        inconsistent_input = 1

    part1 , part2 , part3 = st.columns(3)    
    with part1:
        overs = st.number_input("Overs Finished",min_value=0,max_value=20)

    with part2:
        balls = st.number_input("Balls",min_value=0,max_value=5)
        if(overs == 20 and balls !=0):
            ball_exceed = 1

    with part3:
        wickets = st.number_input("Wickets fallen",min_value=0,max_value=10)

    predict = st.button("Predict Ratio")

    if(predict and make_false == 1):
        st.error("Both Teams are Same ! Cannot Predict")

    elif(predict and inconsistent_input==1):
        st.error("Invalid Input ! Cannot Predict")

    elif(predict  and ball_exceed == 1):
        st.error("Balls faced Exceeded (>120) ! Cannot Predict")

    elif(predict and make_false==0):
        with st.spinner("Predicting..."):
            total_balls = (overs*6 + balls)
            runs_left = target - runs
            balls_left = 120 - total_balls
            wickets_left = 10 - wickets
            if(total_balls!=0):
                CRR = runs/total_balls * 6
            else:
                CRR = 0
            if(balls_left!=0):
                RRR = runs_left/balls_left * 6
            else:
                RRR = 0

            data = {"batting_team":[bat_team],"bowling_team":[bowl_team],"city":[city],"target":[target],"current_score":[runs],"runs_left":[runs_left],"balls_faced":[total_balls],"balls_left":[balls_left],"wickets_fallen":wickets,"wickets_left":[wickets_left],"CRR":[CRR],"RRR":[RRR]}
            df = pd.DataFrame(data)
            result = pipe_o.predict_proba(df)

            time.sleep(2)
            batting = result[0][1]
            bowling = result[0][0]
            if(runs == target - 1 and (wickets==10 or balls_left==0)):
                won = "Match Tie"
                st.markdown(f"<h3 style='text-align: center; color:grey'> {won} </h3>",unsafe_allow_html=True)
                getResult=2

            elif(target>runs and (wickets==10 or balls_left==0)):
                getResult = 1
                batting = 0
                bowling = 1
                won = bowl_team

            elif(target<=runs):
                getResult = 1
                batting = 1
                bowling = 0
                won = bat_team

            if(getResult == 0):
                st.markdown(f"<h3 style='text-align: center; color:{getColor(bat_team)}'>{bat_team} - %d %%</h3>"%(round(batting*100)),unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color:{getColor(bowl_team)}'>{bowl_team} - %d %%</h3>"%(round(bowling*100)),unsafe_allow_html=True)

            elif(getResult == 1):
                st.markdown(f"<h3 style='text-align: center; color:{getColor(won)}'> {won} Won</h3>",unsafe_allow_html=True)
                a,b,c = st.columns([25,50,25])
                with a:
                    st.write("")
                with b:
                    getImage(won)
                with c:
                    st.write("")
                st.markdown(f"<h3 style='text-align: center; color:{getColor(won)}'> {getTagline(won)}</h3>",unsafe_allow_html=True)

elif(select == "Dataset"):
    st.markdown("<h1 style='text-align: center; color:red'>DATASET</h1>"+"<h3 style='text-align: center; color:green'>ANALYZING RANDOM MATCH</h3>",unsafe_allow_html=True)
    
    rand = list(final_data["match_id"])
    match_id = random.choice(rand)
    new_df , tar , bat , ball , venue = match_prob(match_id,pipe_o)

    st.markdown(f"<h3 style='text-align: center;'><span style='color:{getColor(bat)}'>{bat}</span> &nbsp; V/S &nbsp; <span style='color:{getColor(ball)}'>{ball}</span> <br> Venue : <span style='color:#20f1f5;'>{venue}</span></h3>",unsafe_allow_html=True)

    st.markdown(f"<h4 style='text-align:center'> Target given by <span style='color:{getColor(ball)}'>{ball}</span> : {tar} </h4>",unsafe_allow_html=True)
    new_df.reset_index(inplace=True)

    st.markdown(f"<h4 style='text-align:center'><span style='color:{getColor(bat)};text-decoration:underline'>ScoreCard of {bat}</span></h4",unsafe_allow_html=True)

    new_df = new_df.drop(columns=["index"],axis=1)
    st.dataframe(new_df)

    st.markdown(f"<h4 style='text-align:center;font-style:italic'>PROBABILITY PLOT</h4>",unsafe_allow_html=True)
    plt.figure(figsize=(15,7))
    x_value = new_df['end_of_over']
    plt.plot(x_value,new_df['wicket_in_over'],color='yellow',lw=3,label="Fall of wicket")
    plt.plot(x_value,new_df['win_prob'],color='#00a65a',lw=3,label=f"Winning Ratio of {bat}")
    plt.plot(x_value,new_df['lose_prob'],color='red',lw=3,label=f"Losing Ratio of {bat}")
    plt.bar(x_value,new_df['runs_in_over'])
    plt.xticks(np.arange(1,21))
    plt.yticks(np.arange(0,101,20))
    plt.title('Target-' + str(tar))
    plt.xlabel("End Of The Over")
    plt.ylabel("Probabilty")
    plt.legend()
    st.pyplot()

else:
    st.markdown("<h2 style='text-align: center; color:red'>PROJECT DETAILS</h2>",unsafe_allow_html=True)

    st.markdown("<h3>Project Title : <span style='color:green;text-decoration:underline'>MATCH PREDICTOR AND ANALYZER</span></h3>",unsafe_allow_html=True)
    
    st.markdown("<h5>Project Description : <span style='color:blue'>This ML Model Can Predict the Winning Probability of a team based on Past Records using Logistic Regression. The Output Probability Depends on Various Factors like Opposition team , Venue , wickets_fallen , balls_left ,etc. This Model is not Much Accurate , its Accuracy Rate is around 80% , Still gives Output Very Close to Actual Values.</span></h5>",unsafe_allow_html=True)

    st.subheader("Library Used")
    st.markdown("<ul><li>PANDAS : for Data Cleaning and Analyzing The Datasets</li> <li>MATPLOTLIB : for Data Visualization</li><li>NUMPY : for Handling The Data in DataFrame and Manipulating it</li><li>SCIKIT-LEARN / SKLEARN : Machine Learning Library used for Building the Model and applying preprocessing to Data , Training and Testing The Data</li><li>STREAMLIT : for creating simple Web-Based GUI</li><li>PICKLE : For Pickling The Python Object Structure in Byte Stream</li></ul>",unsafe_allow_html=True)