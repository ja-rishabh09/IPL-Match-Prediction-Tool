import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

plt.style.use('fivethirtyeight')

'''
1. Define the problem statement.    
2. Collect the data.
3. Preprocess the data.
4. Train the model.
5. Evaluate the model.
6. Deploy the model.
7. Monitor the model.
8. Maintain the model.
9. Retire the model.
10. Repeat the process.
'''

matches=pd.read_csv('matches.csv')

# print(matches.shape)

'''
Take the data from the 'matches.csv' file and then using the data make a model that may take input such as which 
are the teams that are gonna play the match and then predict the winner based upon their stats .

'''
# print(matches.head(2))

'''
id,season,city,date,team1,team2,toss_winner,toss_decision,result,dl_applied,winner,win_by_runs,win_by_wickets,player_of_match,venue,umpire1,umpire2,umpire3

Removables ->

id(0),season(1) , city(2),date(3) , toss_winner(6),toss_decision(7),dl_applied(9),result(8),
win_by_runs(11),win_by_wickets(12),player_of_match(13),venue(14),umpire1(15),umpire2(15),umpire3(16)

Reqd ranges ->
4-5 , 10
'''

#PREPROCESSING -> Just keeping the column numbers 4,5,10
matches=matches.iloc[:,[2,4,5,10]]
print(matches.head(2))
'''
We have extracted the team1 , team2 , winner .
'''
wins=matches['winner'].value_counts()
# print(wins)

# wins.plot(kind='bar')
# plt.show()

'''Here we have got an overview of how our data looks on scale .'''

'''
We might use logistic regression as the target 'winner' is categorical.
'''

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

matches=matches.dropna(subset=['team1','team2','city','winner'])
#Clarifying our data ->

'''
We would now interpret the data mathematically ->
If team1 wins it is 1 , if team2 wins it is 0.
'''

matches=matches[(matches['winner']==matches['team1']) | (matches['winner']==matches['team2']) ].copy()

matches["target"]=(matches['winner']==matches['team1']).astype(int)
#This returns a true or a false , true if winner is team 1 , else it is false 

'''
Features and target ->

'''

from aliases import team_aliases, city_aliases
#Importing the team and city dictionaries .

x=matches[['team1','team2','city']]
y=matches['target']

'''
Encoding - OneHotEncoder ->
'''
preprocessor=ColumnTransformer([("encoder",OneHotEncoder(handle_unknown="ignore"),['team1','team2','city'])])
#This would just turn the values (strings) into the mathematical values such as to distinguish between them .
#OneHotEncoder creates a binary column for every name so as to recognise it binarily.

#MODEL ->
model=Pipeline([('preprocessor',preprocessor),('clf',LogisticRegression(max_iter=1000))])

'''
Pipeline helps to perform multiple tasks at once such as preprocessing (which includes binaryfication of 
the values), then we use logistic regression to train the model .
max_iter helps in improving the bias towards the weights that the Logistic regression intakes so as 
to filter out its biases .
1000 max_iter ki max limit di hai humne , agr usse kam iterations me value converge hoti dikhegi model ko 
to model utne ke basis se hi answer krdega .
'''

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

'''
Giving 20 percent of the data to train and the rest of the 80 percent to train.
Stratify=y tries to breakdown the test and the train datasets into equal proportions , such as 
ki proportion of winners (team1:team2) equal or closest rahe.
Isse model better banta hai .
'''

#Training ->
model.fit(x_train,y_train)

#Prediction->
y_predict=model.predict(x_test)

#Taking input from the user ->
team1=input("Enter team 1: ").strip().upper()
team2=input("Enter team 2: ").strip().upper()
city=input("Enter the city: ").strip().upper()

team1=team_aliases.get(team1,team1)
team2=team_aliases.get(team2,team2)
city=city_aliases.get(city,city)

#Creating Dataframe ->
new_match=pd.DataFrame({
    "team1":[team1],
    "team2":[team2],
    "city":[city]
})

#Prediction model->
prediction=model.predict(new_match)
#This return an array ->

probabilty=model.predict_proba(new_match)
'''
If team1 wins , the model returns 1 , else if the team2 wins the model returns 0.
'''

#Output ->
if prediction[0]==1:
    print("Predicted Winner: ",team1)
else:
    print("Predicted Winner: ",team2)

print("The winning percentage for Team 1 is: ",probabilty[0][1]*100,"%")
print("The winning percentage for Team 2 is: ",probabilty[0][0]*100,"%")

print("Accuracy: ",accuracy_score(y_test,y_predict))