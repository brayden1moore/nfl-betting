
''' This module gives users the ability to estimate the expected value of NFL bets based on a proprietary algorithm '''

# importing libraries
import os
import time
import math
import datetime
import requests
import html5lib
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from tqdm import tqdm
from os import listdir
from pathlib import Path
from urllib.error import HTTPError
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

from xgboost import XGBRegressor,XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, confusion_matrix

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# defining team names, columns and other important things
fullteamDict = {'Tampa Bay Buccaneers':'tam','Buffalo Bills':'buf','Washington Commanders':'was','Washington Redskins':'was','Arizona Cardinals':'crd','Los Angeles Rams':'ram','St. Louis Rams':'ram','Green Bay Packers':'gnb','Dallas Cowboys':'dal','Los Angeles Chargers':'sdg','Oakland Raiders':'rai','San Diego Chargers':'sdg','Baltimore Ravens':'rav', 'Tennessee Titans':'oti', 'Cincinnati Bengals':'cin', 'New Orleans Saints':'nor','Kansas City Chiefs':'kan','Cleveland Browns':'cle','Las Vegas Raiders':'rai','Minnesota Vikings':'min','Indianapolis Colts':'clt','New England Patriots':'nwe','San Francisco 49ers':'sfo','Seattle Seahawks':'sea','Pittsburgh Steelers':'pit','Denver Broncos':'den','Washington Football Team':'was','Atlanta Falcons':'atl','Philadelphia Eagles':'phi', 'Chicago Bears':'chi', 'Carolina Panthers':'car','Miami Dolphins':'mia', 'New York Giants':'nyg', 'Jacksonville Jaguars':'jax','Houston Texans':'htx', 'New York Jets':'nyj', 'Detroit Lions':'det'}
currentteamDict = {'Tampa Bay Buccaneers':'tam','Buffalo Bills':'buf','Washington Commanders':'was','Arizona Cardinals':'crd','Los Angeles Rams':'ram','Green Bay Packers':'gnb','Dallas Cowboys':'dal','Los Angeles Chargers':'sdg','Oakland Raiders':'rai','Baltimore Ravens':'rav', 'Tennessee Titans':'oti', 'Cincinnati Bengals':'cin', 'New Orleans Saints':'nor','Kansas City Chiefs':'kan','Cleveland Browns':'cle','Las Vegas Raiders':'rai','Minnesota Vikings':'min','Indianapolis Colts':'clt','New England Patriots':'nwe','San Francisco 49ers':'sfo','Seattle Seahawks':'sea','Pittsburgh Steelers':'pit','Denver Broncos':'den','Atlanta Falcons':'atl','Philadelphia Eagles':'phi', 'Chicago Bears':'chi', 'Carolina Panthers':'car','Miami Dolphins':'mia', 'New York Giants':'nyg', 'Jacksonville Jaguars':'jax','Houston Texans':'htx', 'New York Jets':'nyj', 'Detroit Lions':'det'}
reverseDict = dict((v, k) for k, v in fullteamDict.items())
teamDict = {'Buccaneers':'tam','Bills':'buf','Cardinals':'crd','Rams':'ram','Packers':'gnb','Cowboys':'dal','Chargers':'sdg','Ravens':'rav', 'Titans':'oti', 'Bengals':'cin', 'Saints':'nor', 'Chiefs':'kan','Browns':'cle', 'Raiders':'rai', 'Vikings':'min','Colts':'clt', 'Patriots':'nwe', '49ers':'sfo','Seahawks':'sea', 'Steelers':'pit','Broncos':'den', 'Washington':'was', 'Commanders':'was','Falcons':'atl','Eagles':'phi', 'Bears':'chi', 'Panthers':'car','Dolphins':'mia', 'Giants':'nyg','Jaguars':'jax','Texans':'htx', 'Jets':'nyj', 'Lions':'det'}
fullteamDict.update(teamDict)

teams = list(set(teamDict.values()))
cols = ['Week','Day','Date','Type','Outcome','Overtime','At','Name','TeamScore','OppScore','PCmp', 'PAtt','PYds', 'PTD', 'Int', 'Sk', 'SkYds', 'PY/A', 'PNY/A', 'Cmp%', 'Rate', 'RAtt','RYds', 'RY/A', 'RTD', 'FGM', 'FGA', 'XPM', 'XPA', 'Pnt', 'PuntYds', '3DConv','3DAtt', '4DConv', '4DAtt', 'ToP']
oppcols = [i+'_Opp' for i in cols] + ['Home_Opp','Away_Opp','Key']
dropcols = ['Day', 'Date', 'Type', 'Outcome', 'Overtime', 'Name','ToP', 'Home', 'Away','Week_Opp', 'Day_Opp','Date_Opp', 'Type_Opp', 'Outcome_Opp', 'Overtime_Opp', 'At_Opp','Name_Opp', 'TeamScore_Opp', 'OppScore_Opp','Cmp%','Cmp%_Opp','ToP_Opp', 'Home_Opp', 'Away_Opp']
#feats = list(set(['Home Odds Close','Away Odds Close','TeamScore_Home','OppScore_Home','PCmp_Home','PAtt_Home','PYds_Home','PTD_Home','Int_Home','Sk_Home','SkYds_Home','PY/A_Home','PNY/A_Home','Rate_Home','RAtt_Home','RYds_Home','RY/A_Home','RTD_Home','FGM_Home','FGA_Home','XPM_Home','XPA_Home','Pnt_Home','PuntYds_Home','3DConv_Home','3DAtt_Home','4DConv_Home','4DAtt_Home','PCmp_Opp_Home','PAtt_Opp_Home','PYds_Opp_Home','PTD_Opp_Home','Int_Opp_Home','Sk_Opp_Home','SkYds_Opp_Home','PY/A_Opp_Home','PNY/A_Opp_Home','Rate_Opp_Home','RAtt_Opp_Home','RYds_Opp_Home','RY/A_Opp_Home','RTD_Opp_Home','FGM_Opp_Home','FGA_Opp_Home','XPM_Opp_Home','XPA_Opp_Home','Pnt_Opp_Home','PuntYds_Opp_Home','3DConv_Opp_Home','3DAtt_Opp_Home','4DConv_Opp_Home','4DAtt_Opp_Home','TeamScore_Away','OppScore_Away','PCmp_Away','PAtt_Away','PYds_Away','PTD_Away','Int_Away','Sk_Away','SkYds_Away','PY/A_Away','PNY/A_Away','Rate_Away','RAtt_Away','RYds_Away','RY/A_Away','RTD_Away','FGM_Away','FGA_Away','XPM_Away','XPA_Away','Pnt_Away','PuntYds_Away','3DConv_Away','3DAtt_Away','4DConv_Away','4DAtt_Away','PCmp_Opp_Away','PAtt_Opp_Away','PYds_Opp_Away','PTD_Opp_Away','Int_Opp_Away','Sk_Opp_Away','SkYds_Opp_Away','PY/A_Opp_Away','PNY/A_Opp_Away','Rate_Opp_Away','RAtt_Opp_Away','RYds_Opp_Away','RY/A_Opp_Away','RTD_Opp_Away','FGM_Opp_Away','FGA_Opp_Away','XPM_Opp_Away','XPA_Opp_Away','Pnt_Opp_Away','PuntYds_Opp_Away','3DConv_Opp_Away','3DAtt_Opp_Away','4DConv_Opp_Away','4DAtt_Opp_Away']))
feats = ['Away Odds Close','elohome','eloaway_lag1','elohome_lag1','eloaway_lag2','elohome_lag2', 'TeamScore_Away', 'TeamScore_Home', 'Total Score Close','Int_Away', 'Int_Home', 'Int_Opp_Away', 'Int_Opp_Home', 'OppScore_Away', 'OppScore_Home', 'PYds_Away', 'PYds_Home', 'PYds_Opp_Away', 'PYds_Opp_Home', 'Pnt_Away', 'Pnt_Home', 'Pnt_Opp_Away', 'Pnt_Opp_Home', 'RYds_Away', 'RYds_Home', 'RYds_Opp_Away', 'RYds_Opp_Home', 'Sk_Away', 'Sk_Home', 'Sk_Opp_Away', 'Sk_Opp_Home']

def __request(team,year):  
    ''' Requests Pro Football Reference for game data '''  
    url = f'https://www.pro-football-reference.com/teams/{team}/{year}/gamelog/'
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36","X-Requested-With": "XMLHttpRequest"}
    r = requests.get(url)
    print('\r',end='')
    print(f'Status: {r.status_code}',end='')
    #clear_output()
    try:
        if len(pd.read_html(r.text)) > 2:
            opptable = 2
        else:
            opptable = 1
    except: pass
    if r.status_code != 200:
        waittime = int(r.headers["Retry-After"])
        for l in range(waittime+2):
            print('\r',end='')
            print(f'Time till next request: {waittime-l}, {round((waittime-l)/60,2)} mins',end='')
            time.sleep(1)
    return r.text, opptable

def __get(team,year,html,table):
    ''' Searches the HTML for a given table '''
    df = pd.read_html(html)[table]
    df.columns = cols
    df['Home'] = [fullteamDict[i] if j == '@' else team for i,j in zip(df['Name'],df['At'])]
    df['Away'] = [fullteamDict[i] if j != '@' else team for i,j in zip(df['Name'],df['At'])]
    df['Date'] = pd.to_datetime([i + ' ' + str(year+1) if 'Jan' in i else i + ' ' + str(year+1) if 'Feb' in i else i + ' ' + str(year) for i in df['Date']])
    df['Key'] = df['Date'].astype(str) + df['Home'] + df['Away']
    return df


def __complete(team,year):
    ''' Merges tables to include opponent data '''
    html, opptable = __request(team,year)
    df = __get(team,year,html,0)
    opp = __get(team,year,html,opptable)
    if opptable == 2:
        df = df.append(__get(team,year,html,1))
        opp = opp.append(__get(team,year,html,3))
    opp.columns = oppcols
    df.loc[:,8:-4] = df.iloc[:,8:-4].shift().rolling(4).mean()
    opp.loc[:,8:-4] = opp.iloc[:,8:-4].shift().rolling(4).mean()
    data = df.merge(opp,on='Key')
    return data

    
def __save(df, data_old):
    ''' Updates the data file '''
    home = df[df['At']!='@']
    away = df[df['At']=='@']
    home.columns = [i+'_Home' for i in df.columns]
    away.columns = [i+'_Away' for i in df.columns]

    data = home.merge(away,left_on='Key_Home',right_on='Key_Away')
    datafeats = [i for i in feats if i not in ['Away Odds Close','Home Odds Close','playoff']]
    data = pd.concat([data_old,data]).dropna(subset=datafeats)
    data.to_csv('data/nfl.csv')


def __refreshdata(minyr=2022,maxyr=2022):
    ''' Puts the above functions together to update the data up to a specified date '''
    
    data_old = pd.DataFrame()
    
    # getting data
    big = 1
    total = 1
    waittime = 1
    df = pd.DataFrame()
    for j in range(minyr,maxyr+1):
        lil = 1
        for i in teams:
            clear_output()
            time.sleep(2)
            print(i,j)
            print(f'Team {lil} / {len(teams)} - {lil/len(teams)}')
            print(f'Year {big} / {maxyr+1-minyr} - {big/(maxyr+1-minyr)}')
            print(f'Total {total} / {(maxyr+1-minyr)*len(teams)} - {total / ((maxyr+1-minyr)*len(teams))}')

            try:
                if i == teams[0] and j == minyr:
                    df = __complete(i,j)
                    __save(df,data_old)
                    
                else:
                    df = pd.concat([df,__complete(i,j)])
                    df.drop(columns=dropcols,inplace=True)
                    __save(df,data_old)

            except HTTPError as e:
                    waittime = int(e.headers['Retry-After'])

            lil+=1
            total+=1
        big+=1

    # merging with historical odds data

    data = pd.read_csv('data/nfl.csv',index_col=0)
    elodata = pd.read_csv('https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv')

    # loading gambling data
    spreadDf = pd.read_excel('http://www.aussportsbetting.com/historical_data/nfl.xlsx')
    spreadDf['Home Team'] = spreadDf['Home Team'].map(fullteamDict)
    spreadDf['Away Team'] = spreadDf['Away Team'].map(fullteamDict)
    spreadDf = spreadDf[['Date','Away Team','Home Team','Playoff Game?','Away Line Close','Away Odds Close','Home Odds Close','Total Score Close']]

    # standardizing team names
    elonames = {'hou':'htx','lac':'sdg','ari':'crd','sf':'sfo','oak':'rai','no':'nor','ind':'clt','bal':'rav','lar':'ram','tb':'tam','ten':'oti','gb':'gnb','kc':'kan','wsh':'was','ne':'nwe','atl':'atl','rav':'rav','buf':'buf','car':'car','chi':'chi','cin':'cin','cle':'cle','dal':'dal','den':'den','det':'det','jax':'jax','mia':'mia','min':'min','nyg':'nyg','nyj':'nyj','phi':'phi','pit':'pit','sea':'sea'}
    elodata['away'] = elodata['team2'].str.lower().map(elonames)
    elodata['home'] = elodata['team1'].str.lower().map(elonames)

    # adjusting elo values, cropping
    elodata['eloaway'] = elodata['elo2_pre'] + elodata['qb2_adj']
    elodata['elohome'] = elodata['elo1_pre'] + elodata['qb1_adj']
    elodata.rename(columns={'score2':'scoreaway','score1':'scorehome'},inplace=True)
    elodata = elodata[['date','season','away','home','eloaway','elohome','quality','scoreaway','scorehome','playoff']]

    # merging
    elodata['date'] = pd.to_datetime(elodata['date'])
    elodata = elodata[elodata['date']>'2000-01-01']
    elodata = elodata.merge(spreadDf, left_on=['date','away','home'],right_on=['Date','Away Team','Home Team'])

    # creating targets
    elodata['point_diff'] = elodata['scoreaway'] - elodata['scorehome']
    elodata['win'] = (elodata['point_diff']>0).astype(int)
    elodata['cover'] = (elodata['point_diff']>-elodata['Away Line Close']).astype(int)
    elodata['Key'] = elodata['date'].astype(str) + elodata['home'] + elodata['away']
    elodata['playoff'] = elodata['playoff'].map({'w':1,'d':1,'c':1,'s':1}).fillna(0)
    data = elodata.merge(data, left_on='Key',right_on='Key_Home').dropna(subset=feats).drop_duplicates(subset='Key')
    data['playoff'] = [0 if i == 0 else 1 for i in data['playoff']]
    data.to_csv('data/data.csv')

    __lag(df,inplace=True)


def __lag(df,inplace=False):  
    frame = pd.DataFrame()

    for t in teams:
        for s in df['season'].unique():
            temp = df.loc[((df['away']==t)|(df['home']==t)) & (df['season']==s)].sort_values('date')
            
            teamelo = [ea if t==a else eh for ea,eh,a in zip(temp['eloaway'],temp['elohome'],temp['away'])]

            temp['eloaway_lag1'] = [teamelo[teamelo.index(i)-1] if a==t and teamelo.index(i) > 0 else None for i,a in zip(temp['eloaway'],temp['away'])]
            temp['elohome_lag1'] = [teamelo[teamelo.index(i)-1] if h==t and teamelo.index(i) > 0 else None for i,h in zip(temp['elohome'],temp['home'])]

            temp['eloaway_lag2'] = [teamelo[teamelo.index(i)-2] if a==t and teamelo.index(i) > 1 else None for i,a in zip(temp['eloaway'],temp['away'])]
            temp['elohome_lag2'] = [teamelo[teamelo.index(i)-2] if h==t and teamelo.index(i) > 1 else None for i,h in zip(temp['elohome'],temp['home'])]

            frame = pd.concat([frame,temp])

    data = frame.groupby(['Key','date','season','away','home','Away Line Close','Away Odds Close','Home Odds Close','point_diff','win','cover','eloaway','elohome']).agg({'eloaway_lag1':'max','elohome_lag1':'max','eloaway_lag2':'max','elohome_lag2':'max'}).dropna().reset_index()
    datacols = list(data.columns)
    datacols.remove('Key')
    data = data.merge(df[[i for i in df.columns if i not in datacols]],left_on='Key',right_on='Key')

    if inplace:
        data.to_csv('data/datalagged.csv')
    else:
        return data


def __train(df,feats,iterations=100,eta=0.3,max_depth=2,booster='gbtree',num_round=10,testyears=[],purpose='live'):
    ''' Trains XGB Classifier and saves the model if it is more accurate than the one saved. '''

    acclist = [float(i.replace('.json','')) for i in listdir(f'models/{purpose}')]

    for i in tqdm(range(iterations)):
    
        if len(testyears)>0:
            X_train, X_test, y_train, y_test = df.loc[~df['season'].isin(testyears)][feats], df.loc[df['season'].isin(testyears)][feats], df.loc[~df['season'].isin(testyears)]['win'], df.loc[df['season'].isin(testyears)]['win']
        else:
            X_train, X_test, y_train, y_test = train_test_split(df[feats],df['win'])

        dtrain = xgb.DMatrix(X_train,label=y_train)
        dtest = xgb.DMatrix(X_test,label=y_test)

        param = {'booster':booster, 'max_depth':max_depth, 'objective':'multi:softprob','num_class':2, 'eta':eta, 'eval_metric':'auc'}

        bst = xgb.train(param, dtrain, num_round)
        y_pred = [np.argmax(i) for i in bst.predict(dtest)]

        acc = accuracy_score(y_test,y_pred)
        acclist.append(acc)

        if acc == max(acclist):
            bst.save_model(f'models/{purpose}/{round(acc,4)}.json')

    clear_output()
    print(f'Best model accuracy: {max(acclist)*100}%')
    print(f'This model accuracy: {round(acc,4)*100}%')
    xgb.plot_importance(bst)


def __predict(df,feats,purpose='live'):
    ''' Loads the saved model and returns prediction array for df input. '''
    
    modelpath = str(sorted(Path(f'models/{purpose}').iterdir(), key=os.path.getmtime)[-1])
    bst = xgb.Booster()
    bst.load_model(f'{modelpath}')
    dtest = xgb.DMatrix(df[feats],label=df['win'])

    return bst.predict(dtest)


def findev(df,feats,purpose='live',thresh=0):
    ''' Finds ev for given games and returns df with added proba, ev, val, and winnings columns. '''

    df['proba'] = [i[1] for i in __predict(df,feats,purpose)]
    df['ev'] = [(p*(a-1)-(1-p)) if (p*(a-1)-(1-p))>((1-p)*(h-1)-(p)) else -((1-p)*(h-1)-(p)) for p,h,a in zip(df['proba'],df['Home Odds Close'],df['Away Odds Close'])]
    df['val'] = [a-1 if w==1 else -(h-1) for w,h,a in zip(df['win'],df['Home Odds Close'],df['Away Odds Close'])]
    df['winnings'] = [abs(v) if np.sign(v)==np.sign(ev) else -1 for v,ev in zip(df['val'],df['ev'])]

    return df.loc[abs(df['ev'])>=thresh].reset_index(drop=True) 


#def bet(search):
#    tokens = search.split(' ')







