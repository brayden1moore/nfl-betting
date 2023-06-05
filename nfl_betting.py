
''' This module gives users the ability to estimate the expected value of NFL bets based on a proprietary algorithm '''

# importing libraries
import time
import math
import datetime
import requests
import html5lib
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

from tqdm import tqdm
from urllib.error import HTTPError
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
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
teams = list(set(fullteamDict.values()))
cols = ['Week','Day','Date','Type','Outcome','Overtime','At','Name','TeamScore','OppScore','PCmp', 'PAtt','PYds', 'PTD', 'Int', 'Sk', 'SkYds', 'PY/A', 'PNY/A', 'Cmp%', 'Rate', 'RAtt','RYds', 'RY/A', 'RTD', 'FGM', 'FGA', 'XPM', 'XPA', 'Pnt', 'PuntYds', '3DConv','3DAtt', '4DConv', '4DAtt', 'ToP']
oppcols = [i+'_Opp' for i in cols] + ['Home_Opp','Away_Opp','Key']
dropcols = ['Day', 'Date', 'Type', 'Outcome', 'Overtime', 'Name','ToP', 'Home', 'Away','Week_Opp', 'Day_Opp','Date_Opp', 'Type_Opp', 'Outcome_Opp', 'Overtime_Opp', 'At_Opp','Name_Opp', 'TeamScore_Opp', 'OppScore_Opp','Cmp%','Cmp%_Opp','ToP_Opp', 'Home_Opp', 'Away_Opp']
feats = list(set(['Away Odds Close','TeamScore_Home','OppScore_Home','PCmp_Home','PAtt_Home','PYds_Home','PTD_Home','Int_Home','Sk_Home','SkYds_Home','PY/A_Home','PNY/A_Home','Rate_Home','RAtt_Home','RYds_Home','RY/A_Home','RTD_Home','FGM_Home','FGA_Home','XPM_Home','XPA_Home','Pnt_Home','PuntYds_Home','3DConv_Home','3DAtt_Home','4DConv_Home','4DAtt_Home','PCmp_Opp_Home','PAtt_Opp_Home','PYds_Opp_Home','PTD_Opp_Home','Int_Opp_Home','Sk_Opp_Home','SkYds_Opp_Home','PY/A_Opp_Home','PNY/A_Opp_Home','Rate_Opp_Home','RAtt_Opp_Home','RYds_Opp_Home','RY/A_Opp_Home','RTD_Opp_Home','FGM_Opp_Home','FGA_Opp_Home','XPM_Opp_Home','XPA_Opp_Home','Pnt_Opp_Home','PuntYds_Opp_Home','3DConv_Opp_Home','3DAtt_Opp_Home','4DConv_Opp_Home','4DAtt_Opp_Home','TeamScore_Away','OppScore_Away','PCmp_Away','PAtt_Away','PYds_Away','PTD_Away','Int_Away','Sk_Away','SkYds_Away','PY/A_Away','PNY/A_Away','Rate_Away','RAtt_Away','RYds_Away','RY/A_Away','RTD_Away','FGM_Away','FGA_Away','XPM_Away','XPA_Away','Pnt_Away','PuntYds_Away','3DConv_Away','3DAtt_Away','4DConv_Away','4DAtt_Away','PCmp_Opp_Away','PAtt_Opp_Away','PYds_Opp_Away','PTD_Opp_Away','Int_Opp_Away','Sk_Opp_Away','SkYds_Opp_Away','PY/A_Opp_Away','PNY/A_Opp_Away','Rate_Opp_Away','RAtt_Opp_Away','RYds_Opp_Away','RY/A_Opp_Away','RTD_Opp_Away','FGM_Opp_Away','FGA_Opp_Away','XPM_Opp_Away','XPA_Opp_Away','Pnt_Opp_Away','PuntYds_Opp_Away','3DConv_Opp_Away','3DAtt_Opp_Away','4DConv_Opp_Away','4DAtt_Opp_Away']))
evthresh = 0.15

def request(team,year):  
    ''' Requests Pro Football Reference for game data '''  
    url = f'https://www.pro-football-reference.com/teams/{team}/{year}/gamelog/'
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36","X-Requested-With": "XMLHttpRequest"}
    r = requests.get(url, headers=header)
    print('\r',end='')
    print(f'Status: {r.status_code}',end='')
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

def get(team,year,html,table):
    ''' Searches the HTML for a given table '''
    df = pd.read_html(html)[table]
    df.columns = cols
    #df.loc[:,8:] = df.iloc[:,8:].shift().rolling(3).mean()
    df['Home'] = [fullteamDict[i] if j == '@' else team for i,j in zip(df['Name'],df['At'])]
    df['Away'] = [fullteamDict[i] if j != '@' else team for i,j in zip(df['Name'],df['At'])]
    df['Date'] = pd.to_datetime([i + ' ' + str(year+1) if 'Jan' in i else i + ' ' + str(year+1) if 'Feb' in i else i + ' ' + str(year) for i in df['Date']])
    df['Key'] = df['Date'].astype(str) + df['Home'] + df['Away']
    return df

def complete(team,year):
    ''' Merges tables to include opponent data '''
    html, opptable = request(team,year)
    df = get(team,year,html,0)
    opp = get(team,year,html,opptable)
    if opptable == 2:
        df = df.append(get(team,year,html,1))
        opp = opp.append(get(team,year,html,3))
    opp.columns = oppcols
    df.loc[:,8:-4] = df.iloc[:,8:-4].shift().rolling(3).mean()
    opp.loc[:,8:-4] = opp.iloc[:,8:-4].shift().rolling(3).mean()
    data = df.merge(opp,on='Key')
    return data
    
def save(df, data_old):
    ''' Updates the data file '''
    home = df[df['At']!='@']
    away = df[df['At']=='@']
    home.columns = [i+'_Home' for i in df.columns]
    away.columns = [i+'_Away' for i in df.columns]

    data = home.merge(away,left_on='Key_Home',right_on='Key_Away')
    datafeats = [i for i in feats if i not in ['Away Odds Close', 'playoff']]
    data = pd.concat([data_old,data]).dropna(subset=datafeats)
    data.to_csv('data/nfl.csv')

def refreshdata(maxdate):
    ''' Puts the above functions together to update the data up to a specified date '''
    data_old = pd.read_csv('data/nfl.csv',index_col=0)
    data_old['KeyDate'] = pd.to_datetime(data_old['Key_Home'].str[:10])
    dropteams = list(set([i[10:13] for i in data_old[data_old['KeyDate']>=maxdate]['Key_Home']]))
    dropteams += (list(set([i[13:] for i in data_old[data_old['KeyDate']>=maxdate]['Key_Home']])))
    getteams = [i for i in teams if i not in dropteams]
    maxyr = maxdate.year if maxdate.month > 3 else maxdate.year-1
    minyr = data_old['KeyDate'].max().year
    # getting data
    big = 1
    total = 1
    waittime = 1
    df = pd.DataFrame()
    for j in range(minyr,maxyr+1):
        lil = 1
        for i in getteams:
            clear_output()
            time.sleep(1)
            print(i,j)
            print(f'Team {lil} / {len(getteams)} - {lil/len(getteams)}')
            print(f'Year {big} / {maxyr+1-minyr} - {big/(maxyr+1-minyr)}')
            print(f'Total {total} / {(maxyr+1-minyr)*len(getteams)} - {total / ((maxyr+1-minyr)*len(getteams))}')

            try:
                if i == getteams[0] and j == minyr:
                    df = complete(i,j)
                    save(df,data_old)
                    
                else:
                    df = pd.concat([df,complete(i,j)])
                    df.drop(columns=dropcols,inplace=True)
                    save(df,data_old)

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
    
def interp(mislist,awayorhome):
    if awayorhome=='away':
        probdict = dict(zip(np.arange(-9,11)/10, [1,1,1,1,1,1,0.45,0.25,0.20,0.25,0.40,0.475,0.475,0.40,0.25,0,0,0,0,0]))
    if awayorhome=='home':
        probdict = dict(zip(np.arange(-9,11)/10,[0,0,0,0,0,0,0.35,0.50,0.55,0.50,0.35,0.25,0.25,0.35,0.55,1,1,1,1,1]))
    result = []
    for mis in mislist:
        one = probdict[round((mis*100 - (mis*100)%10)/100,2)]
        two = probdict[round(((mis+0.1)*100 - (mis*100)%10)/100,2)]
        bounds = np.array([one,two])
        low = bounds.min()
        high = bounds.max()
    
        ratio = ((mis*100)%10) / 10
        
        diff = high - low 
        prob = one + abs(diff)*ratio if one<two else one - abs(diff)*ratio
        prob = low + abs(diff)*ratio 
        result.append(prob)
    return result

def train():
    ''' Trains the model and predicts on the last 4 seasons '''
    # loading 
    data = pd.read_csv('data/data.csv',index_col=0)

    # splitting
    splitdata = data#[data['playoff']==0]
    traindata = splitdata[splitdata['season'] < 2019]
    testdata = splitdata[splitdata['season'] >= 2019].reset_index()

    # fitting
    scaler = MaxAbsScaler().fit(data[feats])

    X_train = traindata[feats]
    X_train = scaler.transform(traindata[feats])
    y_train = traindata['win']

    X_test = testdata[feats]
    X_test = scaler.transform(testdata[feats])
    y_test = testdata['win']

    clf = GradientBoostingClassifier(n_estimators=280, learning_rate=1, max_depth=1, random_state=0).fit(X_train, y_train)

    # adding columns
    testdata['pred_win'] = clf.predict(X_test)
    testdata['pred_proba'] = [i[1] for i in clf.predict_proba(X_test)]
    testdata['away_mult'] = testdata['Away Odds Close']
    testdata['home_mult'] = testdata['Home Odds Close']
    testdata['implied_proba'] = (1/testdata['Away Odds Close']) / (1/testdata['Away Odds Close'] + 1/testdata['Home Odds Close'])
    testdata['mismatch'] = testdata['pred_proba'] - testdata['implied_proba'] 
    testdata['away_proba'] = [i if a>=2 else 1-j for a,i,j in zip(testdata['away_mult'],interp(testdata['mismatch'],'away'),interp(testdata['mismatch'],'home'))]
    testdata['home_proba'] = 1-testdata['away_proba']
    testdata['home_ev'] = ((testdata['home_proba']) * (testdata['home_mult']-1)) - (1-testdata['home_proba'])
    testdata['away_ev'] = ((testdata['away_proba']) * (testdata['away_mult']-1)) - (1-testdata['away_proba'])
    testdata['bet_home'] = [1 if h>evthresh and h>a else None for h,a in zip(testdata['home_ev'],testdata['away_ev'])]
    testdata['bet_away'] = [1 if a>evthresh and a>h else None for a,h in zip(testdata['away_ev'],testdata['home_ev'])]
    testdata['correct'] = [1 if h==1 and w==0 or a==1 and w==1 else 0 for h,a,w in zip(testdata['bet_home'],testdata['bet_away'],testdata['win'])]
    testdata['mult'] = np.array([a-1 if ba==1 and c==1 else h-1 if bh==1 and c==1 else -1 if ba==1 and c==0 or bh==1 and c==0 else None for a,h,ba,bh,c in zip(testdata['away_mult'],testdata['home_mult'],testdata['bet_away'],testdata['bet_home'],testdata['correct'])])
    testdata['bet'] = [1 if h==1 or a==1 else None for h,a in zip(testdata['bet_away'],testdata['bet_home'])]
    testdata['ev'] = [h if bh == 1 else a if ba==1 else None for h,a,bh,ba in zip(testdata['home_ev'],testdata['away_ev'],testdata['bet_home'],testdata['bet_away'])]
    testdata['ev_binned'] = pd.cut(testdata['ev'],bins = (np.arange(-10,100)/10)+.05, labels = (np.arange(-10,100)/10)[:-1]+.05)
    testdata['underdogwin'] = [1 if w==1 and a>h or w==0 and a<h else 0 for w,a,h in zip(testdata['win'],testdata['away_mult'],testdata['home_mult'])]
    testdata['underdogmult'] = np.array([a-1 if a>h and u==1 else h-1 if a<h and u==1 else -1 for a,h,u in zip(testdata['away_mult'],testdata['home_mult'],testdata['underdogwin'])]).cumsum()
    testdata['favoritewin'] = [1 if w==1 and a<h or w==0 and a>h else 0 for w,a,h in zip(testdata['win'],testdata['away_mult'],testdata['home_mult'])]
    testdata['favoritemult'] = np.array([a-1 if a<h and u==1 else h-1 if a>h and u==1 else -1 for a,h,u in zip(testdata['away_mult'],testdata['home_mult'],testdata['favoritewin'])]).cumsum()

    # tallying
    bets = testdata.dropna(subset=['bet']).reset_index()
    bets['losses'] = np.array([i if i<0 else 0 for i in bets['mult']]).cumsum()
    bets['wins'] = np.array([i if i>0 else 0 for i in bets['mult']]).cumsum()
    bets['winnings'] = bets['wins'] + bets['losses']

    return clf,scaler,testdata,bets


def showmodel():
    ''' Displays test set performance versus other betting strategies '''
    clf, scaler, testdata, bets = train()
    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20,6))
    ev_agg = bets.groupby('ev_binned').agg({'mult':'mean','Key':'nunique'}).reset_index()
    ax[2].axhline(y=0,xmin=-1,xmax=10,color='lightgrey',linestyle='dotted')
    ax[2].axvline(x=0,ymin=-1,ymax=10,color='lightgrey',linestyle='dotted')
    ax[2].axvline(x=evthresh,ymin=-1,ymax=10,color='grey',linestyle='dotted')
    ax[2].set_title('Expected vs Realized Payoff')
    ax[2].scatter(ev_agg['ev_binned'],ev_agg['mult'],s=ev_agg['Key']*2,color='green')
    ax[2].plot(ev_agg['ev_binned'],ev_agg['ev_binned'],color='lightgrey',linewidth=1,linestyle='dotted')
    ax[2].set_yscale('symlog',linthresh=1)
    ax[2].set_xscale('symlog',linthresh=1)
    ax[2].set_xlabel('Expected Payoff')
    ax[2].set_ylabel('Actual Payoff')
    ax[2].xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[2].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[1].plot(testdata['underdogmult'],linewidth=4,color='green',label='Just Betting Underdogs')
    ax[1].plot(testdata['favoritemult'],linewidth=4,color='orange',label='Just Betting Favorites')
    ax[1].set_title('\nBad Strategies')
    ax[1].set_xlabel("Game")
    ax[1].legend()
    ax[1].set_ylabel("Payoff")
    x = np.array(range(len(bets['winnings']))).reshape(-1, 1)
    lin = LinearRegression(fit_intercept=False).fit(x,bets['winnings'])
    b = lin.intercept_
    y = lin.coef_[0] * x + b
    ax[0].plot(bets['winnings'],linewidth=4,label=f"Actual: {round(bets['winnings'][bets.shape[0]-1],2)}",color='green')
    ax[0].plot(x,y,label=f'Expected: {round(lin.coef_[0]*len(x),2)}',linewidth=3,color='orange')
    ax[0].set_ylabel("Payoff (Multiplier of Initial Bet)")
    ax[0].set_title('\nBetting With MARCI')
    ax[0].set_xlabel("Game")
    ax[0].text(len(bets['winnings'])*.5,bets['winnings'].min(),f'Accuracy: {round(bets["correct"].mean(),2)}')
    ax[0].legend()
    plt.show()

def getstats(away,home,date):
    ''' Gets necessary data for given matchup '''
    df = pd.read_csv('data/nfl.csv',index_col=0).drop_duplicates(subset=['Key_Home'])
    key = date + home + away
    df = df[df['Key_Home'] == key]

    df['awayname'] = reverseDict[away]
    df['homename'] = reverseDict[home]
    return df.reset_index()

def predict(awayodds,homeodds,data,scaler,clf):
    ''' Estimates EV '''
    awayname = data['awayname'][0].split(' ')[-1]
    homename = data['homename'][0].split(' ')[-1]

    awaymult = awayodds/100+1 if awayodds>0 else (100/(-1*awayodds)+1) 
    homemult = homeodds/100+1 if homeodds>0 else (100/(-1*homeodds)+1)
    dogmult = awaymult-1 if awaymult > homemult else homemult-1
    
    data['Away Odds Close'] = awaymult
    X = scaler.transform(data[feats])
    awaypred = clf.predict_proba(X)[0][1]

    awayimplied = (1/awaymult) / (1/awaymult + 1/homemult)
    mismatch = round(awaypred-awayimplied,3)

    dog = 'home' if homemult > awaymult else 'away'
    homeprob = interp([mismatch],'home')[0] if dog == 'home' else (1-interp([mismatch],'away')[0])
    awayprob = interp([mismatch],'away')[0] if dog == 'away' else (1-interp([mismatch],'home')[0])
    dogprob = homeprob if dog == 'home' else awayprob
    
    homeev = homeprob * (homemult-1) - (1 - homeprob)
    awayev = awayprob * (awaymult-1) - (1 - awayprob)
    maxev = homeev if homeev > awayev else awayev
    bestvalue = homename if homeev==maxev else awayname

    bet_home = 1 if homeev>evthresh and homeev>awayev else 0
    bet_away = 1 if awayev>evthresh and awayev>homeev else 0

    rec = awayname if bet_away == 1 else homename if bet_home == 1 else bestvalue
    recprob = awayprob if rec==awayname and dog=='home' else homeprob if rec==homename and dog=='away' else dogprob
    implied = awayimplied if bestvalue==awayname else 1-awayimplied
    return [awayprob, homeprob, awayev, homeev, implied, mismatch, rec, recprob, maxev]


def bet(games=[['Bills','Jets',-475,360,'2022-11-06']]):
    ''' Gives detailed output of predictions '''
    clf, scaler, testdata, bets = train()
    for i in tqdm(games): 
        away = teamDict[i[0]]
        home = teamDict[i[1]]
        i.append(predict(i[2],i[3],getstats(away,home,i[4]),scaler,clf))

    rec = pd.DataFrame(games)
    rec[list(range(5,5+len(rec.loc[0,5])))] = rec[5].apply(pd.Series)
    rec.columns =  ['Away','Home','Away Odds','Home Odds','Date','Away Win Prob','Home Win Prob','Away EV','Home EV','Their Implied Win Probability','Classifier Difference','Better Value Team','Their Win Probability','Their EV']
    #rec['Away'] = rec['Away'].map(reverseDict)
    #rec['Home'] = rec['Home'].map(reverseDict)
    rec['Bet On'] = [bvt if ev>evthresh else '' for bvt,ev in zip(rec['Better Value Team'],rec['Their EV'])]
    rec.insert(loc=7, column='|', value=['|' for i in range(rec.shape[0])])

    display(rec[['Date','Away','Home','|','Away Odds','Home Odds','Away Win Prob','Home Win Prob','Away EV','Home EV','|','Better Value Team','Their EV','Their Win Probability','Their Implied Win Probability','Classifier Difference','|','Bet On']].sort_values('Their EV',ascending=False))