import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import statsmodels.api as sm
#import arviz as az
#import pymc as pm
#import nutpie
#import pytensor
import pickle
import math
import matplotlib.pyplot as plt
import time
import concurrent.futures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def split(name):
    formatted = pd.read_csv("./intermediates/regression_formatted/"+name+".csv").dropna()
    seasons = ""
    for yr in range(1997, 2014):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    train = formatted[formatted["season"].str.contains(seasons[:-1])]

    seasons = ""
    for yr in range(2014, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    test = formatted[formatted["season"].str.contains(seasons[:-1])]

    train.to_csv('./intermediates/regression_formatted/'+name+'_train.csv', index=False)
    test.to_csv('./intermediates/regression_formatted/'+name+'_test.csv', index=False)

    

def linear_normal_check(name):
    formatted = pd.read_csv("./intermediates/regression_formatted/"+name+"_train.csv")
    for col in formatted.columns:
        if (col not in ['game_id','date','season','off_team_id','def_team_id','home_off','actual_eff']):
            plt.hist(formatted[col],bins=100)
            plt.title(col)
            plt.show()
            plt.scatter(formatted[col], formatted['actual_eff'],alpha=0.3)
            plt.title(col)
            plt.show()

def OLS_summary(name):
    if (name in ['first','second']):
        features = ['last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        features = ['last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off', 'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def',
                    'cur_pred_3pt_pct_off_last_5','cur_pred_2pt_pct_off_last_5','cur_pred_to_pct_off_last_5', 'cur_pred_ftrXftp_off_last_5','cur_pred_ftrXftp_def_last_5',
                    'cur_pred_3pt_pct_off_last_10','cur_pred_2pt_pct_off_last_10','cur_pred_to_pct_off_last_10', 'cur_pred_ftrXftp_off_last_10','cur_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'_train.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    df = df.merge(games,how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['last_pred_rtg'] = pred_eff
    df["intercept"] = 1
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")]

    #remove outliers
    for x in features:
        print (x)
        print (len(df.index))
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]
        print (len(df.index))
    
    if (name in ['first','second']):
        results = sm.OLS(df['actual_eff'], df[features+['intercept','home_off']]).fit()
    elif (name in ['third']):
        results = sm.OLS(df['actual_eff'], df[features+['intercept','home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']]).fit()        
    print (results.summary())

def OLS_first():
    #Test with last pred, lazy rn
    df = pd.read_csv('./intermediates/regression_formatted/first_train.csv').dropna()
    reg = LinearRegression().fit(df[['home_off','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']], df['actual_eff'])
    train_pred = reg.predict(df[['home_off','last_pred_3pt_pct_off','last_pred_2pt_pct_off','last_pred_to_pct_off',
                                           'last_pred_ftrXftp_off','last_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    test = pd.read_csv('./intermediates/regression_formatted/first_test.csv').dropna()
    test_pred = reg.predict(test[['home_off','last_pred_3pt_pct_off','last_pred_2pt_pct_off','last_pred_to_pct_off',
                                           'last_pred_ftrXftp_off','last_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/first_eff_ols.csv',index=False)

    table = []
    for gid in test['game_id'].unique():
        cur_f = {}
        cur_game = test.loc[test['game_id']==gid,].reset_index(drop=True)
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)
    print (len(table))
    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final.to_csv('./predictions/betting/pre_bet/first.csv', index=False)

def OLS_w_best_eff(name,l2_weight=0):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'_train.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    df = df.merge(games,how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")]

    #use cur lineup to train, last lineup to test
    for col in df.columns:
        if ('cur_pred' in col):
            df = df.rename(columns={col:'reg_pred'+col.split('cur_pred')[1]})

    #remove outliers
    for x in cont_features:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]

    scaler = StandardScaler()
    scaler.fit(df[cont_features])
    scaler.transform(df[cont_features])

    if (name in ['first','second']):
        if (l2_weight == 0):
            reg = LinearRegression().fit(df[cont_features+['home_off']], df['actual_eff'])
        else:
            reg = Ridge(alpha=l2_weight).fit(df[cont_features+['home_off']], df['actual_eff'])
        train_pred = reg.predict(df[cont_features+['home_off']])
    elif (name in ['third']):
        if (l2_weight == 0):
            reg = LinearRegression().fit(df[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], df['actual_eff'])
        else:
            reg = Ridge(alpha=l2_weight).fit(df[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], df['actual_eff'])
        train_pred = reg.predict(df[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])
    

    test = pd.read_csv('./intermediates/regression_formatted/'+name+'_test.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    test = test.merge(games[['game_id','game_type']],how='left',on='game_id')
    
    pred_eff = []
    for gid in test['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(test.loc[test['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    test['reg_pred_rtg'] = pred_eff
    test = test.dropna()

    #exclude playoffs
    test = test[test["game_type"].str.contains("Regular Season")]

    #use cur lineup to train, last lineup to test
    for col in test.columns:
        if ('last_pred' in col):
            test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})

    #remove outliers based on train data
    for x in cont_features:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]
    scaler.transform(test[cont_features])

    if (name in ['first','second']):
        test_pred = reg.predict(test[cont_features+['home_off']])
    elif (name in ['third']):
        test_pred = reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])
        
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/'+name+'_eff_w_best_eff_ols.csv',index=False)

    table = []
    for gid in test['game_id'].unique():
        cur_f = {}
        cur_game = test.loc[test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        if (cur_game.at[0,'significant_player_removed'] == 1 or cur_game.at[1,'significant_player_removed'] == 1):
            cur_f['significant_player_removed'] = 1
        else:
            cur_f['significant_player_removed'] = 0
        if (cur_game.at[0,'significant_player_added'] == 1 or cur_game.at[1,'significant_player_added'] == 1):
            cur_f['significant_player_added'] = 1
        else:
            cur_f['significant_player_added'] = 0
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_w_best_eff_reg_season_last_lu_test.csv', index=False)

def RF_first_w_best_eff():
    #Test with last pred, lazy rn
    df = pd.read_csv('./intermediates/regression_formatted/first_train.csv').dropna()

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_b2b.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['last_pred_rtg'] = pred_eff
    df = df.dropna()

    reg = RandomForestRegressor(random_state=1,ccp_alpha=0.07).fit(df[['home_off','last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']], df['actual_eff'])
    train_pred = reg.predict(df[['home_off','last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    

    test = pd.read_csv('./intermediates/regression_formatted/first_test.csv').dropna()
    
    pred_eff = []
    for gid in test['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(test.loc[test['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    test['last_pred_rtg'] = pred_eff
    test = test.dropna()

    test_pred = reg.predict(test[['home_off','last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/first_eff_w_best_eff_rf.csv',index=False)

    table = []
    for gid in test['game_id'].unique():
        cur_f = {}
        cur_game = test.loc[test['game_id']==gid,].reset_index(drop=True)
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final.to_csv('./predictions/betting/pre_bet/rf_first_w_best_eff.csv', index=False)

#PYMC Has mem leak so have to wrap it in multiprocessing
def run_pymc(priors, cur):
    with pm.Model() as pymc_model:
        coefs = pm.Normal('coefs', mu=priors[0], sigma=priors[1], shape=len(priors[0]))

        mu = (np.array(cur['home_off'])*coefs[0] + np.array(cur['last_pred_rtg'])*coefs[1] + np.array(cur['cur_pred_3pt_pct_off'])*coefs[2] + 
                np.array(cur['cur_pred_2pt_pct_off'])*coefs[3] + np.array(cur['cur_pred_to_pct_off'])*coefs[4] + np.array(cur['cur_pred_ftrXftp_off'])*coefs[5] + 
                np.array(cur['cur_pred_ftrXftp_def'])*coefs[6] + np.array(cur['pred_3pt_pct_def'])*coefs[7] + np.array(cur['pred_2pt_pct_def'])*coefs[8] + 
                np.array(cur['pred_to_pct_def'])*coefs[9] + np.array(cur['pred_oreb_pct_off'])*coefs[10] + np.array(cur['pred_oreb_pct_def'])*coefs[11] + coefs[12])
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=1, observed=cur['actual_eff'])
        
    compiled_model = nutpie.compile_pymc_model(pymc_model)

    trace_pymc = nutpie.sample(compiled_model, draws = 1000, seed=1, chains=10, cores=8, save_warmup=False, progress_bar=False)

    return (trace_pymc)

def pymc_regression_first(fatten):
    df = pd.read_csv('./intermediates/regression_formatted/first.csv').dropna()

    seasons = ""
    for yr in range(2014, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_b2b.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['last_pred_rtg'] = pred_eff
    df = df.dropna()

    #init to OLS from train set
    priors = [[0.6811,0.8477,13.8862,34.2249,-46.1679,16.6051,13.0749,-15.3312,35.2256,-38.3688,4.4276,13.4116,1.699],[]]
    for i in range(len(priors[0])):
        priors[1].append(0.1*abs(priors[0][i]))

    pred = []
    
    count = 1
    for date in tqdm(df['date'].unique()):
        cur = df.loc[df['date']==date,].reset_index(drop=True)
        if (count%500==0):
            print (priors)
        count += 1
        
        for index, row in cur.iterrows():
            pred.append(row['home_off']*priors[0][0] + row['last_pred_rtg']*priors[0][1] + row['cur_pred_3pt_pct_off']*priors[0][2] + 
                  row['cur_pred_2pt_pct_off']*priors[0][3] + row['cur_pred_to_pct_off']*priors[0][4] + row['cur_pred_ftrXftp_off']*priors[0][5] + 
                  row['cur_pred_ftrXftp_def']*priors[0][6] + row['pred_3pt_pct_def']*priors[0][7] + row['pred_2pt_pct_def']*priors[0][8] + 
                  row['pred_to_pct_def']*priors[0][9] + row['pred_oreb_pct_off']*priors[0][10] + row['pred_oreb_pct_def']*priors[0][11] + priors[0][12])
        
        #Update PYMC
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_pymc, priors, cur)
            trace_pymc = future.result()
            # print (az.summary(trace_pymc))
            # return

        for index, row in az.summary(trace_pymc).iterrows():
            priors[0][int(index.split('[')[1].split(']')[0])] = row['mean']
            priors[1][int(index.split('[')[1].split(']')[0])] = row['sd'] * fatten
    

    df['pred_eff'] = pred

    df.to_csv('./predictions/latent/first_eff_w_best_eff_bhm_' + str(fatten) + '.csv',index=False)

    seasons = ""
    for yr in range(2014, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    table = []
    for gid in df['game_id'].unique():
        cur_f = {}
        cur_game = df.loc[df['game_id']==gid,].reset_index(drop=True)
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(df['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final.to_csv('./predictions/betting/pre_bet/bhm_first_w_best_eff_' + str(fatten) + '.csv', index=False)

#Best Params first: {'eta': 0.15, 'max_depth': 1, 'min_child_weight': 0.1, 'gamma': 3000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_parallel_tree': 1}
#Best Params third: {'eta': 0.15, 'max_depth': 1, 'min_child_weight': 100, 'gamma': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_parallel_tree': 1}
#interesting params third: {'eta': 0.15, 'max_depth': 2, 'min_child_weight': 10, 'gamma': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_parallel_tree': 1} {'eta': 0.15, 'max_depth': 5, 'min_child_weight': 600, 'gamma': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_parallel_tree': 1}
def xgboost_regression(name):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'_train.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")]

    #use cur lineup to train, last lineup to test
    for col in df.columns:
        if ('cur_' in col):
            df = df.rename(columns={col:'reg_'+col.split('cur_')[1]})

    #remove outliers
    for x in cont_features:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]
    

    reg = xgb.XGBRegressor(eta=0.15,max_depth=5,min_child_weight=600,gamma=1000,subsample=0.8,colsample_bytree=0.8)
    if (name in ['first','second']):
        reg.fit(df[cont_features+['home_off']], df['actual_eff'])
        train_pred = reg.predict(df[cont_features+['home_off']])
    elif (name in ['third']):
        reg.fit(df[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], df['actual_eff'])
        train_pred = reg.predict(df[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])
    

    test = pd.read_csv('./intermediates/regression_formatted/'+name+'_test.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    test = test.merge(games[['game_id','game_type']],how='left',on='game_id')
    
    pred_eff = []
    for gid in test['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(test.loc[test['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    test['last_pred_rtg'] = pred_eff
    test = test.dropna()

    #exclude playoffs
    test = test[test["game_type"].str.contains("Regular Season")]

    #use cur lineup to train, last lineup to test
    for col in test.columns:
        if ('last_pred' in col):
            test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})

    #remove outliers based on train data
    for x in cont_features:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]

    if (name in ['first','second']):
        test_pred = reg.predict(test[cont_features+['home_off']])
    elif (name in ['third']):
        test_pred = reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])
    
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/'+name+'_xgb.csv',index=False)

    table = []
    for gid in test['game_id'].unique():
        cur_f = {}
        cur_game = test.loc[test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_xgb_interesting_2.csv', index=False)

def tune_xgboost_sub_call(name,params):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'_train.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")]

    seasons = ""
    for yr in range(1997, 2010):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    train = df[df["season"].str.contains(seasons[:-1])]

    seasons = ""
    for yr in range(2010, 2014):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    test = df[df["season"].str.contains(seasons[:-1])]

    #use cur lineup to train, last lineup to test
    for col in train.columns:
        if ('cur_pred' in col):
            train = train.rename(columns={col:'reg_pred'+col.split('cur_pred')[1]})

    #remove outliers
    for x in cont_features:
        q75, q25 = np.percentile(train[x], [75 ,25])
        iqr = q75 - q25
        train = train[~(train[x] <= q25 - 3*iqr)]
        train = train[~(train[x] >= q75 + 3*iqr)]

    reg = xgb.XGBRegressor(seed=0,eta=params['eta'],max_depth=params['max_depth'],min_child_weight=params['min_child_weight'],gamma=params['gamma'],subsample=params['subsample'],
                           colsample_bytree=params['colsample_bytree'])
    if (name in ['first','second']):
        reg.fit(train[cont_features+['home_off']], train['actual_eff'])
        train_pred = reg.predict(train[cont_features+['home_off']])
    elif (name in ['third']):
        reg.fit(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], train['actual_eff'])
        train_pred = reg.predict(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])

    #use cur lineup to train, last lineup to test
    for col in test.columns:
        if ('last_pred' in col):
            test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})

    #remove outliers based on train data
    for x in cont_features:
        q75, q25 = np.percentile(train[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]

    if (name in ['first','second']):
        test_pred = reg.predict(test[cont_features+['home_off']])
    elif (name in ['third']):
        test_pred = reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']])

    print (params)
    print ("MSE of Training Set",mean_squared_error(train['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))
    return (mean_squared_error(test['actual_eff'],test_pred))

def tune_xgboost(name):
    params = {'eta':0.15,'max_depth':6,'min_child_weight':1,'gamma':0,'subsample':0.8,'colsample_bytree':0.8,'num_parallel_tree':1}

    best = 99999
    best_md = -1
    best_mcw = -1
    best_g = -1
    for md in range(1,10):
        params['max_depth'] = md
        for mcw in [0.1,1,10,50,100,300,600,1000,3000,10000]:
            params['min_child_weight'] = mcw
            for g in [1,10,50,100,300,600,1000,3000,10000]:
                params['gamma'] = g
                mse = tune_xgboost_sub_call(name, params)
                if (mse < best):
                    best = mse
                    best_md = md
                    best_mcw = mcw
                    best_g = g
    params['max_depth'] = best_md
    params['min_child_weight'] = best_mcw
    params['gamma'] = best_g

    print ("--------------------------------")
    print ('Best Params:',params)

#for finding best window size and analyzing changing coefs
def rolling_OLS_train_only(name,window_size):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'_train.csv').dropna()
    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type','game_date']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")].reset_index(drop=True)

    #doing this just to simplify here
    for col in df.columns:
        if ('last_pred' in col):
            df = df.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})
    
    scaler = StandardScaler()
    df[cont_features+['home_off']] = scaler.fit_transform(df[cont_features+['home_off']])

    all_pred = []
    train_bounds = [0,window_size]
    next_test_start = window_size + 20

    tracker = {'date':[],'coefs':[]}
    for f in cont_features:
        tracker['coefs'].append([])
    tracker['coefs'].append([])
    

    while(next_test_start < len(df.index)):
        train = df.iloc[train_bounds[0]:train_bounds[1]]
        cur_date = df.at[next_test_start,'game_date']
        test = df.loc[df['game_date']==cur_date,]
        if (train_bounds[0] == 0):
            all_test_start = test.index[0]
        
        if (name in ['first','second']):
            reg = LinearRegression().fit(train[cont_features+['home_off']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off']]))
        elif (name in ['third']):
            reg = LinearRegression().fit(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']]))
        
        tracker['date'].append(cur_date)
        for i in range(len(reg.coef_)):
            tracker['coefs'][i].append(reg.coef_[i])
        
        train_bounds = [test.index[-1]-window_size,test.index[-1]]
        next_test_start = test.index[-1] + 1


    all_test = df.iloc[all_test_start:].copy(deep=True)
    all_test['pred_eff'] = all_pred

    print ("Window Size:", window_size)
    print ("MSE of Test Set",mean_squared_error(all_test['actual_eff'],all_pred))


    table = []
    for gid in all_test['game_id'].unique():
        cur_f = {}
        cur_game = all_test.loc[all_test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(all_test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_rolling_reg.csv', index=False)

    for i in range(len(tracker['coefs'])):
        plt.plot(tracker['date'], tracker['coefs'][i], label=(cont_features+['home_off'])[i])
    plt.figtext(0.8,0.82,"Squared Error: "+str(mean_squared_error(all_test['actual_eff'],all_pred).round(2)), fontsize=10)
    plt.legend(loc = 'upper left',fontsize = 'xx-small')
    plt.xlabel("Date")
    plt.ylabel("Coefficient")
    plt.title("Rolling Regression with Window Size of " + str(window_size))
    plt.gcf().set_size_inches(16,9)
    plt.savefig('./figures/rolling_reg/'+str(window_size)+'.png', dpi=100)
    plt.close()

def rolling_OLS(name,window_size):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'.csv').dropna()

    seasons = ""
    for yr in range(1997, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type','game_date']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")].reset_index(drop=True)

    all_pred = []
    next_test_start = df.loc[df['season']=='2014-15',].index[0]
    train_bounds = [next_test_start-1-window_size,next_test_start-1]
    iter_one = True

    while(next_test_start < len(df.index)):
        train = df.iloc[train_bounds[0]:train_bounds[1]]
        cur_date = df.at[next_test_start,'game_date']
        test = df.loc[df['game_date']==cur_date,]
        if (iter_one):
            all_test_start = test.index[0]
            iter_one = False

        for col in train.columns:
            if ('cur_pred' in col):
                train = train.rename(columns={col:'reg_pred'+col.split('cur_pred')[1]})
        for col in test.columns:
            if ('last_pred' in col):
                test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})
        
        if (name in ['first','second']):
            reg = LinearRegression().fit(train[cont_features+['home_off']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off']]))
        elif (name in ['third']):
            reg = LinearRegression().fit(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']]))
        
        train_bounds = [test.index[-1]-window_size,test.index[-1]]
        next_test_start = test.index[-1] + 1

    all_test = df.iloc[all_test_start:].copy(deep=True)
    all_test['pred_eff'] = all_pred

    print ("MSE of Test Set",mean_squared_error(all_test['actual_eff'],all_pred))


    table = []
    for gid in all_test['game_id'].unique():
        cur_f = {}
        cur_game = all_test.loc[all_test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(all_test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_rolling_reg.csv', index=False)

def expanding_OLS(name):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'.csv').dropna()

    seasons = ""
    for yr in range(1997, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type','game_date']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")].reset_index(drop=True)

    all_pred = []
    next_test_start = df.loc[df['season']=='2014-15',].index[0]
    train_bounds = [0,next_test_start-1]
    iter_one = True
    while(next_test_start < len(df.index)):
        train = df.iloc[train_bounds[0]:train_bounds[1]]
        cur_date = df.at[next_test_start,'game_date']
        test = df.loc[df['game_date']==cur_date,]
        if (iter_one):
            all_test_start = test.index[0]
            iter_one = False

        for col in train.columns:
            if ('cur_pred' in col):
                train = train.rename(columns={col:'reg_pred'+col.split('cur_pred')[1]})
        for col in test.columns:
            if ('last_pred' in col):
                test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})
        
        if (name in ['first','second']):
            reg = LinearRegression().fit(train[cont_features+['home_off']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off']]))
        elif (name in ['third']):
            reg = LinearRegression().fit(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']]))
        
        train_bounds = [0,test.index[-1]]
        next_test_start = test.index[-1] + 1


    all_test = df.iloc[all_test_start:].copy(deep=True)
    all_test['pred_eff'] = all_pred

    print ("MSE of Test Set",mean_squared_error(all_test['actual_eff'],all_pred))


    table = []
    for gid in all_test['game_id'].unique():
        cur_f = {}
        cur_game = all_test.loc[all_test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(all_test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_expanding_reg.csv', index=False)

def expanding_xgb(name):
    if (name in ['first','second']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']
    elif (name in ['third']):
        cont_features = ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off', 'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def',
                    'reg_pred_3pt_pct_off_last_5','reg_pred_2pt_pct_off_last_5','reg_pred_to_pct_off_last_5', 'reg_pred_ftrXftp_off_last_5','reg_pred_ftrXftp_def_last_5',
                    'reg_pred_3pt_pct_off_last_10','reg_pred_2pt_pct_off_last_10','reg_pred_to_pct_off_last_10', 'reg_pred_ftrXftp_off_last_10','reg_pred_ftrXftp_def_last_10',
                    'pred_3pt_pct_def','pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def',
                    'pred_3pt_pct_def_last_5','pred_2pt_pct_def_last_5','pred_to_pct_def_last_5','pred_oreb_pct_off_last_5','pred_oreb_pct_def_last_5',
                    'pred_3pt_pct_def_last_10','pred_2pt_pct_def_last_10','pred_to_pct_def_last_10','pred_oreb_pct_off_last_10','pred_oreb_pct_def_last_10',
                    'pred_3pt_pct_matchup_form','pred_2pt_pct_matchup_form','pred_to_pct_matchup_form','pred_oreb_pct_matchup_form','pred_ftr_matchup_form','pred_rtg_matchup_form']
        
    
    df = pd.read_csv('./intermediates/regression_formatted/'+name+'.csv').dropna()

    seasons = ""
    for yr in range(1997, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    games = pd.read_csv('./database/games.csv')
    df = df.merge(games[['game_id','game_type','game_date']],how='left',on='game_id')

    eff_pred = pd.read_csv('./predictions/betting/pre_bet/player_eff_bhm_usg.csv')
    pred_eff = []
    for gid in df['game_id'].unique():
        cur_game = eff_pred.loc[eff_pred['game_id']==gid,].reset_index(drop=True)
        if (len(df.loc[df['game_id']==gid,].index)!=2):
            pred_eff.append(np.nan)
            continue
        pred_eff.append(cur_game.at[0,'h_rtg_pred'])
        pred_eff.append(cur_game.at[0,'a_rtg_pred'])
    df['reg_pred_rtg'] = pred_eff
    df = df.dropna()

    #exclude playoffs
    df = df[df["game_type"].str.contains("Regular Season")].reset_index(drop=True)

    all_pred = []
    next_test_start = df.loc[df['season']=='2014-15',].index[0]
    train_bounds = [0,next_test_start-1]
    iter_one = True
    while(next_test_start < len(df.index)):
        print (next_test_start)
        train = df.iloc[train_bounds[0]:train_bounds[1]]
        cur_date = df.at[next_test_start,'game_date']
        test = df.loc[df['game_date']==cur_date,]
        if (iter_one):
            all_test_start = test.index[0]
            iter_one = False

        for col in train.columns:
            if ('cur_pred' in col):
                train = train.rename(columns={col:'reg_pred'+col.split('cur_pred')[1]})
        for col in test.columns:
            if ('last_pred' in col):
                test = test.rename(columns={col:'reg_pred'+col.split('last_pred')[1]})
        
        reg = xgb.XGBRegressor(eta=0.15,max_depth=2,min_child_weight=10,gamma=1000,subsample=0.8,colsample_bytree=0.8)
        if (name in ['first','second']):
            reg.fit(train[cont_features+['home_off']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off']]))
        elif (name in ['third']):
            reg.fit(train[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']], train['actual_eff'])
            all_pred += list(reg.predict(test[cont_features+['home_off','off_b2b_leg_1','off_b2b_leg_2','def_b2b_leg_1','def_b2b_leg_2']]))
        
        train_bounds = [0,test.index[-1]]
        next_test_start = test.index[-1] + 1


    all_test = df.iloc[all_test_start:].copy(deep=True)
    all_test['pred_eff'] = all_pred

    print ("MSE of Test Set",mean_squared_error(all_test['actual_eff'],all_pred))


    table = []
    for gid in all_test['game_id'].unique():
        cur_f = {}
        cur_game = all_test.loc[all_test['game_id']==gid,].reset_index(drop=True)
        if (len(cur_game.index) < 2):
            table.append(cur_f)
            continue
        cur_f['game_id'] = gid
        cur_f['date'] = cur_game.at[0,'date']
        cur_f['season'] = cur_game.at[0,'season']
        cur_f['h_team_id'] = cur_game.at[0,'off_team_id']
        cur_f['a_team_id'] = cur_game.at[1,'off_team_id']
        cur_f['h_rtg_pred'] = cur_game.at[0,'pred_eff']
        cur_f['a_rtg_pred'] = cur_game.at[1,'pred_eff']
        cur_f['actual_h'] = cur_game.at[0,'actual_eff']
        cur_f['actual_a'] = cur_game.at[1,'actual_eff']
        table.append(cur_f)

    
    eff = pd.DataFrame(table)

    pace = pd.read_csv('./predictions/latent/pace_arma_0.1.csv')
    pace = pace[pace['game_id'].isin(all_test['game_id'].unique())].reset_index(drop=True)
    
    final = pd.concat([eff,pace],axis=1)
    final = final.dropna()
    final.to_csv('./predictions/betting/pre_bet/'+name+'_expanding_xgb.csv', index=False)

#tune_xgboost('third')
#xgboost_regression('third')
#expanding_OLS('third')
#expanding_OLS('third',7500)
split('third')
OLS_w_best_eff('third',0)