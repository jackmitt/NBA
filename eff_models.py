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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


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
    for x in ['last_pred_rtg','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        print (x)
        print (len(df.index))
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]
        print (len(df.index))
    

    results = sm.OLS(df['actual_eff'], df[['intercept','last_pred_rtg','home_off','cur_pred_3pt_pct_off','cur_pred_2pt_pct_off','cur_pred_to_pct_off',
                                           'cur_pred_ftrXftp_off','cur_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']]).fit()
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

def OLS_w_best_eff(first_second):
    df = pd.read_csv('./intermediates/regression_formatted/'+first_second+'_train.csv').dropna()
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
        if ('cur_' in col):
            df = df.rename(columns={col:'reg_'+col.split('cur_')[1]})

    #remove outliers
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]
    

    reg = LinearRegression().fit(df[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']], df['actual_eff'])
    train_pred = reg.predict(df[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    

    test = pd.read_csv('./intermediates/regression_formatted/'+first_second+'_test.csv').dropna()
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
        if ('last_' in col):
            test = test.rename(columns={col:'reg_'+col.split('last_')[1]})

    #remove outliers based on train data
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]

    test_pred = reg.predict(test[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/'+first_second+'_eff_w_best_eff_ols.csv',index=False)

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
    final.to_csv('./predictions/betting/pre_bet/'+first_second+'_w_best_eff_reg_season_last_lu_test.csv', index=False)

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

#Best Params: {'eta': 0.15, 'max_depth': 1, 'min_child_weight': 0.1, 'gamma': 3000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'num_parallel_tree': 1}
def xgboost_regression(first_second):
    df = pd.read_csv('./intermediates/regression_formatted/'+first_second+'_train.csv').dropna()
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
        if ('cur_' in col):
            df = df.rename(columns={col:'reg_'+col.split('cur_')[1]})

    #remove outliers
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        df = df[~(df[x] <= q25 - 3*iqr)]
        df = df[~(df[x] >= q75 + 3*iqr)]
    

    reg = xgb.XGBRegressor(eta=0.15,max_depth=1,min_child_weight=0.1,gamma=3000,subsample=0.8,colsample_bytree=0.8)
    reg.fit(df[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']], df['actual_eff'])
    train_pred = reg.predict(df[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    

    test = pd.read_csv('./intermediates/regression_formatted/'+first_second+'_test.csv').dropna()
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
        if ('last_' in col):
            test = test.rename(columns={col:'reg_'+col.split('last_')[1]})

    #remove outliers based on train data
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(df[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]

    test_pred = reg.predict(test[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    print ("MSE of Training Set",mean_squared_error(df['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))

    test['pred_eff'] = test_pred
    test.to_csv('./predictions/latent/'+first_second+'_xgb.csv',index=False)

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
    final.to_csv('./predictions/betting/pre_bet/'+first_second+'_xgb.csv', index=False)

def tune_xgboost_sub_call(first_second,params):
    df = pd.read_csv('./intermediates/regression_formatted/'+first_second+'_train.csv').dropna()
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
        if ('cur_' in col):
            train = train.rename(columns={col:'reg_'+col.split('cur_')[1]})

    #remove outliers
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(train[x], [75 ,25])
        iqr = q75 - q25
        train = train[~(train[x] <= q25 - 3*iqr)]
        train = train[~(train[x] >= q75 + 3*iqr)]
    

    reg = xgb.XGBRegressor(seed=0,eta=params['eta'],max_depth=params['max_depth'],min_child_weight=params['min_child_weight'],gamma=params['gamma'],subsample=params['subsample'],
                           colsample_bytree=params['colsample_bytree'])
    reg.fit(train[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']], train['actual_eff'])
    train_pred = reg.predict(train[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])

    #use cur lineup to train, last lineup to test
    for col in test.columns:
        if ('last_' in col):
            test = test.rename(columns={col:'reg_'+col.split('last_')[1]})

    #remove outliers based on train data
    for x in ['reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']:
        q75, q25 = np.percentile(train[x], [75 ,25])
        iqr = q75 - q25
        test = test[~(test[x] <= q25 - 3*iqr)]
        test = test[~(test[x] >= q75 + 3*iqr)]

    test_pred = reg.predict(test[['home_off','reg_pred_rtg','reg_pred_3pt_pct_off','reg_pred_2pt_pct_off','reg_pred_to_pct_off',
                                           'reg_pred_ftrXftp_off','reg_pred_ftrXftp_def','pred_3pt_pct_def',
                                           'pred_2pt_pct_def','pred_to_pct_def','pred_oreb_pct_off','pred_oreb_pct_def']])
    print (params)
    print ("MSE of Training Set",mean_squared_error(train['actual_eff'],train_pred))
    print ("MSE of Test Set",mean_squared_error(test['actual_eff'],test_pred))
    return (mean_squared_error(test['actual_eff'],test_pred))

def tune_xgboost(first_second):
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
                mse = tune_xgboost_sub_call(first_second, params)
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

xgboost_regression('second')