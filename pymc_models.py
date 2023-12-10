import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import statsmodels.api as sm
import arviz as az
import pymc as pm
import nutpie
import pytensor

def team_pace():
    ### Hyperparams
    start_mu = 47.5
    start_sigma = 2
    seed = 1

    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()

    features = []

    team_map = {}
    priors = [[],[]]
    for i in range(len(teams)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        team_map[teams[i]] = i

    for date in games['game_date'].unique():
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        h_ids=[]
        a_ids=[]
        obs = []

        for gid in game_ids:
            cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]

            try:
                pace = cur_game.at[0,"pace"]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_ids.append(team_map[cur_game.at[0,'team_id']])
            a_ids.append(team_map[cur_game.at[1,'team_id']])
            obs.append(pace)
            
            home_prior = priors[0][team_map[cur_game.at[0,'team_id']]]
            away_prior = priors[0][team_map[cur_game.at[1,'team_id']]]


            cur_f = {}
            if (cur_season != "1996-97"):
                cur_f["pred"] = home_prior[0] + away_prior[0]
            
                cur_f["actual"] = pace     
                features.append(cur_f)

        #Updating below
        with pm.Model() as pymc_model:
            pace = pm.Normal('pace', mu=priors[0], sigma=priors[1], shape=len(team_map))

            mu = pace[h_ids] + pace[a_ids]
            sigma = pm.HalfNormal('sigma', sigma=1)
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=obs)
        
        compiled_model = nutpie.compile_pymc_model(pymc_model)
        trace_pymc = nutpie.sample(compiled_model, draws = 10000, seed=seed)
        print (h_ids, a_ids, obs)
        print (az.summary(trace_pymc, round_to=2))
        return (1)

    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace.csv", index=False)

if __name__ == '__main__':
    team_pace()