import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import statsmodels.api as sm
import arviz as az
import pymc as pm
import nutpie
import pytensor
import pickle
import math

def team_pace():
    ### Hyperparams
    start_mu = 47.5
    start_sigma = 2
    max_sigma = 3
    per_game_fatten = 1.05
    per_season_fatten = 2
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
    tracker = {}
    for i in range(len(teams)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        team_map[teams[i]] = i
        tracker[teams[i]] = {}

    last_date_season = ''

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        h_ids=[]
        a_ids=[]
        obs = []

        for gid in game_ids:
            cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]

            if (last_date_season != '' and last_date_season != cur_season):
                for i in range(len(teams)):
                    priors[1][i] = min(priors[1][i] * per_season_fatten, max_sigma)
            last_date_season = cur_season

            try:
                pace = cur_game.at[0,"pace"]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            for i in range(2):
                tracker[cur_game.at[i,'team_id']][date] = [priors[0][team_map[cur_game.at[i,'team_id']]],priors[1][team_map[cur_game.at[i,'team_id']]]]

            h_ids.append(team_map[cur_game.at[0,'team_id']])
            a_ids.append(team_map[cur_game.at[1,'team_id']])
            obs.append(pace)
            
            home_prior = priors[0][team_map[cur_game.at[0,'team_id']]]
            away_prior = priors[0][team_map[cur_game.at[1,'team_id']]]


            cur_f = {}
            if (cur_season != "1996-97"):
                cur_f["pred"] = home_prior + away_prior
            
                cur_f["actual"] = pace     
                features.append(cur_f)

        #Updating below
        with pm.Model() as pymc_model:
            pace = pm.Normal('pace', mu=priors[0], sigma=priors[1], shape=len(team_map))

            mu = pace[h_ids] + pace[a_ids]
            sigma = pm.HalfNormal('sigma', sigma=1)
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=obs)
        
        compiled_model = nutpie.compile_pymc_model(pymc_model)
        trace_pymc = nutpie.sample(compiled_model, draws = 10000, seed=seed, chains=10, cores=8, save_warmup=False, progress_bar=False)
        for index, row in az.summary(trace_pymc).iterrows():
            if ('pace' in index):
                i = int(index.split("[")[1].split(']')[0])
                priors[0][i] = row['mean']
                if (i in h_ids or i in a_ids):
                    priors[1][i] = min(row['sd'] * per_game_fatten, max_sigma)
                else:
                    priors[1][i] = row['sd']

    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    with open('./intermediates/BHM_base_tracker_v4.pkl', 'wb') as f:
        pickle.dump(tracker, f)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace_BHM_base_v4.csv", index=False)
    

def player_pace_jfihds():
    ### Hyperparams
    start_mu = 47.5
    start_sigma = 2
    max_sigma = 3
    per_game_fatten = 1.05
    #per_season_fatten = 2
    seed = 1

    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    priors = [[],[]]
    tracker = {}
    for i in range(len(players)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        player_map[players[i]] = i
        tracker[players[i]] = {}
    priors[0] = np.array(priors[0])
    priors[1] = np.array(priors[1])

    last_lineup = {}
    cur_lineup = {}
    expected_pace = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}
        expected_pace[team] = 95

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        p_ids=[]
        p_sec_inv=[]
        opp_team_ids=[]
        obs = []

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            try:
                team_pace = team_game.at[0,"pace"]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            avg_player_pace = (cur_game['seconds'] * cur_game['pace']).sum() / (cur_game['seconds'].sum())

            #for i in range(2):
                #tracker[cur_game.at[i,'team_id']][date] = [priors[0][team_map[cur_game.at[i,'team_id']]],priors[1][team_map[cur_game.at[i,'team_id']]]]


            cur_lineup[h_id] = {}
            expected_pace[h_id] = 0
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                cur_lineup[h_id][row['player_id']] = row['seconds']/total_sec
                expected_pace[h_id] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                p_ids.append(player_map[row['player_id']])
                p_sec_inv.append((total_sec/5) / row['seconds'])
                opp_team_ids.append(a_id)
                obs.append(row['pace'] * team_pace/avg_player_pace)

            cur_lineup[a_id] = {}
            expected_pace[a_id] = 0
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                cur_lineup[a_id][row['player_id']] = row['seconds']/total_sec
                expected_pace[a_id] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                p_ids.append(player_map[row['player_id']])
                p_sec_inv.append((total_sec/5) / row['seconds'])
                opp_team_ids.append(h_id)
                obs.append(row['pace'] * team_pace/avg_player_pace)

            

            cur_f = {'last_pred':0,'cur_pred':0}
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    for z in cur_lineup[x]:
                        cur_f['cur_pred'] += cur_lineup[x][z] * priors[0][player_map[z]]
                    for z in last_lineup[x]:
                        cur_f['last_pred'] += last_lineup[x][z] * priors[0][player_map[z]]
            
                cur_f["actual"] = pace     
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec


        #Updating below
        with pm.Model() as pymc_model:
            pace = pm.Normal('pace', mu=priors[0][p_ids], sigma=priors[1][p_ids], shape=len(p_ids))

            opp_pace = []
            for opp_id in opp_team_ids:
                opp_pace.append(expected_pace[opp_id])
            opp_pace = np.array(opp_pace)

            p_sec_inv = np.array(p_sec_inv)*100

            mu = pace + opp_pace
            #sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=p_sec_inv, observed=obs)
        
        compiled_model = nutpie.compile_pymc_model(pymc_model)
        trace_pymc = nutpie.sample(compiled_model, draws = 10000, seed=seed, chains=10, cores=8, save_warmup=False)
        print (az.summary(trace_pymc))
        print (p_ids[0:5], p_ids[-5:])
        print (p_sec_inv[0:5], p_sec_inv[-5:])
        print (obs[0:5], obs[-5:])
        return (1)
        # for index, row in az.summary(trace_pymc).iterrows():
        #     if ('pace' in index):
        #         i = int(index.split("[")[1].split(']')[0])
        #         priors[0][i] = row['mean']
        #         if (i in h_ids or i in a_ids):
        #             priors[1][i] = min(row['sd'] * per_game_fatten, max_sigma)
        #         else:
        #             priors[1][i] = row['sd']

    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    with open('./intermediates/BHM_base_tracker_v4.pkl', 'wb') as f:
        pickle.dump(tracker, f)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace_BHM_base_v4.csv", index=False)


def player_pace():
    ### Hyperparams
    start_mu = 47.5
    start_sigma = 3
    max_sigma = 6
    per_game_fatten = 1.05
    obs_sigma = 3
    #per_season_fatten = 2
    seed = 1

    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    priors = [[],[]]
    tracker = {}
    for i in range(len(players)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        player_map[players[i]] = i
        tracker[players[i]] = {}
    priors[0] = np.array(priors[0], dtype='float')
    priors[1] = np.array(priors[1], dtype='float')


    last_lineup = {}
    cur_lineup = {}
    expected_pace = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}
        expected_pace[team] = 95

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        p_ids=[]
        p_sec_inv=[]
        opp_team_ids=[]
        obs = []

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()

            try:
                team_pace = team_game.at[0,"pace"]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()
            
            avg_player_pace = (cur_game['seconds'] * cur_game['pace']).sum() / (cur_game['seconds'].sum())


            cur_lineup[h_id] = {}
            expected_pace[h_id] = 0
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                cur_lineup[h_id][row['player_id']] = row['seconds']/total_sec
                expected_pace[h_id] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                tracker[row['player_id']][date] = [priors[0][player_map[row['player_id']]], priors[1][player_map[row['player_id']]]]

                p_ids.append(player_map[row['player_id']])
                p_sec_inv.append((total_sec/5) / row['seconds'])
                opp_team_ids.append(a_id)
                obs.append(row['pace'] * team_pace/avg_player_pace)

            cur_lineup[a_id] = {}
            expected_pace[a_id] = 0
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                cur_lineup[a_id][row['player_id']] = row['seconds']/total_sec
                expected_pace[a_id] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                tracker[row['player_id']][date] = [priors[0][player_map[row['player_id']]], priors[1][player_map[row['player_id']]]]

                p_ids.append(player_map[row['player_id']])
                p_sec_inv.append((total_sec/5) / row['seconds'])
                opp_team_ids.append(h_id)
                obs.append(row['pace'] * team_pace/avg_player_pace)


            cur_f = {'last_pred':0,'cur_pred':0}
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    for z in cur_lineup[x]:
                        cur_f['cur_pred'] += cur_lineup[x][z] * priors[0][player_map[z]]
                    for z in last_lineup[x]:
                        cur_f['last_pred'] += last_lineup[x][z] * priors[0][player_map[z]]
            
                cur_f["actual"] = team_pace   
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec

        opp_pace = []
        for opp_id in opp_team_ids:
            opp_pace.append(expected_pace[opp_id])

        for i in range(len(p_ids)):
            if (p_sec_inv[i] > 10):
                continue
            priors[0][p_ids[i]] = (priors[0][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_pace[i])*priors[1][p_ids[i]]**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
            priors[1][p_ids[i]] = min(math.sqrt((priors[1][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)
        
        # if (np.isnan(np.sum(priors[0]))):
        #     return (1)

        
        # for index, row in az.summary(trace_pymc).iterrows():
        #     if ('pace' in index):
        #         i = int(index.split("[")[1].split(']')[0])
        #         priors[0][i] = row['mean']
        #         if (i in h_ids or i in a_ids):
        #             priors[1][i] = min(row['sd'] * per_game_fatten, max_sigma)
        #         else:
        #             priors[1][i] = row['sd']

    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    with open('./intermediates/BHM_player_tracker_v6.pkl', 'wb') as f:
        pickle.dump(tracker, f)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace_BHM_player_v6.csv", index=False)

if __name__ == '__main__':
    player_pace()