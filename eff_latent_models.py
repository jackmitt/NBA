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
import matplotlib.pyplot as plt
import time
import concurrent.futures

def team():
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    game_ids = games["game_id"].to_list()
    raw = {}
    seas = {}
    arma = {0.5:{},0.25:{},0.1:{},0.05:{},0.01:{}}
    bayes = {0.5:{},0.25:{},0.1:{},0.05:{},0.01:{}}

    features = []

    for team in games["h_team_id"].unique():
        raw[team] = [[],[]]
        for weight in arma:
            arma[weight][team] = [50,50]
        for weight in bayes:
            bayes[weight][team] = [50,50]
        seas[team] = {}
        for season in games["season"].unique():
            seas[team][season] = [[],[]]
    
    for gid in tqdm(game_ids):
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        pri_season = games["season"].unique()[games["season"].unique().tolist().index(cur_season)-1]

        try:
            h_rtg = cur_game.at[0,"offensiveRating"]
            a_rtg = cur_game.at[1,"offensiveRating"]
        except:
            if (cur_season != "1996-97"):
                cur_f = {}
                features.append(cur_f)
            continue

        cur_f = {}
        if (cur_season != "1996-97"):
            hn = len(seas[cur_game.at[0,"team_id"]][cur_season])
            an = len(seas[cur_game.at[1,"team_id"]][cur_season])
            if (hn < 5):
                if (hn == 0):
                    h_o = np.average(seas[cur_game.at[0,"team_id"]][pri_season][0])
                    h_d = np.average(seas[cur_game.at[0,"team_id"]][pri_season][1])
                else:
                    h_o = (5-hn)/5 * np.average(seas[cur_game.at[0,"team_id"]][pri_season][0]) + hn/5 * np.average(seas[cur_game.at[0,"team_id"]][cur_season][0])
                    h_d = (5-hn)/5 * np.average(seas[cur_game.at[0,"team_id"]][pri_season][1]) + hn/5 * np.average(seas[cur_game.at[0,"team_id"]][cur_season][1])
            else:
                h_o = np.average(seas[cur_game.at[0,"team_id"]][cur_season][0])
                h_d = np.average(seas[cur_game.at[0,"team_id"]][cur_season][1])
            if (an < 5):
                if (an == 0):
                    a_o = np.average(seas[cur_game.at[1,"team_id"]][pri_season][0])
                    a_d = np.average(seas[cur_game.at[1,"team_id"]][pri_season][1])
                else:
                    a_o = (5-hn)/5 * np.average(seas[cur_game.at[1,"team_id"]][pri_season][0]) + hn/5 * np.average(seas[cur_game.at[1,"team_id"]][cur_season][0])
                    a_d = (5-hn)/5 * np.average(seas[cur_game.at[1,"team_id"]][pri_season][1]) + hn/5 * np.average(seas[cur_game.at[1,"team_id"]][cur_season][1])
            else:
                a_o = np.average(seas[cur_game.at[1,"team_id"]][cur_season][0])
                a_d = np.average(seas[cur_game.at[1,"team_id"]][cur_season][1])
            cur_f["season_avg_h"] = (h_o + a_d) / 2
            cur_f["season_avg_a"] = (a_o + h_d) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-5:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-5:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-5:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-5:])
            cur_f["ma_5_h"] = (h_o+a_d) / 2
            cur_f["ma_5_a"] = (h_d+a_o) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-10:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-10:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-10:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-10:])
            cur_f["ma_10_h"] = (h_o+a_d) / 2
            cur_f["ma_10_a"] = (h_d+a_o) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-30:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-30:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-30:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-30:])
            cur_f["ma_30_h"] = (h_o+a_d) / 2
            cur_f["ma_30_a"] = (h_d+a_o) / 2

            for weight in arma:
                cur_f["arma_"+str(weight)+'_h'] = arma[weight][cur_game.at[0,"team_id"]][0] + arma[weight][cur_game.at[1,"team_id"]][1]
                cur_f["arma_"+str(weight)+'_a'] = arma[weight][cur_game.at[0,"team_id"]][1] + arma[weight][cur_game.at[1,"team_id"]][0]
            
            for weight in bayes:
                cur_f["bayes_"+str(weight)+'_h'] = bayes[weight][cur_game.at[0,"team_id"]][0] + bayes[weight][cur_game.at[1,"team_id"]][1]
                cur_f["bayes_"+str(weight)+'_a'] = bayes[weight][cur_game.at[0,"team_id"]][1] + bayes[weight][cur_game.at[1,"team_id"]][0]
        
            cur_f["actual_h"] = h_rtg 
            cur_f["actual_a"] = a_rtg   
            features.append(cur_f)

        #Updating below

       
        raw[cur_game.at[0,"team_id"]][0].append(h_rtg)
        raw[cur_game.at[1,"team_id"]][1].append(h_rtg)
        raw[cur_game.at[1,"team_id"]][0].append(a_rtg)
        raw[cur_game.at[0,"team_id"]][1].append(a_rtg)
        seas[cur_game.at[0,"team_id"]][cur_season][0].append(h_rtg)
        seas[cur_game.at[1,"team_id"]][cur_season][1].append(h_rtg)
        seas[cur_game.at[1,"team_id"]][cur_season][0].append(a_rtg)
        seas[cur_game.at[0,"team_id"]][cur_season][1].append(a_rtg)

        for weight in arma:
            error_term_h = h_rtg - arma[weight][cur_game.at[0,"team_id"]][0] - arma[weight][cur_game.at[1,"team_id"]][1]
            error_term_a = a_rtg - arma[weight][cur_game.at[1,"team_id"]][0] - arma[weight][cur_game.at[0,"team_id"]][1]
            
            arma[weight][cur_game.at[0,"team_id"]][0] += error_term_h*weight
            arma[weight][cur_game.at[1,"team_id"]][1] += error_term_h*weight
            arma[weight][cur_game.at[1,"team_id"]][0] += error_term_a*weight
            arma[weight][cur_game.at[0,"team_id"]][1] += error_term_a*weight

        for weight in bayes:
            bayes[weight][cur_game.at[0,"team_id"]][0] = (1-weight) * bayes[weight][cur_game.at[0,"team_id"]][0] + weight * h_rtg/2
            bayes[weight][cur_game.at[1,"team_id"]][1] = (1-weight) * bayes[weight][cur_game.at[1,"team_id"]][1] + weight * h_rtg/2
            bayes[weight][cur_game.at[1,"team_id"]][0] = (1-weight) * bayes[weight][cur_game.at[1,"team_id"]][0] + weight * a_rtg/2
            bayes[weight][cur_game.at[0,"team_id"]][1] = (1-weight) * bayes[weight][cur_game.at[0,"team_id"]][1] + weight * a_rtg/2
    
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/eff.csv", index=False)

def team_bhm():
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    game_ids = games["game_id"].to_list()
    raw = {}
    seas = {}
    arma = {0.5:{},0.25:{},0.1:{},0.05:{},0.01:{}}
    bayes = {0.5:{},0.25:{},0.1:{},0.05:{},0.01:{}}

    features = []

    for team in games["h_team_id"].unique():
        raw[team] = [[],[]]
        for weight in arma:
            arma[weight][team] = [50,50]
        for weight in bayes:
            bayes[weight][team] = [50,50]
        seas[team] = {}
        for season in games["season"].unique():
            seas[team][season] = [[],[]]
    
    for gid in tqdm(game_ids):
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        pri_season = games["season"].unique()[games["season"].unique().tolist().index(cur_season)-1]

        try:
            h_rtg = cur_game.at[0,"offensiveRating"]
            a_rtg = cur_game.at[1,"offensiveRating"]
        except:
            if (cur_season != "1996-97"):
                cur_f = {}
                features.append(cur_f)
            continue

        cur_f = {}
        if (cur_season != "1996-97"):
            hn = len(seas[cur_game.at[0,"team_id"]][cur_season])
            an = len(seas[cur_game.at[1,"team_id"]][cur_season])
            if (hn < 5):
                if (hn == 0):
                    h_o = np.average(seas[cur_game.at[0,"team_id"]][pri_season][0])
                    h_d = np.average(seas[cur_game.at[0,"team_id"]][pri_season][1])
                else:
                    h_o = (5-hn)/5 * np.average(seas[cur_game.at[0,"team_id"]][pri_season][0]) + hn/5 * np.average(seas[cur_game.at[0,"team_id"]][cur_season][0])
                    h_d = (5-hn)/5 * np.average(seas[cur_game.at[0,"team_id"]][pri_season][1]) + hn/5 * np.average(seas[cur_game.at[0,"team_id"]][cur_season][1])
            else:
                h_o = np.average(seas[cur_game.at[0,"team_id"]][cur_season][0])
                h_d = np.average(seas[cur_game.at[0,"team_id"]][cur_season][1])
            if (an < 5):
                if (an == 0):
                    a_o = np.average(seas[cur_game.at[1,"team_id"]][pri_season][0])
                    a_d = np.average(seas[cur_game.at[1,"team_id"]][pri_season][1])
                else:
                    a_o = (5-hn)/5 * np.average(seas[cur_game.at[1,"team_id"]][pri_season][0]) + hn/5 * np.average(seas[cur_game.at[1,"team_id"]][cur_season][0])
                    a_d = (5-hn)/5 * np.average(seas[cur_game.at[1,"team_id"]][pri_season][1]) + hn/5 * np.average(seas[cur_game.at[1,"team_id"]][cur_season][1])
            else:
                a_o = np.average(seas[cur_game.at[1,"team_id"]][cur_season][0])
                a_d = np.average(seas[cur_game.at[1,"team_id"]][cur_season][1])
            cur_f["season_avg_h"] = (h_o + a_d) / 2
            cur_f["season_avg_a"] = (a_o + h_d) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-5:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-5:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-5:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-5:])
            cur_f["ma_5_h"] = (h_o+a_d) / 2
            cur_f["ma_5_a"] = (h_d+a_o) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-10:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-10:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-10:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-10:])
            cur_f["ma_10_h"] = (h_o+a_d) / 2
            cur_f["ma_10_a"] = (h_d+a_o) / 2

            h_o = np.average(raw[cur_game.at[0,"team_id"]][0][-30:])
            h_d = np.average(raw[cur_game.at[0,"team_id"]][1][-30:])
            a_o = np.average(raw[cur_game.at[1,"team_id"]][0][-30:])
            a_d = np.average(raw[cur_game.at[1,"team_id"]][1][-30:])
            cur_f["ma_30_h"] = (h_o+a_d) / 2
            cur_f["ma_30_a"] = (h_d+a_o) / 2

            for weight in arma:
                cur_f["arma_"+str(weight)+'_h'] = arma[weight][cur_game.at[0,"team_id"]][0] + arma[weight][cur_game.at[1,"team_id"]][1]
                cur_f["arma_"+str(weight)+'_a'] = arma[weight][cur_game.at[0,"team_id"]][1] + arma[weight][cur_game.at[1,"team_id"]][0]
            
            for weight in bayes:
                cur_f["bayes_"+str(weight)+'_h'] = bayes[weight][cur_game.at[0,"team_id"]][0] + bayes[weight][cur_game.at[1,"team_id"]][1]
                cur_f["bayes_"+str(weight)+'_a'] = bayes[weight][cur_game.at[0,"team_id"]][1] + bayes[weight][cur_game.at[1,"team_id"]][0]
        
            cur_f["actual_h"] = h_rtg 
            cur_f["actual_a"] = a_rtg   
            features.append(cur_f)

        #Updating below

       
        raw[cur_game.at[0,"team_id"]][0].append(h_rtg)
        raw[cur_game.at[1,"team_id"]][1].append(h_rtg)
        raw[cur_game.at[1,"team_id"]][0].append(a_rtg)
        raw[cur_game.at[0,"team_id"]][1].append(a_rtg)
        seas[cur_game.at[0,"team_id"]][cur_season][0].append(h_rtg)
        seas[cur_game.at[1,"team_id"]][cur_season][1].append(h_rtg)
        seas[cur_game.at[1,"team_id"]][cur_season][0].append(a_rtg)
        seas[cur_game.at[0,"team_id"]][cur_season][1].append(a_rtg)

        for weight in arma:
            error_term_h = h_rtg - arma[weight][cur_game.at[0,"team_id"]][0] - arma[weight][cur_game.at[1,"team_id"]][1]
            error_term_a = a_rtg - arma[weight][cur_game.at[1,"team_id"]][0] - arma[weight][cur_game.at[0,"team_id"]][1]
            
            arma[weight][cur_game.at[0,"team_id"]][0] += error_term_h*weight
            arma[weight][cur_game.at[1,"team_id"]][1] += error_term_h*weight
            arma[weight][cur_game.at[1,"team_id"]][0] += error_term_a*weight
            arma[weight][cur_game.at[0,"team_id"]][1] += error_term_a*weight

        for weight in bayes:
            bayes[weight][cur_game.at[0,"team_id"]][0] = (1-weight) * bayes[weight][cur_game.at[0,"team_id"]][0] + weight * h_rtg/2
            bayes[weight][cur_game.at[1,"team_id"]][1] = (1-weight) * bayes[weight][cur_game.at[1,"team_id"]][1] + weight * h_rtg/2
            bayes[weight][cur_game.at[1,"team_id"]][0] = (1-weight) * bayes[weight][cur_game.at[1,"team_id"]][0] + weight * a_rtg/2
            bayes[weight][cur_game.at[0,"team_id"]][1] = (1-weight) * bayes[weight][cur_game.at[0,"team_id"]][1] + weight * a_rtg/2
    
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/eff.csv", index=False)

def player_eff_naive_bhm(per_game_fatten, obs_sigma, home_adv, b2b_pen):
    ### Hyperparams
    start_mu = 52.5
    start_sigma = 4
    max_sigma = 10
    #per_game_fatten = 1.05
    #obs_sigma = 1
    #per_season_fatten = 2
    seed = 1

    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    player_team = {}
    priors = [[],[],[],[]]
    tracker = {}
    for i in range(len(players)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        priors[2].append(start_mu)
        priors[3].append(start_sigma)
        player_map[players[i]] = i
        player_team[players[i]] = -1
        tracker[players[i]] = {}
    priors[0] = np.array(priors[0], dtype='float')
    priors[1] = np.array(priors[1], dtype='float')
    priors[2] = np.array(priors[0], dtype='float')
    priors[3] = np.array(priors[1], dtype='float')


    last_lineup = {}
    cur_lineup = {}
    expected_off_eff = {}
    expected_def_eff = {}
    last_game = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}
        expected_off_eff[team] = 50
        expected_def_eff[team] = 50
        last_game[team] = datetime.datetime(1800,1,1)

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        p_ids=[]
        p_sec_inv=[]
        opp_team_ids=[]
        obs = []
        off_def = []

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_eff = team_game.at[0,'offensiveRating']
                a_team_eff = team_game.at[1,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            if ((date-last_game[h_id]).days == 1):
                h_b2b = True
            else:
                h_b2b = False

            if ((date-last_game[a_id]).days == 1):
                a_b2b = True
            else:
                a_b2b = False
            
            last_game[h_id] = date
            last_game[a_id] = date

            cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                expected_off_eff[x] = 0
                expected_def_eff[x] = 0
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec

                    expected_off_eff[x] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                    p_ids.append(player_map[row['player_id']])
                    if (row['seconds'] != 0):
                        p_sec_inv.append(((total_sec/5) / row['seconds']))
                    else:
                        p_sec_inv.append(100)
                    if (x == h_id):
                        opp_team_ids.append(a_id)
                    else:
                        opp_team_ids.append(h_id)
                    if (x == h_id):
                        obs.append(row['offensiveRating'] * h_team_eff/avg_h_off_player_eff - home_adv)
                        if (h_b2b):
                            obs[-1] += b2b_pen
                        if (a_b2b):
                            obs[-1] -= b2b_pen
                    else:
                        obs.append(row['offensiveRating'] * a_team_eff/avg_a_off_player_eff + home_adv)
                        if (h_b2b):
                            obs[-1] -= b2b_pen
                        if (a_b2b):
                            obs[-1] += b2b_pen
                    off_def.append('off')

                    expected_def_eff[x] += (row['seconds']/total_sec) * priors[2][player_map[row['player_id']]]

                    p_ids.append(player_map[row['player_id']])
                    if (row['seconds'] != 0):
                        p_sec_inv.append(((total_sec/5) / row['seconds']))
                    else:
                        p_sec_inv.append(100)
                    if (x == h_id):
                        opp_team_ids.append(a_id)
                    else:
                        opp_team_ids.append(h_id)
                    if (x == h_id):
                        obs.append(row['defensiveRating'] * a_team_eff/avg_h_def_player_eff + home_adv)
                        if (h_b2b):
                            obs[-1] -= b2b_pen
                        if (a_b2b):
                            obs[-1] += b2b_pen
                    else:
                        obs.append(row['defensiveRating'] * h_team_eff/avg_a_def_player_eff - home_adv)
                        if (h_b2b):
                            obs[-1] += b2b_pen
                        if (a_b2b):
                            obs[-1] -= b2b_pen
                    off_def.append('def')

    
            cur_f = {'last_pred_h_eff':home_adv,'cur_pred_h_eff':home_adv,'last_pred_a_eff':-home_adv,'cur_pred_a_eff':-home_adv}
            if (h_b2b):
                cur_f['last_pred_h_eff'] -= b2b_pen
                cur_f['cur_pred_h_eff'] -= b2b_pen
                cur_f['last_pred_a_eff'] += b2b_pen
                cur_f['cur_pred_a_eff'] += b2b_pen
            if (a_b2b):
                cur_f['last_pred_a_eff'] -= b2b_pen
                cur_f['cur_pred_a_eff'] -= b2b_pen
                cur_f['last_pred_h_eff'] += b2b_pen
                cur_f['cur_pred_h_eff'] += b2b_pen
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    for z in cur_lineup[x]:
                        if (x == h_id):
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
                        else:
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
                    for z in last_lineup[x]:
                        if (x == h_id):
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
                        else:
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            
                cur_f["actual_h_eff"] = h_team_eff
                cur_f["actual_a_eff"] = a_team_eff   
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec

        opp_off_eff = []
        opp_def_eff = []
        for opp_id in opp_team_ids:
            opp_off_eff.append(expected_off_eff[opp_id])
            opp_def_eff.append(expected_def_eff[opp_id])

        for i in range(len(p_ids)):
            if (p_sec_inv[i] > 25):
                continue
            if (off_def[i] == 'off'):
                priors[0][p_ids[i]] = (priors[0][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_def_eff[i])*priors[1][p_ids[i]]**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
                priors[1][p_ids[i]] = min(math.sqrt((priors[1][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)
            else:
                priors[2][p_ids[i]] = (priors[2][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_off_eff[i])*priors[3][p_ids[i]]**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
                priors[3][p_ids[i]] = min(math.sqrt((priors[3][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)

    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/player_eff_bhm_"+str(obs_sigma)+"_"+str(per_game_fatten)+"_"+str(home_adv)+"_"+str(b2b_pen)+".csv", index=False)

    # plt.hist(priors[1])
    # plt.show()
    # plt.hist(priors[3])
    # plt.show()

class eff_latent_models:
    #gets the data loaded in and set up
    def load_data(self):
        self.player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
        self.team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
        self.games = pd.read_csv("./database/games.csv")
        self.lu = pd.read_csv('./database/unique_lineup_stats.csv')
        seasons = ""
        for yr in range(1996, 2003):
            seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
        self.games = self.games[self.games["season"].str.contains(seasons[:-1])]
        self.games = self.games[self.games["game_type"].str.contains("Regular Season|Playoffs")]
        self.teams = self.games['h_team_id'].unique()
        self.players = self.player_bs['player_id'].unique()

        self.player_bs['seconds'] = self.player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    def init_priors(self, start_mu, start_sigma):
        self.player_map = {}
        self.player_team = {}
        self.priors = [[],[],[],[]]
        self.home_prior = [3, 3]
        self.tracker = {}
        for i in range(len(self.players)):
            self.priors[0].append(start_mu)
            self.priors[1].append(start_sigma)
            self.priors[2].append(start_mu)
            self.priors[3].append(start_sigma)
            self.player_map[self.players[i]] = i
            self.player_team[self.players[i]] = -1
            self.tracker[self.players[i]] = {}
        self.priors[0] = np.array(self.priors[0], dtype='float')
        self.priors[1] = np.array(self.priors[1], dtype='float')
        self.priors[2] = np.array(self.priors[2], dtype='float')
        self.priors[3] = np.array(self.priors[3], dtype='float')
    
    def init_lineup_track(self):
        self.last_lineup = {}
        self.cur_lineup = {}
        self.expected_off = {}
        self.expected_def = {}
        for team in self.teams:
            self.last_lineup[team] = {}
            self.cur_lineup[team] = {}
            self.expected_off[team] = 100
            self.expected_def[team] = 100


    def player_eff_bhm(self, start_mu, start_sigma, max_sigma, obs_sigma, per_game_fatten):
        self.load_data()
        self.init_priors(start_mu,start_sigma)
        self.init_lineup_track()

        features = []
        
        for date in tqdm(self.games['game_date'].unique()):
            game_ids = self.games.loc[self.games['game_date']==date,]['game_id'].unique()       
            
            off_def = {"off":{},"def":{}}
            for key in off_def:
                off_def[key]['p_ids'] = []
                off_def[key]['h_ind'] = []
                off_def[key]['p_sec_inv'] = []
                off_def[key]['opp_eff'] = []
                off_def[key]['obs'] = []

            for gid in game_ids:
                cur_game = self.player_bs.loc[self.player_bs["game_id"] == gid,].dropna().reset_index()
                cur_season = self.games.loc[self.games["game_id"] == gid, ]["season"].to_list()[0]
                team_game = self.team_bs.loc[self.team_bs["game_id"] == gid,].reset_index()


                try:
                    h_eff = team_game.at[0,"offensiveRating"]
                    a_eff = team_game.at[1,"offensiveRating"]
                except:
                    if (cur_season != "1996-97"):
                        cur_f = {}
                        features.append(cur_f)
                    continue

                h_id = cur_game['team_id'].unique()[0]
                a_id = cur_game['team_id'].unique()[1]

                cur_h = cur_game.loc[cur_game['team_id']==h_id,]
                cur_a = cur_game.loc[cur_game['team_id']==a_id,]

                total_sec = cur_h['seconds'].sum()
                
                avg_h_player = (cur_h['seconds'] * cur_h['offensiveRating']).sum() / (cur_h['seconds'].sum())
                avg_a_player = (cur_a['seconds'] * cur_a['offensiveRating']).sum() / (cur_a['seconds'].sum())


                for x in [h_id, a_id]:
                    self.cur_lineup[x] = {}
                    self.expected_h_off[x] = 0
                    self.expected_a_off[x] = 0
                    self.expected_h_def[x] = 0
                    self.expected_a_def[x] = 0
                    for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                        self.cur_lineup[x][row['player_id']] = row['seconds']/total_sec
                        if (x == h_id):
                            self.expected_h_off[x] += (row['seconds']/total_sec) * self.priors[0][self.player_map[row['player_id']]]
                            self.expected_h_def[x] += (row['seconds']/total_sec) * self.priors[2][self.player_map[row['player_id']]]
                        else:
                            self.expected_a_off[x] += (row['seconds']/total_sec) * self.priors[0][self.player_map[row['player_id']]]
                            self.expected_a_def[x] += (row['seconds']/total_sec) * self.priors[2][self.player_map[row['player_id']]]

                        self.tracker[row['player_id']][date] = [self.priors[0][self.player_map[row['player_id']]], self.priors[1][self.player_map[row['player_id']]]]
                    for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                        off_def['off']['p_ids'].append(self.player_map[row['player_id']])
                        if (row['seconds'] != 0):
                            off_def['off']['p_sec_inv'].append(((total_sec/5) / row['seconds']))
                        else:
                            off_def['off']['p_sec_inv'].append(100)
                        if (x == h_id):
                            off_def['off']['opp_eff'].append(self.expected_a_def)
                            off_def['off']['obs'].append(row['offensiveRating'] * h_eff/avg_h_player)
                            off_def['off']['h_ind'].append(1)
                        else:
                            off_def['off']['opp_eff'].append(self.expected_h_def)
                            off_def['off']['obs'].append(row['offensiveRating'] * a_eff/avg_a_player)
                            off_def['off']['h_ind'].append(0)

                        off_def['def']['p_ids'].append(self.player_map[row['player_id']])
                        if (row['seconds'] != 0):
                            off_def['def']['p_sec_inv'].append(((total_sec/5) / row['seconds']))
                        else:
                            off_def['def']['p_sec_inv'].append(100)
                        if (x == h_id):
                            off_def['def']['opp_eff'].append(self.expected_a_off)
                            off_def['def']['obs'].append(row['defensiveRating'] * a_eff/avg_a_player)
                            off_def['def']['h_ind'].append(0)
                        else:
                            off_def['def']['opp_eff'].append(self.expected_h_off)
                            off_def['def']['obs'].append(row['defensiveRating'] * h_eff/avg_h_player)
                            off_def['def']['h_ind'].append(1)


                cur_f = {'last_pred_h':0,'cur_pred_h':0,'last_pred_a':0,'cur_pred_a':0}
                if (cur_season != "1996-97"):
                    for z in self.cur_lineup[h_id]:
                        cur_f['cur_pred_h'] += self.cur_lineup[h_id][z] * self.priors[0][self.player_map[z]]
                        cur_f['cur_pred_a'] += self.cur_lineup[h_id][z] * self.priors[2][self.player_map[z]]
                    for z in self.cur_lineup[a_id]:
                        cur_f['cur_pred_a'] += self.cur_lineup[a_id][z] * self.priors[0][self.player_map[z]]
                        cur_f['cur_pred_h'] += self.cur_lineup[a_id][z] * self.priors[2][self.player_map[z]]
                    for z in self.last_lineup[h_id]:
                        cur_f['last_pred_h'] += self.last_lineup[h_id][z] * self.priors[0][self.player_map[z]]
                        cur_f['last_pred_a'] += self.last_lineup[h_id][z] * self.priors[2][self.player_map[z]]
                    for z in self.last_lineup[a_id]:
                        cur_f['last_pred_a'] += self.last_lineup[a_id][z] * self.priors[0][self.player_map[z]]
                        cur_f['last_pred_h'] += self.last_lineup[a_id][z] * self.priors[2][self.player_map[z]]
                
                    cur_f["actual_h"] = h_eff 
                    cur_f["actual_a"] = a_eff 
                    features.append(cur_f)

                self.last_lineup[h_id] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                    self.last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
                self.last_lineup[a_id] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                    self.last_lineup[a_id][row['player_id']] = row['seconds']/total_sec

            print (np.array(off_def['off']['obs']))
            print (np.array(off_def['off']['opp_eff']))
            #Updating below
            with pm.Model() as pymc_model:
                o_eff = pm.Normal('o_eff', mu=self.priors[0], sigma=self.priors[1], shape=len(self.player_map))
                d_eff = pm.Normal('d_eff', mu=self.priors[2], sigma=self.priors[3], shape=len(self.player_map))
                home = pm.Normal('home', mu=self.home_prior[0], sigma=self.home_prior[1], shape=1)

                mu_off = home*np.array(off_def['off']['h_ind']) + o_eff[off_def['off']['p_ids']]
                Y_obs_off = pm.Normal('Y_obs_off', mu=mu_off, sigma=off_def['off']['p_sec_inv'], observed=np.array(off_def['off']['obs']) - np.array(off_def['off']['opp_eff']))

                mu_def = home*np.array(off_def['def']['h_ind']) + o_eff[off_def['def']['p_ids']]
                Y_obs_def = pm.Normal('Y_obs_def', mu=mu_def, sigma=off_def['def']['p_sec_inv'], observed=np.array(off_def['def']['obs']) - np.array(off_def['def']['opp_eff']))
            
            compiled_model = nutpie.compile_pymc_model(pymc_model)
            trace_pymc = nutpie.sample(compiled_model, draws = 10000, seed=seed, chains=10, cores=8, save_warmup=False, progress_bar=False)
            print (az.summary(trace_pymc).iterrows())
            return
            for index, row in az.summary(trace_pymc).iterrows():
                if ('pace' in index):
                    i = int(index.split("[")[1].split(']')[0])
                    priors[0][i] = row['mean']
                    if (i in h_ids or i in a_ids):
                        priors[1][i] = min(row['sd'] * per_game_fatten, max_sigma)
                    else:
                        priors[1][i] = row['sd']

    def player_eff_bhm_lu(self, start_mu, start_sigma, max_sigma, obs_sigma, per_game_fatten_base, home_adv):
        self.load_data()
        self.init_priors(start_mu,start_sigma)
        self.init_lineup_track()

        features = []
        
        game_ids = self.games['game_id'].unique()       
        count = 0
        for gid in tqdm(game_ids):
            count+=1
            cur_game = self.player_bs.loc[self.player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = self.games.loc[self.games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = self.team_bs.loc[self.team_bs["game_id"] == gid,].reset_index()
            cur_lu = self.lu.loc[self.lu['game_id'] == gid,].reset_index()


            try:
                h_eff = team_game.at[0,"offensiveRating"]
                a_eff = team_game.at[1,"offensiveRating"]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            cur_h = cur_game.loc[cur_game['team_id']==h_id,]
            cur_a = cur_game.loc[cur_game['team_id']==a_id,]

            total_sec = cur_h['seconds'].sum()


            for x in [h_id, a_id]:
                self.cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    self.cur_lineup[x][row['player_id']] = row['seconds']/total_sec

                    #self.tracker[row['player_id']][date] = [self.priors[0][self.player_map[row['player_id']]], self.priors[1][self.player_map[row['player_id']]]]


            cur_f = {'last_pred_h':0,'cur_pred_h':0,'last_pred_a':0,'cur_pred_a':0}
            if (cur_season != "1996-97"):
                for z in self.cur_lineup[h_id]:
                    cur_f['cur_pred_h'] += self.cur_lineup[h_id][z] * self.priors[0][self.player_map[z]]
                    cur_f['cur_pred_a'] += self.cur_lineup[h_id][z] * self.priors[2][self.player_map[z]]
                for z in self.cur_lineup[a_id]:
                    cur_f['cur_pred_a'] += self.cur_lineup[a_id][z] * self.priors[0][self.player_map[z]]
                    cur_f['cur_pred_h'] += self.cur_lineup[a_id][z] * self.priors[2][self.player_map[z]]
                for z in self.last_lineup[h_id]:
                    cur_f['last_pred_h'] += self.last_lineup[h_id][z] * self.priors[0][self.player_map[z]]
                    cur_f['last_pred_a'] += self.last_lineup[h_id][z] * self.priors[2][self.player_map[z]]
                for z in self.last_lineup[a_id]:
                    cur_f['last_pred_a'] += self.last_lineup[a_id][z] * self.priors[0][self.player_map[z]]
                    cur_f['last_pred_h'] += self.last_lineup[a_id][z] * self.priors[2][self.player_map[z]]
            
                cur_f["actual_h"] = h_eff 
                cur_f["actual_a"] = a_eff 
                features.append(cur_f)

            self.last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                self.last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            self.last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                self.last_lineup[a_id][row['player_id']] = row['seconds']/total_sec


            for index, row in cur_lu.iterrows():
                if (row['end'] == row['start'] or pd.isnull(row['h_off_rtg'])):
                    continue
                self.expected_off[row['h_id']] = 0
                self.expected_off[row['a_id']] = 0
                self.expected_def[row['h_id']] = 0
                self.expected_def[row['a_id']] = 0
                missing_h = 0
                missing_a = 0
                for col in cur_lu.columns:
                    if ('h_player' in col and len(col) == 10 and '6' not in col):
                        if (row[col] > 0):
                            self.expected_off[row['h_id']] += self.priors[0][self.player_map[row[col]]]
                            self.expected_def[row['h_id']] += self.priors[2][self.player_map[row[col]]]
                        else:
                            missing_h += 1
                        
                    if ('a_player' in col and len(col) == 10 and '6' not in col):
                        if (row[col] > 0):
                            self.expected_off[row['a_id']] += self.priors[0][self.player_map[row[col]]]
                            self.expected_def[row['a_id']] += self.priors[2][self.player_map[row[col]]]
                        else:
                            missing_a += 1
                self.expected_off[row['h_id']] = self.expected_off[row['h_id']] * 5 / (5-missing_h)
                self.expected_def[row['h_id']] = self.expected_def[row['h_id']] * 5 / (5-missing_h)
                self.expected_off[row['a_id']] = self.expected_off[row['a_id']] * 5 / (5-missing_a)
                self.expected_def[row['a_id']] = self.expected_def[row['a_id']] * 5 / (5-missing_a)

                        
                for col in cur_lu.columns:
                    if ('h_player' in col and len(col) == 10 and '6' not in col and row[col] > 0):
                        obs_mu = row['h_off_rtg'] - self.expected_def[row['a_id']] - home_adv
                        obs_sd = (2880 / (row['end'] - row['start'])) * obs_sigma
                        prior_mu = self.priors[0][self.player_map[row[col]]]
                        prior_sd = self.priors[1][self.player_map[row[col]]]
                        self.priors[0][self.player_map[row[col]]] = (prior_mu*obs_sd**2 + obs_mu*prior_sd**2) / (obs_sd**2 + prior_sd**2)
                        self.priors[1][self.player_map[row[col]]] = min(math.sqrt((prior_sd**2*(obs_sd)**2) / (prior_sd**2 + (obs_sd)**2)) * (1 + per_game_fatten_base*((row['end'] - row['start'])/2880)), max_sigma)

                        obs_mu = row['a_off_rtg'] - self.expected_off[row['a_id']] + home_adv
                        obs_sd = (2880 / (row['end'] - row['start'])) * obs_sigma
                        prior_mu = self.priors[2][self.player_map[row[col]]]
                        prior_sd = self.priors[3][self.player_map[row[col]]]
                        self.priors[2][self.player_map[row[col]]] = (prior_mu*obs_sd**2 + obs_mu*prior_sd**2) / (obs_sd**2 + prior_sd**2)
                        self.priors[3][self.player_map[row[col]]] = min(math.sqrt((prior_sd**2*(obs_sd)**2) / (prior_sd**2 + (obs_sd)**2)) * (1 + per_game_fatten_base*((row['end'] - row['start'])/2880)), max_sigma)

                    if ('a_player' in col and len(col) == 10 and '6' not in col and row[col] > 0):
                        obs_mu = row['a_off_rtg'] - self.expected_def[row['h_id']] + home_adv
                        obs_sd = (2880 / (row['end'] - row['start'])) * obs_sigma
                        prior_mu = self.priors[0][self.player_map[row[col]]]
                        prior_sd = self.priors[1][self.player_map[row[col]]]
                        self.priors[0][self.player_map[row[col]]] = (prior_mu*obs_sd**2 + obs_mu*prior_sd**2) / (obs_sd**2 + prior_sd**2)
                        self.priors[1][self.player_map[row[col]]] = min(math.sqrt((prior_sd**2*(obs_sd)**2) / (prior_sd**2 + (obs_sd)**2)) * (1 + per_game_fatten_base*((row['end'] - row['start'])/2880)), max_sigma)

                        obs_mu = row['h_off_rtg'] - self.expected_off[row['h_id']] - home_adv
                        obs_sd = (2880 / (row['end'] - row['start'])) * obs_sigma
                        prior_mu = self.priors[2][self.player_map[row[col]]]
                        prior_sd = self.priors[3][self.player_map[row[col]]]
                        self.priors[2][self.player_map[row[col]]] = (prior_mu*obs_sd**2 + obs_mu*prior_sd**2) / (obs_sd**2 + prior_sd**2)
                        self.priors[3][self.player_map[row[col]]] = min(math.sqrt((prior_sd**2*(obs_sd)**2) / (prior_sd**2 + (obs_sd)**2)) * (1 + per_game_fatten_base*((row['end'] - row['start'])/2880)), max_sigma)                
            # for index, row in cur_lu.iterrows():
            #     h_players = []
            #     a_players = []
            #     all_players = []
            #     cur_map = {}
            #     for col in cur_lu.columns:
            #         if ('h_player' in col and len(col) == 10 and '6' not in col):
            #             h_players.append(self.player_map[row[col]])
            #             all_players.append(self.player_map[row[col]])
            #         elif ('a_player' in col and len(col) == 10 and '6' not in col):
            #             a_players.append(self.player_map[row[col]])
            #             all_players.append(self.player_map[row[col]])
            #     cur_priors_off_mu = self.priors[0][all_players]
            #     cur_priors_def_mu = self.priors[2][all_players]
            #     cur_priors_off_sd = self.priors[1][all_players]
            #     cur_priors_def_sd = self.priors[3][all_players]
            #     #Updating below
            #     with pm.Model() as pymc_model:
            #         o_eff = pm.Normal('o_eff', mu=cur_priors_off_mu, sigma=cur_priors_off_sd, shape=len(all_players))
            #         d_eff = pm.Normal('d_eff', mu=cur_priors_def_mu, sigma=cur_priors_def_sd, shape=len(all_players))
            #         home = pm.Normal('home', mu=self.home_prior[0], sigma=self.home_prior[1])

            #         mu_h = home
            #         for i in range(5):
            #             mu_h += o_eff[i]
            #         for i in range(5,10):
            #             mu_h += d_eff[i]
            #         Y_obs_h = pm.Normal('Y_obs_h', mu=mu_h, sigma=2880/(row['end']-row['start']), observed=row['h_off_rtg'])

            #         mu_a = 0
            #         for i in range(5):
            #             mu_a += d_eff[i]
            #         for i in range(5,10):
            #             mu_a += o_eff[i]
            #         Y_obs_a = pm.Normal('Y_obs_a', mu=mu_a, sigma=2880/(row['end']-row['start']), observed=row['a_off_rtg'])

            #         #mu_def = home*np.array(off_def['def']['h_ind']) + o_eff[off_def['def']['p_ids']]
            #         #Y_obs_def = pm.Normal('Y_obs_def', mu=mu_def, sigma=off_def['def']['p_sec_inv'], observed=np.array(off_def['def']['obs']) - np.array(off_def['def']['opp_eff']))
                
            #     compiled_model = nutpie.compile_pymc_model(pymc_model)
            #     trace_pymc = nutpie.sample(compiled_model, draws = 10000, seed=1, chains=10, cores=8, save_warmup=False, progress_bar=False)
            #     a = az.summary(trace_pymc)
                
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
        self.games = self.games[self.games["season"].str.contains(seasons[:-1])].reset_index()
        
        print (self.games)
        fdf = pd.DataFrame(features)
        print  (fdf)
        with open('./intermediates/BHM_eff_player_tracker_lu_v1.pkl', 'wb') as f:
            pickle.dump(tracker, f)
        self.games = pd.concat([self.games, fdf], axis=1)
        self.games.to_csv("./predictions/latent/eff_BHM_player_lu_v1.csv", index=False)
#a = eff_latent_models()
#a.player_eff_bhm_lu(10,1,2,1,0,0)

def player_eff_bhm(per_game_fatten, obs_sigma):
    ### Hyperparams
    start_mu = 105
    start_sigma = 4
    max_sigma = 10
    #per_game_fatten = 1.05
    #obs_sigma = 1
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
    p_n = len(players)

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    player_team = {}
    #priors[0] is for mus, every i/i+p_n pair is a players off/def rtg pair, priors[1] for sds
    priors = [[],[]]
    tracker = {}
    for i in range(p_n):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        player_map[players[i]] = i
        player_team[players[i]] = -1
        tracker[players[i]] = {}
    priors[0] = np.array(priors[0], dtype='float')
    priors[1] = np.array(priors[1], dtype='float')



    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()      

        #each element of game_side coincides with each element of obs and is a list of tuples like: (p_id, side, weight)
        #p_id is the player's index in the prior array, side is either 0 or 2 if they are on offense or defense for whatever is being updated, and weight is the proportion of their game time wrt all players
        #this way, each element of game_side contains all the information that is needed to update based on an observed eff
        game_side = []
        obs = []


        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_eff = team_game.at[0,'offensiveRating']
                a_team_eff = team_game.at[1,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec


            #For Home on offense
            gs_el = []
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],0,row['seconds']/2/total_sec))
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],2,row['seconds']/2/total_sec))
            game_side.append(gs_el)
            obs.append(h_team_eff)

            #for Away on offense
            gs_el = []
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],0,row['seconds']/2/total_sec))
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],2,row['seconds']/2/total_sec))
            game_side.append(gs_el)
            obs.append(a_team_eff)


            cur_f = {'last_pred_h_eff':0,'cur_pred_h_eff':0,'last_pred_a_eff':0,'cur_pred_a_eff':0}
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    for z in cur_lineup[x]:
                        if (x == h_id):
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]+p_n]
                        else:
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]+p_n]
                    for z in last_lineup[x]:
                        if (x == h_id):
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]+p_n]
                        else:
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]+p_n]
            
                cur_f["actual_h_eff"] = h_team_eff
                cur_f["actual_a_eff"] = a_team_eff   
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec


        for gs_num in range(len(game_side)):
            #We need to make a new temporary array of priors that include only players that played in the game
            weights = []
            ids = []
            map = {}
            rev_map = {}
            for i in range(len(game_side[gs_num])):
                if (game_side[gs_num][i][1] == 0):
                    ids.append(game_side[gs_num][i][0])
                    weights.append(game_side[gs_num][i][2])
                    map[game_side[gs_num][i][0]] = len(ids) - 1
                    rev_map[len(ids) - 1] = game_side[gs_num][i][0]
                else:
                    ids.append(game_side[gs_num][i][0]+p_n)
                    weights.append(game_side[gs_num][i][2])
                    map[game_side[gs_num][i][0]+p_n] = len(ids) - 1
                    rev_map[len(ids) - 1] = game_side[gs_num][i][0]+p_n
            new_mus = priors[0][ids]
            new_sds = priors[1][ids]

            weights = np.array(weights)

            cur_obs = obs[gs_num]

            #Update PYMC
            with pm.Model() as pymc_model:
                eff = pm.Normal('eff', mu=new_mus, sigma=new_sds, shape=len(new_mus))

                mu = sum(eff*weights)
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=obs_sigma, observed=[cur_obs])
            
            compiled_model = nutpie.compile_pymc_model(pymc_model)

            trace_pymc = nutpie.sample(compiled_model, draws = 1000, seed=1, chains=10, cores=8, save_warmup=False, progress_bar=False)
            print (az.summary(trace_pymc))
            #print (weights)
        return
        
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/player_eff_REAL_bhm.csv", index=False)

#PYMC Has mem leak so have to wrap it in multiprocessing
def run_pymc(new_mus, new_sds, all_weights, all_new_ids, obs_sigma, all_obs):
    with pm.Model() as pymc_model:
        eff = pm.Normal('eff', mu=new_mus, sigma=new_sds, shape=len(new_mus))

        mu = []
        for i in range(len(all_weights)):
            mu.append(sum(eff[all_new_ids[i]] * all_weights[i]))
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=obs_sigma, observed=all_obs)
        
    compiled_model = nutpie.compile_pymc_model(pymc_model)

    trace_pymc = nutpie.sample(compiled_model, draws = 1000, seed=1, chains=10, cores=8, save_warmup=False, progress_bar=False)

    return (trace_pymc)

def run_pymc_home(new_mus, new_sds, all_weights, all_new_ids, obs_sigma, all_obs, home_prior, home_inds):
    with pm.Model() as pymc_model:
        eff = pm.Normal('eff', mu=new_mus, sigma=new_sds, shape=len(new_mus))
        home = pm.Normal('home', mu=home_prior[0], sigma=home_prior[1], shape=1)

        mu = []
        for i in range(len(all_weights)):
            mu.append(sum(home*home_inds) + sum(eff[all_new_ids[i]] * all_weights[i]))

        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=obs_sigma, observed=all_obs)
        
    compiled_model = nutpie.compile_pymc_model(pymc_model)

    trace_pymc = nutpie.sample(compiled_model, draws = 1000, seed=1, chains=10, cores=8, save_warmup=False, progress_bar=False)

    return (trace_pymc)

#slower actually
def player_eff_bhm_fast(per_game_fatten, obs_sigma):
    ### Hyperparams
    start_mu = 105
    start_sigma = 4
    max_sigma = 6
    #per_game_fatten = 1.05
    #obs_sigma = 1
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
    p_n = len(players)

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    player_team = {}
    #priors[0] is for mus, every i/i+p_n pair is a players off/def rtg pair, priors[1] for sds
    priors = [[],[]]
    home = [1,1]
    tracker = {}
    for i in range(p_n):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        player_map[players[i]] = i
        player_team[players[i]] = -1
        tracker[players[i]] = {}
    priors[0] = np.array(priors[0], dtype='float')
    priors[1] = np.array(priors[1], dtype='float')



    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()      

        #each element of game_side coincides with each element of obs and is a list of tuples like: (p_id, side, weight,home_ind)
        #p_id is the player's index in the prior array, side is either 0 or 2 if they are on offense or defense for whatever is being updated, and weight is the proportion of their game time wrt all players
        #this way, each element of game_side contains all the information that is needed to update based on an observed eff
        game_side = []
        obs = []


        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_eff = team_game.at[0,'offensiveRating']
                a_team_eff = team_game.at[1,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec


            #For Home on offense
            gs_el = []
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],0,row['seconds']/2/total_sec,1))
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],2,row['seconds']/2/total_sec,1))
            game_side.append(gs_el)
            obs.append(h_team_eff)

            #for Away on offense
            gs_el = []
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],0,row['seconds']/2/total_sec,0))
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                gs_el.append((player_map[row['player_id']],2,row['seconds']/2/total_sec,0))
            game_side.append(gs_el)
            obs.append(a_team_eff)


            cur_f = {'last_pred_h_eff':0,'cur_pred_h_eff':0,'last_pred_a_eff':0,'cur_pred_a_eff':0}
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    for z in cur_lineup[x]:
                        if (x == h_id):
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]+p_n]
                        else:
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]+p_n]
                    for z in last_lineup[x]:
                        if (x == h_id):
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]+p_n]
                        else:
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]+p_n]
            
                cur_f["actual_h_eff"] = h_team_eff
                cur_f["actual_a_eff"] = a_team_eff   
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec


        tic = time.perf_counter()
        all_weights = []
        all_ids = []
        all_new_ids = []
        all_obs = []
        home_inds = []
        map = {}
        rev_map = {}
        for gs_num in range(len(game_side)):
            #We need to make a new temporary array of priors that include only players that played in the game
            weights = []
            new_ids = []
            for i in range(len(game_side[gs_num])):
                #if offense
                if (game_side[gs_num][i][1] == 0):
                    all_ids.append(game_side[gs_num][i][0])
                    new_ids.append(len(all_ids) - 1)
                    weights.append(game_side[gs_num][i][2])
                    map[game_side[gs_num][i][0]] = len(all_ids) - 1
                    rev_map[len(all_ids) - 1] = game_side[gs_num][i][0]
                #if defense
                else:
                    all_ids.append(game_side[gs_num][i][0]+p_n)
                    new_ids.append(len(all_ids) - 1)
                    weights.append(game_side[gs_num][i][2])
                    map[game_side[gs_num][i][0]+p_n] = len(all_ids) - 1
                    rev_map[len(all_ids) - 1] = game_side[gs_num][i][0]+p_n
            
            all_weights.append(np.array(weights))
            all_new_ids.append(new_ids)
            all_obs.append(obs[gs_num])

            if (game_side[gs_num][0][3] == 1):
                home_inds.append(1)
            else:
                home_inds.append(0)

        home_inds = np.array(home_inds)
        new_mus = priors[0][all_ids]
        new_sds = priors[1][all_ids]


        #Update PYMC
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_pymc_home, new_mus, new_sds, all_weights, all_new_ids, obs_sigma, all_obs, home, home_inds)
            trace_pymc = future.result()
            # print (az.summary(trace_pymc))
            # return
        
        for index, row in az.summary(trace_pymc).iterrows():
            if ('eff' in index):
                new_ind = int(index.split("[")[1].split("]")[0])
                priors[0][rev_map[new_ind]] = row['mean']
                priors[1][rev_map[new_ind]] = row['sd'] * per_game_fatten
            if ('home' in index):
                home[0] = row['mean']
                home[1] = row['sd'] * 1.05
        print (home)

        
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/player_eff_REAL_bhm_home.csv", index=False)

def player_naive_bhm_usg(per_game_fatten, obs_sigma, home_adv, b2b_pen, arma_weight):
    ### Hyperparams
    start_mu = 52.5
    start_sigma = 4
    max_sigma = 10
    #per_game_fatten = 1.05
    #obs_sigma = 1
    #per_season_fatten = 2
    seed = 1

    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    player_map = {}
    player_team = {}
    priors = [[],[],[],[]]
    tracker = {}
    usg = {}
    for i in range(len(players)):
        priors[0].append(start_mu)
        priors[1].append(start_sigma)
        priors[2].append(start_mu)
        priors[3].append(start_sigma)
        player_map[players[i]] = i
        player_team[players[i]] = -1
        tracker[players[i]] = {}
        usg[players[i]] = 0.2
    priors[0] = np.array(priors[0], dtype='float')
    priors[1] = np.array(priors[1], dtype='float')
    priors[2] = np.array(priors[0], dtype='float')
    priors[3] = np.array(priors[1], dtype='float')


    last_lineup = {}
    cur_lineup = {}
    expected_off_eff = {}
    expected_def_eff = {}
    last_game = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}
        expected_off_eff[team] = 50
        expected_def_eff[team] = 50
        last_game[team] = datetime.datetime(1800,1,1)

    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       
        
        p_ids=[]
        p_sec_inv=[]
        opp_team_ids=[]
        obs = []
        off_def = []

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_eff = team_game.at[0,'offensiveRating']
                a_team_eff = team_game.at[1,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = cur_game['team_id'].unique()[0]
            a_id = cur_game['team_id'].unique()[1]

            if ((date-last_game[h_id]).days == 1):
                h_b2b = True
            else:
                h_b2b = False

            if ((date-last_game[a_id]).days == 1):
                a_b2b = True
            else:
                a_b2b = False
            
            last_game[h_id] = date
            last_game[a_id] = date

            cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                expected_off_eff[x] = 0
                expected_def_eff[x] = 0
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec

                    expected_off_eff[x] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

                    p_ids.append(player_map[row['player_id']])
                    if (row['seconds'] != 0):
                        p_sec_inv.append(((total_sec/5) / row['seconds']))
                    else:
                        p_sec_inv.append(100)
                    if (x == h_id):
                        opp_team_ids.append(a_id)
                    else:
                        opp_team_ids.append(h_id)
                    if (x == h_id):
                        obs.append(row['offensiveRating'] * h_team_eff/avg_h_off_player_eff - home_adv)
                        if (h_b2b):
                            obs[-1] += b2b_pen
                        if (a_b2b):
                            obs[-1] -= b2b_pen
                    else:
                        obs.append(row['offensiveRating'] * a_team_eff/avg_a_off_player_eff + home_adv)
                        if (h_b2b):
                            obs[-1] -= b2b_pen
                        if (a_b2b):
                            obs[-1] += b2b_pen
                    off_def.append('off')

                    expected_def_eff[x] += (row['seconds']/total_sec) * priors[2][player_map[row['player_id']]]

                    p_ids.append(player_map[row['player_id']])
                    if (row['seconds'] != 0):
                        p_sec_inv.append(((total_sec/5) / row['seconds']))
                    else:
                        p_sec_inv.append(100)
                    if (x == h_id):
                        opp_team_ids.append(a_id)
                    else:
                        opp_team_ids.append(h_id)
                    if (x == h_id):
                        obs.append(row['defensiveRating'] * a_team_eff/avg_h_def_player_eff + home_adv)
                        if (h_b2b):
                            obs[-1] -= b2b_pen
                        if (a_b2b):
                            obs[-1] += b2b_pen
                    else:
                        obs.append(row['defensiveRating'] * h_team_eff/avg_a_def_player_eff - home_adv)
                        if (h_b2b):
                            obs[-1] += b2b_pen
                        if (a_b2b):
                            obs[-1] -= b2b_pen
                    off_def.append('def')

    
            cur_f = {'last_pred_h_eff':home_adv,'cur_pred_h_eff':home_adv,'last_pred_a_eff':-home_adv,'cur_pred_a_eff':-home_adv}
            if (h_b2b):
                cur_f['last_pred_h_eff'] -= b2b_pen
                cur_f['cur_pred_h_eff'] -= b2b_pen
                cur_f['last_pred_a_eff'] += b2b_pen
                cur_f['cur_pred_a_eff'] += b2b_pen
            if (a_b2b):
                cur_f['last_pred_a_eff'] -= b2b_pen
                cur_f['cur_pred_a_eff'] -= b2b_pen
                cur_f['last_pred_h_eff'] += b2b_pen
                cur_f['cur_pred_h_eff'] += b2b_pen
            if (cur_season != "1996-97"):
                for x in [h_id,a_id]:
                    usg_time_weight = {}
                    total = 0
                    for z in cur_lineup[x]:
                        usg_time_weight[z] = cur_lineup[x][z]*usg[z]
                        total += cur_lineup[x][z]*usg[z]
                    for z in usg_time_weight:
                        usg_time_weight[z] = usg_time_weight[z] / total
                    for z in cur_lineup[x]:
                        if (x == h_id):
                            cur_f['cur_pred_h_eff'] += usg_time_weight[z] * priors[0][player_map[z]]
                            cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
                        else:
                            cur_f['cur_pred_a_eff'] += usg_time_weight[z] * priors[0][player_map[z]]
                            cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]

                    usg_time_weight = {}
                    total = 0
                    for z in last_lineup[x]:
                        usg_time_weight[z] = last_lineup[x][z]*usg[z]
                        total += last_lineup[x][z]*usg[z]
                    for z in usg_time_weight:
                        usg_time_weight[z] = usg_time_weight[z] / total
                    for z in last_lineup[x]:
                        if (x == h_id):
                            cur_f['last_pred_h_eff'] += usg_time_weight[z] * priors[0][player_map[z]]
                            cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
                        else:
                            cur_f['last_pred_a_eff'] += usg_time_weight[z] * priors[0][player_map[z]]
                            cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            
                cur_f["actual_h_eff"] = h_team_eff
                cur_f["actual_a_eff"] = a_team_eff   
                features.append(cur_f)

            last_lineup[h_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
                last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
                time_played = row['seconds'] / total_sec
                usg[row['player_id']] += (row['usagePercentage'] - usg[row['player_id']]) * arma_weight * time_played
            last_lineup[a_id] = {}
            for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
                last_lineup[a_id][row['player_id']] = row['seconds']/total_sec
                time_played = row['seconds'] / total_sec
                usg[row['player_id']] += (row['usagePercentage'] - usg[row['player_id']]) * arma_weight * time_played


        opp_off_eff = []
        opp_def_eff = []
        for opp_id in opp_team_ids:
            opp_off_eff.append(expected_off_eff[opp_id])
            opp_def_eff.append(expected_def_eff[opp_id])

        for i in range(len(p_ids)):
            if (p_sec_inv[i] > 25):
                continue
            if (off_def[i] == 'off'):
                priors[0][p_ids[i]] = (priors[0][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_def_eff[i])*priors[1][p_ids[i]]**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
                priors[1][p_ids[i]] = min(math.sqrt((priors[1][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)
            else:
                priors[2][p_ids[i]] = (priors[2][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_off_eff[i])*priors[3][p_ids[i]]**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
                priors[3][p_ids[i]] = min(math.sqrt((priors[3][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)

    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/player_eff_bhm_"+str(obs_sigma)+"_"+str(per_game_fatten)+"_"+str(home_adv)+"_"+str(b2b_pen)+"_"+str(arma_weight)+".csv", index=False)
