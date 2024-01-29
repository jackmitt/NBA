import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import nutpie
import pickle
import math

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

team()