import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats

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
        raw[team] = []
        for weight in arma:
            arma[weight][team] = 42.5
        for weight in bayes:
            bayes[weight][team] = 42.5
        seas[team] = {}
        for season in games["season"].unique():
            seas[team][season] = []
    
    for gid in tqdm(game_ids):
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        pri_season = games["season"].unique()[games["season"].unique().tolist().index(cur_season)-1]

        try:
            pace = cur_game.at[0,"pace"]
        except:
            if (cur_season != "1996-97"):
                cur_f = {}
                cur_f["season_avg"] = np.nan
                cur_f["ma_5"] = np.nan
                cur_f["ma_10"] = np.nan
                cur_f["ma_30"] = np.nan
                cur_f["actual"] = np.nan
                features.append(cur_f)
            continue

        cur_f = {}
        if (cur_season != "1996-97"):
            hn = len(seas[cur_game.at[0,"team_id"]][cur_season])
            an = len(seas[cur_game.at[1,"team_id"]][cur_season])
            if (hn < 5):
                if (hn == 0):
                    h = np.average(seas[cur_game.at[0,"team_id"]][pri_season])
                else:
                    h = (5-hn)/5 * np.average(seas[cur_game.at[0,"team_id"]][pri_season]) + hn/5 * np.average(seas[cur_game.at[0,"team_id"]][cur_season])
            else:
                h = np.average(seas[cur_game.at[0,"team_id"]][cur_season])
            if (an < 5):
                if (an == 0):
                    a = np.average(seas[cur_game.at[1,"team_id"]][pri_season])
                else:
                    a = (5-an)/5 * np.average(seas[cur_game.at[1,"team_id"]][pri_season]) + an/5 * np.average(seas[cur_game.at[1,"team_id"]][cur_season])
            else:
                a = np.average(seas[cur_game.at[1,"team_id"]][cur_season])
            cur_f["season_avg"] = (h + a) / 2

            h = np.average(raw[cur_game.at[0,"team_id"]][-5:])
            a = np.average(raw[cur_game.at[1,"team_id"]][-5:])
            cur_f["ma_5"] = (h+a) / 2

            h = np.average(raw[cur_game.at[0,"team_id"]][-10:])
            a = np.average(raw[cur_game.at[1,"team_id"]][-10:])
            cur_f["ma_10"] = (h+a) / 2

            h = np.average(raw[cur_game.at[0,"team_id"]][-30:])
            a = np.average(raw[cur_game.at[1,"team_id"]][-30:])
            cur_f["ma_30"] = (h+a) / 2

            for weight in arma:
                cur_f["arma_"+str(weight)] = arma[weight][cur_game.at[0,"team_id"]] + arma[weight][cur_game.at[1,"team_id"]]
            
            for weight in bayes:
                cur_f["bayes_"+str(weight)] = bayes[weight][cur_game.at[0,"team_id"]] + bayes[weight][cur_game.at[1,"team_id"]]
        
            cur_f["actual"] = pace     
            features.append(cur_f)

        #Updating below

        for i in range(2):
            raw[cur_game.at[i,"team_id"]].append(pace)
            seas[cur_game.at[i,"team_id"]][cur_season].append(pace)

        for weight in arma:
            error_term = pace - arma[weight][cur_game.at[0,"team_id"]] - arma[weight][cur_game.at[1,"team_id"]]
            for i in range(2):
                arma[weight][cur_game.at[i,"team_id"]] += error_term*weight

        for weight in bayes:
            for i in range(2):
                bayes[weight][cur_game.at[i,"team_id"]] = (1-weight) * bayes[weight][cur_game.at[i,"team_id"]] + weight * pace/2
    
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace.csv", index=False)

def player_arma():
    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    game_ids = games["game_id"].to_list()

    player_bs = player_bs[player_bs["game_id"].isin(game_ids)]
    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    arma = {}
    arma_weights = [0.7,0.5,0.3,0.2,0.1,0.05]
    last_lineup = {}
    for team in team_bs["team_id"].unique():
        last_lineup[team] = {}
        arma[team] = {}
        for player in player_bs.loc[player_bs["team_id"] == team, ]["player_id"].unique():
            arma[team][player] = {}
            for weight in [0.7,0.5,0.3,0.2,0.1,0.05]:
                arma[team][player][weight] = 95

    features = []

    for gid in tqdm(game_ids):
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        try:
            team_pace = team_bs.loc[team_bs["game_id"]==gid,].reset_index().at[0, 'pace']
        except:
            if (cur_season != "1996-97"):
                features.append({})
            continue

        psubset = player_bs.loc[player_bs["game_id"]==gid,].dropna().reset_index()
        seconds_played = psubset['seconds'].sum()
        player_pace_total = (psubset['seconds'] * psubset['pace']).sum() / seconds_played
        teams = psubset['team_id'].unique()

        pred_pace = {}
        for weight in [0.7,0.5,0.3,0.2,0.1,0.05,0.01]:
            pred_pace[weight] = {teams[0]:0,teams[1]:0}

        cur_f = {}
        for a in ['fk_','lk_']:
            for weight in arma_weights:
                cur_f[a+'arma_'+str(weight)] = 0
        if (cur_season != "1996-97"):
            total_sec = psubset['seconds'].sum()
            for i in range(len(psubset.index)):
                for weight in arma_weights:
                    cur_f['fk_arma_'+str(weight)] += arma[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] * (psubset.at[i, 'seconds']/total_sec)
                    pred_pace[weight][psubset.at[i, 'team_id']] += 2 * arma[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] * (psubset.at[i, 'seconds']/total_sec)
            
            for tid in teams:
                total_last_sec = sum(last_lineup[tid].values())
                for pid in last_lineup[tid]:
                    if (last_lineup[tid][pid] > 0):
                        for weight in arma_weights:
                            cur_f['lk_arma_'+str(weight)] += arma[tid][pid][weight] / 2 * (last_lineup[tid][pid]/total_last_sec)

            cur_f['actual'] = team_pace
            features.append(cur_f)

        if (cur_season == '1996-97'):
            total_sec = psubset['seconds'].sum()
            for i in range(len(psubset.index)):
                for weight in arma_weights:
                    pred_pace[weight][psubset.at[i, 'team_id']] += 2 * arma[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] * (psubset.at[i, 'seconds']/total_sec)

        for tid in teams:
            last_lineup[tid] = {}
        for i in range(len(psubset.index)):
            for weight in arma_weights:
                error_term = (psubset.at[i,'pace']*2 - pred_pace[weight][psubset.at[i, 'team_id']] - arma[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight]) * (team_pace/player_pace_total)
                arma[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] += error_term*weight
            last_lineup[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']] = psubset.at[i, 'seconds']
    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace_players_arma.csv", index=False)

def player_bayes():
    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    game_ids = games["game_id"].to_list()

    player_bs = player_bs[player_bs["game_id"].isin(game_ids)]
    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    bayes = {}
    bayes_weights = [0.6,0.45,0.3,0.2,0.1,0.05]
    last_lineup = {}
    for team in team_bs["team_id"].unique():
        last_lineup[team] = {}
        bayes[team] = {}
        for player in player_bs.loc[player_bs["team_id"] == team, ]["player_id"].unique():
            bayes[team][player] = {}
            for weight in bayes_weights:
                bayes[team][player][weight] = 95

    features = []

    for gid in tqdm(game_ids):
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        try:
            team_pace = team_bs.loc[team_bs["game_id"]==gid,].reset_index().at[0, 'pace']
        except:
            if (cur_season != "1996-97"):
                features.append({})
            continue

        psubset = player_bs.loc[player_bs["game_id"]==gid,].dropna().reset_index()
        seconds_played = psubset['seconds'].sum()
        player_pace_total = (psubset['seconds'] * psubset['pace']).sum() / seconds_played
        teams = psubset['team_id'].unique()

        cur_f = {}
        for a in ['fk_','lk_']:
            for weight in bayes_weights:
                cur_f[a+'bayes_'+str(weight)] = 0
        if (cur_season != "1996-97"):
            total_sec = psubset['seconds'].sum()
            for i in range(len(psubset.index)):
                for weight in bayes_weights:
                    cur_f['fk_bayes_'+str(weight)] += bayes[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] * (psubset.at[i, 'seconds']/total_sec)
            
            for tid in teams:
                total_last_sec = sum(last_lineup[tid].values())
                for pid in last_lineup[tid]:
                    if (last_lineup[tid][pid] > 0):
                        for weight in bayes_weights:
                            cur_f['lk_bayes_'+str(weight)] += bayes[tid][pid][weight] / 2 * (last_lineup[tid][pid]/total_last_sec)

            cur_f['actual'] = team_pace
            features.append(cur_f)

        for tid in teams:
            last_lineup[tid] = {}
        for i in range(len(psubset.index)):
            for weight in bayes_weights:
                cur_weight = weight * psubset.at[i,'seconds'] / (48*60)
                bayes[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] = (1-cur_weight) * bayes[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']][weight] + cur_weight * (team_pace/player_pace_total) * psubset.at[i,'pace']
            last_lineup[psubset.at[i, 'team_id']][psubset.at[i, 'player_id']] = psubset.at[i, 'seconds']
    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/pace_players_bayes.csv", index=False)

def additional_factors_modeling():
    df = pd.read_csv('./processed/pace_arma_additional.csv').dropna()
    df["intercept"] = 1
    results = sm.OLS(df['error'], df[['intercept','h_game_density','a_game_density']]).fit()
    print (results.summary())

def additional_factors_processing():
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    games['game_date'] = pd.to_datetime(games['game_date'])
    game_ids = games["game_id"].to_list()

    arma = {0.1:{}}

    game_hist = {}
 

    features = []

    for team in games["h_team_id"].unique():
        for weight in arma:
            arma[weight][team] = 42.5
        game_hist[team] = [(datetime.datetime(1960,1,1),"H")]
    
    for gid in tqdm(game_ids):
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        date = games.loc[games["game_id"]==gid, ].reset_index().at[0, 'game_date']

        try:
            pace = cur_game.at[0,"pace"]
        except:
            if (cur_season != "1996-97"):
                cur_f = {}
                features.append(cur_f)
            continue

        cur_f = {}
        if (cur_season != "1996-97"):
            for weight in arma:
                cur_f["arma_"+str(weight)] = arma[weight][cur_game.at[0,"team_id"]] + arma[weight][cur_game.at[1,"team_id"]]

            cur_f['h_last_game'] = min((date - game_hist[cur_game.at[0,"team_id"]][-1][0]).days,7)
            cur_f['a_last_game'] = min((date - game_hist[cur_game.at[1,"team_id"]][-1][0]).days,7)

            if ((date - game_hist[cur_game.at[0,"team_id"]][-1][0]).days == 1):
                cur_f["h_b2b"] = 1
            else:
                cur_f["h_b2b"] = 0
            if ((date - game_hist[cur_game.at[1,"team_id"]][-1][0]).days == 1):
                cur_f["a_b2b"] = 1
            else:
                cur_f["a_b2b"] = 0

            count = 0
            for i in range(1,8):
                try:
                    if ((date - game_hist[cur_game.at[0,"team_id"]][-i][0]).days < 7):
                        count += 1
                except IndexError:
                    break
            cur_f['h_game_density'] = count

            count = 0
            for i in range(1,8):
                try:
                    if ((date - game_hist[cur_game.at[1,"team_id"]][-i][0]).days < 7):
                        count += 1
                except:
                    break
            cur_f['a_game_density'] = count
        
            cur_f["actual"] = pace     
            features.append(cur_f)

        #Updating below

        for weight in arma:
            error_term = pace - arma[weight][cur_game.at[0,"team_id"]] - arma[weight][cur_game.at[1,"team_id"]]
            cur_f['error'] = error_term
            for i in range(2):
                arma[weight][cur_game.at[i,"team_id"]] += error_term*weight

        for i in range(2):
            if (i == 0):
                game_hist[cur_game.at[i,"team_id"]].append((date,"H"))
            else:
                game_hist[cur_game.at[i,"team_id"]].append((date,"A"))
    
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)

    games.to_csv("./processed/pace_arma_additional.csv", index=False)

#Generates figures from the naive_pace models
def team_visualizations():
    df = pd.read_csv("./predictions/pace.csv").dropna()

    seasons = ""
    for yr in range(1998, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    methods = ["season_avg","ma_5","ma_10","ma_30","arma_0.5","arma_0.25","arma_0.1","arma_0.05","arma_0.01"
               ,"bayes_0.5","bayes_0.25","bayes_0.1","bayes_0.05","bayes_0.01"]
    t_data = []

    for m in methods:
        df[m+"_AE"] = abs(df[m] - df["actual"])
        df[m+"_SE"] = (df[m] - df["actual"])**2

        t_data.append([df[m+"_AE"].mean(),df[m+"_SE"].mean()])
        
    
    t_df = pd.DataFrame(
        index = ["Mean Absolute Error", "Mean Squared Error"],
        columns = methods,
        data = np.array(t_data).T
    ).round(2)

    fig = go.Figure(go.Table(header={"values":t_df.reset_index().columns, "font":{"size":5}, "align":"left"},
                  cells={"values":t_df.reset_index().T, "align":"left", "font":{"size":7}}))
    
    fig.write_image("./figures/pace/naive_table.png")

    fig, ax = plt.subplots()
    ax.scatter(df["season_avg"],df["actual"], s = 0.1)
    ax.axline((90, 90), slope=1, c="k")
    ax.title.set_text("Season-Averaged Pace Predictions vs Actual")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Actual Pace")
    fig.savefig("./figures/pace/season_avg_plot.png")

    df["season_avg_residual"] = df["actual"] - df["season_avg"]
    fig, ax = plt.subplots()
    ax.scatter(df["season_avg"],df["season_avg_residual"], s = 0.1)
    ax.axline((90, 0), slope=0, c="k")
    r, p = stats.pearsonr(df["season_avg"], df["season_avg_residual"])
    plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.95), xycoords='axes fraction')
    ax.title.set_text("Season-Averaged Pace Residual Plot")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Residual")
    fig.savefig("./figures/pace/season_avg_resid_plot.png")

    fig, ax = plt.subplots()
    ax.scatter(df["ma_10"],df["actual"], s = 0.1)
    ax.axline((90, 90), slope=1, c="k")
    ax.title.set_text("10-Game Moving Average Pace Predictions vs Actual")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Actual Pace")
    fig.savefig("./figures/pace/ma10_plot.png")

    df["ma10_residual"] = df["actual"] - df["ma_10"]
    fig, ax = plt.subplots()
    ax.scatter(df["ma_10"],df["ma10_residual"], s = 0.1)
    ax.axline((90, 0), slope=0, c="k")
    r, p = stats.pearsonr(df["ma_10"], df["ma10_residual"])
    plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.95), xycoords='axes fraction')
    ax.title.set_text("10-Game Moving Average Pace Residual Plot")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Residual")
    fig.savefig("./figures/pace/ma10_resid_plot.png")

    fig, ax = plt.subplots()
    ax.scatter(df["arma_0.1"],df["actual"], s = 0.1)
    ax.axline((90, 90), slope=1, c="k")
    ax.title.set_text("ARMA Weight-0.1 Pace Predictions vs Actual")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Actual Pace")
    fig.savefig("./figures/pace/arma_0.1_plot.png")

    df["arma_0.1_residual"] = df["actual"] - df["arma_0.1"]
    fig, ax = plt.subplots()
    ax.scatter(df["arma_0.1"],df["arma_0.1_residual"], s = 0.1)
    ax.axline((90, 0), slope=0, c="k")
    r, p = stats.pearsonr(df["arma_0.1"], df["arma_0.1_residual"])
    plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.95), xycoords='axes fraction')
    ax.title.set_text("ARMA Weight-0.1 Pace Residual Plot")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Residual")
    fig.savefig("./figures/pace/arma_0.1_resid_plot.png")

    fig, ax = plt.subplots()
    ax.scatter(df["bayes_0.25"],df["actual"], s = 0.1)
    ax.axline((90, 90), slope=1, c="k")
    ax.title.set_text("Bayes Weight-0.25 Pace Predictions vs Actual")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Actual Pace")
    fig.savefig("./figures/pace/bayes_0.25_plot.png")

    df["bayes_0.25_residual"] = df["actual"] - df["bayes_0.25"]
    fig, ax = plt.subplots()
    ax.scatter(df["bayes_0.25"],df["bayes_0.25_residual"], s = 0.1)
    ax.axline((90, 0), slope=0, c="k")
    r, p = stats.pearsonr(df["bayes_0.25"], df["bayes_0.25_residual"])
    plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.95), xycoords='axes fraction')
    ax.title.set_text("ARMA Weight-0.1 Pace Residual Plot")
    ax.set_xlabel("Predicted Pace")
    ax.set_ylabel("Residual")
    fig.savefig("./figures/pace/bayes_0.25_resid_plot.png")

def player_visualizations():
    df = pd.read_csv("./predictions/pace_players_arma.csv").dropna()

    seasons = ""
    for yr in range(1998, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    methods = []
    for x in ['fk_','lk_']:
        for w in [0.7,0.5,0.3,0.2,0.1,0.05]:
            methods.append(x+'arma_'+str(w))
    t_data = []

    for m in methods:
        df[m+"_AE"] = abs(df[m] - df["actual"])
        df[m+"_SE"] = (df[m] - df["actual"])**2

        t_data.append([df[m+"_AE"].mean(),df[m+"_SE"].mean()])
        
    
    t_df = pd.DataFrame(
        index = ["Mean Absolute Error", "Mean Squared Error"],
        columns = methods,
        data = np.array(t_data).T
    ).round(2)

    fig = go.Figure(go.Table(header={"values":t_df.reset_index().columns, "font":{"size":5}, "align":"left"},
                  cells={"values":t_df.reset_index().T, "align":"left", "font":{"size":7}}))
    
    fig.write_image("./figures/pace/naive_arma_player_table.png")

    #------------------------------------------------------------------------------------------------------------------

    df = pd.read_csv("./predictions/pace_players_bayes.csv").dropna()

    bayes_weights = [0.6,0.45,0.3,0.2,0.1,0.05]

    seasons = ""
    for yr in range(1998, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    df = df[df["season"].str.contains(seasons[:-1])]

    methods = []
    for x in ['fk_','lk_']:
        for w in bayes_weights:
            methods.append(x+'bayes_'+str(w))
    t_data = []

    for m in methods:
        df[m+"_AE"] = abs(df[m] - df["actual"])
        df[m+"_SE"] = (df[m] - df["actual"])**2

        t_data.append([df[m+"_AE"].mean(),df[m+"_SE"].mean()])
        
    
    t_df = pd.DataFrame(
        index = ["Mean Absolute Error", "Mean Squared Error"],
        columns = methods,
        data = np.array(t_data).T
    ).round(2)

    fig = go.Figure(go.Table(header={"values":t_df.reset_index().columns, "font":{"size":5}, "align":"left"},
                  cells={"values":t_df.reset_index().T, "align":"left", "font":{"size":7}}))
    
    fig.write_image("./figures/pace/naive_bayes_player_table.png")

    

    # fig, ax = plt.subplots()
    # ax.scatter(df["bayes_0.25"],df["actual"], s = 0.1)
    # ax.axline((90, 90), slope=1, c="k")
    # ax.title.set_text("Bayes Weight-0.25 Pace Predictions vs Actual")
    # ax.set_xlabel("Predicted Pace")
    # ax.set_ylabel("Actual Pace")
    # fig.savefig("./figures/pace/bayes_0.25_plot.png")

    # df["bayes_0.25_residual"] = df["actual"] - df["bayes_0.25"]
    # fig, ax = plt.subplots()
    # ax.scatter(df["bayes_0.25"],df["bayes_0.25_residual"], s = 0.1)
    # ax.axline((90, 0), slope=0, c="k")
    # r, p = stats.pearsonr(df["bayes_0.25"], df["bayes_0.25_residual"])
    # plt.annotate('r = {:.2f}'.format(r), xy=(0.8, 0.95), xycoords='axes fraction')
    # ax.title.set_text("ARMA Weight-0.1 Pace Residual Plot")
    # ax.set_xlabel("Predicted Pace")
    # ax.set_ylabel("Residual")
    # fig.savefig("./figures/pace/bayes_0.25_resid_plot.png")

#generates a bar chart showing each player's average pace for the season for the 2010-11 OKC Thunder
def pace_within_team():
    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(2010, 2011):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    h_games = games.loc[games["h_team_id"]==1610612760, ]
    a_games = games.loc[games["a_team_id"]==1610612760, ]
    games = pd.concat([h_games,a_games]).sort_values("game_date")
    game_ids = games["game_id"].to_list()

    team_pace = []
    players = {}
    player_bs = player_bs.loc[player_bs["team_id"]==1610612760,]
    player_bs = player_bs[player_bs["game_id"].isin(game_ids)]
    player_list = player_bs["player_id"].unique()
    for p in player_list:
        players[p] = {"pace":0,"min":0}

    for gid in game_ids:
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        team_pace.append(cur_game.at[0,"pace"])
        
        cur_game = player_bs.loc[player_bs["game_id"]==gid,].reset_index()
        for i in range(len(cur_game.index)):
            if (cur_game.at[i, "pace"] != 0):
                players[cur_game.at[i, "player_id"]]['pace'] += cur_game.at[i, "pace"] * int(cur_game.at[i, "minutes"].split(":")[0])
                players[cur_game.at[i, "player_id"]]['min'] += int(cur_game.at[i, "minutes"].split(":")[0])

    players_df = pd.read_csv("./database/players.csv")
    averages = {}
    total = 0
    total_min = 0
    for p in players:
        if (players[p]['min'] > 800):
            name = players_df.loc[players_df["id"]==p,].reset_index().at[0, "full_name"]
            averages[name] = players[p]['pace'] / players[p]['min']
            total += players[p]['pace']
            total_min += players[p]['min']
    

    plt.bar(list(averages.keys()), list(averages.values()))
    plt.ylim(90,100)
    plt.axhline(y=np.average(team_pace), color='k')
    plt.axhline(y=total/total_min, color='r')
    plt.title("Average Pace Amongst the 2010-2011 OKC Thunder")
    plt.xlabel("Player")
    plt.ylabel("Average Pace for the Season")
    plt.figtext(0.75,0.8,"Average Pace given by Player Boxscores",color="r")
    plt.figtext(0.75,0.78,"Average Pace given by Team Boxscores",color="k")
    plt.show()
    
