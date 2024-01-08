import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import datetime

#Generates figures from the naive_pace models
def naive_pace_team():
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

def naive_pace_player():
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
def player_pace_within_team():
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
    
def BHM_tracker():
    pred = pd.read_csv('./predictions/pace_BHM_base_v4.csv').dropna()
    pred["AE"] = abs(pred["pred"] - pred["actual"])
    pred["SE"] = (pred["pred"] - pred["actual"])**2
    abs_err = np.average(pred['AE']).round(2)
    sq_err = np.average(pred['SE']).round(2)

    teams = pd.read_csv('./database/teams.csv')
    team_map = {}
    for index, row in teams.iterrows():
        team_map[row['id']] = row['full_name']

    with open('./intermediates/BHM_base_tracker_v4.pkl', 'rb') as f:
        tracker = pickle.load(f)
    
    formatted = {}
    for key in tracker:
        formatted[key] = {"date":[],"mean":[],"sd":[]}
        for dk in tracker[key]:
            formatted[key]['date'].append(datetime.datetime.strptime(dk, "%Y-%m-%d"))
            formatted[key]['mean'].append(tracker[key][dk][0])
            formatted[key]['sd'].append(tracker[key][dk][1])
    for key in formatted:
        #if (key == list(formatted.keys())[5]):
            #break
        plt.plot(formatted[key]['date'], formatted[key]['mean'], label=team_map[key])
        plt.fill_between(formatted[key]['date'], np.array(formatted[key]['mean']) - 2*np.array(formatted[key]['sd']), np.array(formatted[key]['mean']) + 2*np.array(formatted[key]['sd']), alpha=0.1)
    plt.figtext(0.8,0.85,"Absolute Error: "+str(abs_err), fontsize=10)
    plt.figtext(0.8,0.82,"Squared Error: "+str(sq_err), fontsize=10)
    plt.legend(loc = 'upper left',fontsize = 'xx-small')
    plt.xlabel("Date")
    plt.ylabel("Pace Rating")
    plt.title("V4 Bayesian Hierarchical Model 1996-97 thru 2002-03")
    plt.show()



BHM_tracker()