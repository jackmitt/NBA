from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscoremiscv3
from nba_api.stats.endpoints import boxscorescoringv3
from nba_api.stats.endpoints import boxscoreusagev3
from nba_api.stats.endpoints import boxscorefourfactorsv3
from nba_api.stats.endpoints import playbyplayv3
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from os.path import exists
import os
import time

db_path = "./database/"

def teams_table():
    pd.DataFrame(teams.get_teams()).to_csv(db_path + "teams.csv", index=False)

def players_table():
    pd.DataFrame(players.get_players()).to_csv(db_path + "players.csv", index=False)

def games_table(from_year=1990):
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(from_year, 2023)]
    teams = pd.read_csv(db_path + "teams.csv").id.to_list()
    types = ["Regular Season","Pre Season","Playoffs","Preseason"]
    list = []
    for season in tqdm(seasons):
        for team in teams:
            for type in types:
                time.sleep(1)
                for game in teamgamelog.TeamGameLog(season=season,season_type_all_star=type,team_id=team).get_dict()["resultSets"][0]["rowSet"]:
                    list.append({"game_id":game[1],"season":season,"game_type":type,"team_id":game[0],"game_date":game[2],"matchup":game[3],"WL":game[4]})
    games = pd.DataFrame(list).drop_duplicates(ignore_index=True)
    temp = []
    for game in games["game_id"].unique():
        subset = games.loc[games["game_id"]==game,].reset_index()
        if (len(subset.index) != 2):
            continue
        if ("vs." in subset.at[0,"matchup"]):
            temp.append({"game_id":subset.at[0,"game_id"],"season":subset.at[0,"season"],"game_type":subset.at[0,"game_type"],"game_date":subset.at[0,"game_date"],
                        "h_team_id":subset.at[0,"team_id"],"a_team_id":subset.at[1,"team_id"],"matchup":subset.at[0,"matchup"]})
        else:
            temp.append({"game_id":subset.at[0,"game_id"],"season":subset.at[0,"season"],"game_type":subset.at[0,"game_type"],"game_date":subset.at[0,"game_date"],
                        "h_team_id":subset.at[1,"team_id"],"a_team_id":subset.at[0,"team_id"],"matchup":subset.at[1,"matchup"]})
    final = pd.DataFrame(temp)
    final["game_date"] = pd.to_datetime(final["game_date"])
    final = final.sort_values(by="game_date")
    final.to_csv(db_path + "games.csv", index = False)
    

#traditional, advanced, misc, scoring, usage, four_factors
def boxscore_table(type):
    games = pd.read_csv(db_path + "games.csv")
    #valid box score years
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(1996, 2023)]
    string_s = seasons[0]
    for season in seasons:
        string_s += "|" + season
    game_ids = list(games[games["season"].str.contains(string_s[8:])]["game_id"])

    if (exists(db_path+type+"_boxscores_players.csv")):
        team_df = pd.read_csv(db_path+type+"_boxscores_teams.csv")
        player_df = pd.read_csv(db_path+type+"_boxscores_players.csv")
        stored_games = team_df["game_id"].unique()
        for game in stored_games:
            game_ids.remove(game)

    player_table = []
    team_table = []
    for game in tqdm(game_ids):
        #time.sleep(0.15)
        try:
            if (type == "traditional"):
                a = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreTraditional"
            elif (type == "advanced"):
                a = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreAdvanced"
            elif (type == "misc"):
                a = boxscoremiscv3.BoxScoreMiscV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreMisc"
            elif (type == "scoring"):
                a = boxscorescoringv3.BoxScoreScoringV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreScoring"
            elif (type == "usage"):
                a = boxscoreusagev3.BoxScoreUsageV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreUsage"
            elif (type == "four_factors"):
                a = boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreFourFactors"
        except AttributeError:
            #print ("Error: 00" + str(game))
            continue
        except (requests.exceptions.ReadTimeout, KeyboardInterrupt):
            if (len(team_table) == 0):
                return (-1)
            df = pd.DataFrame(team_table)
            if (exists(db_path+type+"_boxscores_teams.csv")):
                df = pd.concat([team_df,df])
            df.to_csv(db_path+type+"_boxscores_teams.csv",index=False)

            df = pd.DataFrame(player_table)
            if (exists(db_path+type+"_boxscores_players.csv")):
                df = pd.concat([player_df,df])
            df.to_csv(db_path+type+"_boxscores_players.csv",index=False)
            return (-1)
        for side in ["homeTeam","awayTeam"]:
            temp_t = {"game_id":a[bs_containter]['gameId'],"team_id":a[bs_containter][side]["teamId"]}
            for key in a[bs_containter][side]['statistics']:
                temp_t[key] = a[bs_containter][side]['statistics'][key]
            team_table.append(temp_t)
            for player in a[bs_containter][side]["players"]:
                temp_p = {"game_id":a[bs_containter]['gameId'],"player_id":player['personId'],"team_id":a[bs_containter][side]["teamId"]}
                for key in player['statistics']:
                    temp_p[key] = player['statistics'][key]
                player_table.append(temp_p)

    
    df = pd.DataFrame(team_table)
    if (exists(db_path+type+"_boxscores_teams.csv")):
                df = pd.concat([team_df,df])
    df.to_csv(db_path+type+"_boxscores_teams.csv",index=False)

    df = pd.DataFrame(player_table)
    if (exists(db_path+type+"_boxscores_players.csv")):
        df = pd.concat([player_df,df])
    df.to_csv(db_path+type+"_boxscores_players.csv",index=False)


def play_by_play():
    games = pd.read_csv(db_path + "games.csv")
    #valid years
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(1996, 2023)]
    string_s = seasons[0]
    for season in seasons:
        string_s += "|" + season
    game_ids = list(games[games["season"].str.contains(string_s[8:])]["game_id"])

    if (exists(db_path+"play_by_play.csv")):
        df_old = pd.read_csv(db_path+"play_by_play.csv")
        stored_games = df_old["game_id"].unique()
        for game in stored_games:
            game_ids.remove(game)

    table = []
    for game in tqdm(game_ids):
        try:
            a = playbyplayv3.PlayByPlayV3(game_id="00"+str(game)).get_dict()
        except (requests.exceptions.ReadTimeout, KeyboardInterrupt):
            if (len(table) == 0):
                return (-1)
            df = pd.DataFrame(table)
            if (exists(db_path+"play_by_play.csv")):
                df = pd.concat([df_old,df])
            df.to_csv(db_path+"play_by_play.csv",index=False)

            return (-1)
        except:
            #print ("Error: 00" + str(game))
            continue
        for action in a['game']['actions']:
            temp_t = {'game_id':a['game']['gameId']}
            for key in action:
                temp_t[key] = action[key]
            table.append(temp_t)


    
    df = pd.DataFrame(table)
    if (exists(db_path+"play_by_play.csv")):
        df = pd.concat([df_old,df])
    df.to_csv(db_path+"play_by_play.csv",index=False)
    return (1)

#for logging issues with lineup processing
log_error = False
#For excluding games who have serious errors
bad_gids = []
#function used in lineups_on_court:
#takes the name and team of the player being subbed on, as well as the dfs for play by play for each team (home and away) and traditional game boxscore 
def substitution_player_id(replace_name, replace_team, h_pbp, a_pbp, game_bs):
    global log_error
    global bad_gids
    h_team = game_bs['team_id'].unique()[0]

    #replace_name = replace_name.lower()

    #Hard Coding for when all else fails... NBA doesn't know how to handle names from different cultures in their data
    hard_code = {"Ming":2397,"Nene":2403,"N'Diaye":1823,"Seung-Jin":2775,"N'Dong":101238,"McRay":200840,
                 "Jianlian":201146,"Yue":201180,"Pendergraph":201965,"Ewing Jr.":201607,"Kanter":202683,
                 "Mark Morris":202693,"Marc Morris":202694,"Jones III":203103,"Ennis III":203516,"Robinson III":203922,
                 "O'Bryant III":203948,"Drew II":203580,"Baldwin IV":1627735,"McClellan":1627815,"Jones, Jr.":1627884,
                 "Zhou":1627753,"Mason III":1628412,"Dowtin":1630288,"Nembhard Jr.":1630612,"Baldwin Jr.":1631116}
    if (replace_name in hard_code):
        return (hard_code[replace_name])


    if ('.' in replace_name.split(" ")[0]):
        if (replace_team == h_team):
            if (len(h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()) != 0):
                replace_id = h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
            else:
                #For when the replace name does not fit the format of X. Lastname - for example Er. Johnson when PlayerNameI is E. Johnson
                #Or when there is an Mik. Smith and Mic. Smith, 
                lname = replace_name.split(" ")[1]
                for index, row in h_pbp.loc[h_pbp['playerName']==lname, ].iterrows():
                    if (pd.isnull(row['description'])):
                        continue
                    if (replace_name in row['description'] and row['description'].count(lname) == 1):
                        return (row['personId'])
                potentials = []

                #Below finds the total seconds played for players without a single contribution, so we can find their ID from the boxscore
                subbed_on_rows = h_pbp[h_pbp["description"].str.contains('SUB: '+replace_name+' ', na=False)]
                sec_played = 0
                for index, row in subbed_on_rows.iterrows():
                    sec_played += float(row['clock'].split('PT')[1].split('M')[0]) * 60 + float(row['clock'].split('M')[1].split('S')[0])

                #Below finds the player from the boxscore who matches our mystery player
                for index, row in game_bs.loc[game_bs['team_id'] == replace_team].iterrows():
                    if (row['seconds'] >= sec_played - 1 and row['seconds'] <= sec_played + 1):
                        all_0 = True
                        for col in game_bs.columns:
                            #assists don't make id show up in play by play
                            if (col not in ['game_id','player_id','team_id','minutes','seconds','plusMinusPoints','assists']):
                                if (row[col] > 0):
                                    all_0 = False
                        if (all_0):
                            potentials.append(row['player_id'])
                if (len(potentials) == 1):
                    replace_id = potentials[0]
                else:
                    replace_id = np.nan
                    print ("ERROR: "+ str(len(potentials)) +" potential subs found", h_pbp['game_id'].unique()[0], replace_team, replace_name)
                    log_error = True
            if (len(h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                print ("ERROR: Duplicate Names", replace_name, h_pbp['game_id'].unique()[0])
                log_error = True
                bad_gids.append(h_pbp['game_id'].unique()[0])
        else:
            if (len(a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()) != 0):
                replace_id = a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
            else:
                lname = replace_name.split(" ")[1]
                for index, row in a_pbp.loc[a_pbp['playerName']==lname, ].iterrows():
                    if (pd.isnull(row['description'])):
                        continue
                    if (replace_name in row['description'] and row['description'].count(lname) == 1):
                        return (row['personId'])
                potentials = []

                #Below finds the total seconds played for players without a single contribution, so we can find their ID from the boxscore
                subbed_on_rows = a_pbp[a_pbp["description"].str.contains('SUB: '+replace_name+' ', na=False)]
                sec_played = 0
                for index, row in subbed_on_rows.iterrows():
                    sec_played += float(row['clock'].split('PT')[1].split('M')[0]) * 60 + float(row['clock'].split('M')[1].split('S')[0])

                #Below finds the player from the boxscore who matches our mystery player
                for index, row in game_bs.loc[game_bs['team_id'] == replace_team].iterrows():
                    if (row['seconds'] >= sec_played - 1 and row['seconds'] <= sec_played + 1):
                        all_0 = True
                        for col in game_bs.columns:
                            #assists don't make id show up in play by play
                            if (col not in ['game_id','player_id','team_id','minutes','seconds','plusMinusPoints','assists']):
                                if (row[col] > 0):
                                    all_0 = False
                        if (all_0):
                            potentials.append(row['player_id'])
                if (len(potentials) == 1):
                    replace_id = potentials[0]
                else:
                    replace_id = np.nan
                    print ("ERROR: "+ str(len(potentials)) +" potential subs found", h_pbp['game_id'].unique()[0], replace_team, replace_name)
                    log_error = True
            if (len(a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                print ("ERROR: Duplicate Names", replace_name, h_pbp['game_id'].unique()[0])
                log_error = True
                bad_gids.append(h_pbp['game_id'].unique()[0])
    else:
        if (replace_team == h_team):
            if (len(h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()) != 0):
                replace_id = h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()[0]
            else:
                potentials = []

                #Below finds the total seconds played for players without a single contribution, so we can find their ID from the boxscore
                subbed_on_rows = h_pbp[h_pbp["description"].str.contains('SUB: '+replace_name+' ', na=False)]
                sec_played = 0
                for index, row in subbed_on_rows.iterrows():
                    sec_played += float(row['clock'].split('PT')[1].split('M')[0]) * 60 + float(row['clock'].split('M')[1].split('S')[0])

                #Below finds the player from the boxscore who matches our mystery player
                for index, row in game_bs.loc[game_bs['team_id'] == replace_team].iterrows():
                    if (row['seconds'] >= sec_played - 1 and row['seconds'] <= sec_played + 1):
                        all_0 = True
                        for col in game_bs.columns:
                            #assists don't make id show up in play by play
                            if (col not in ['game_id','player_id','team_id','minutes','seconds','plusMinusPoints','assists']):
                                if (row[col] > 0):
                                    all_0 = False
                        if (all_0):
                            potentials.append(row['player_id'])
                if (len(potentials) == 1):
                    replace_id = potentials[0]
                else:
                    #For the case of Otto Porter Jr. who appears as Porter and not Porter Jr.
                    if (replace_name == "Porter" and "O. Porter Jr." in list(h_pbp['playerNameI'])):
                        return (203490)
                    replace_id = np.nan
                    print ("ERROR: "+ str(len(potentials)) +" potential subs found", h_pbp['game_id'].unique()[0], replace_team, replace_name)
                    log_error = True
            if (len(h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                print ("ERROR: Duplicate Names", replace_name, h_pbp['game_id'].unique()[0])
                log_error = True
                bad_gids.append(h_pbp['game_id'].unique()[0])
        else:
            if (len(a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()) != 0):
                replace_id = a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()[0]
            else:
                potentials = []

                #Below finds the total seconds played for players without a single contribution, so we can find their ID from the boxscore
                subbed_on_rows = a_pbp[a_pbp["description"].str.contains('SUB: '+replace_name+' ', na=False)]
                sec_played = 0
                for index, row in subbed_on_rows.iterrows():
                    sec_played += float(row['clock'].split('PT')[1].split('M')[0]) * 60 + float(row['clock'].split('M')[1].split('S')[0])

                #Below finds the player from the boxscore who matches our mystery player
                for index, row in game_bs.loc[game_bs['team_id'] == replace_team].iterrows():
                    if (row['seconds'] >= sec_played - 1 and row['seconds'] <= sec_played + 1):
                        all_0 = True
                        for col in game_bs.columns:
                            #assists don't make id show up in play by play
                            if (col not in ['game_id','player_id','team_id','minutes','seconds','plusMinusPoints','assists']):
                                if (row[col] > 0):
                                    all_0 = False
                        if (all_0):
                            potentials.append(row['player_id'])
                if (len(potentials) == 1):
                    replace_id = potentials[0]
                else:
                    #For the case of Otto Porter Jr. who appears as Porter and not Porter Jr.
                    if (replace_name == "Porter" and "O. Porter Jr." in list(a_pbp['playerNameI'])):
                        return (203490)
                    replace_id = np.nan
                    print ("ERROR: "+ str(len(potentials)) +" potential subs found", h_pbp['game_id'].unique()[0], replace_team, replace_name)
                    log_error = True
            if (len(a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                print ("ERROR: Duplicate Names", replace_name, h_pbp['game_id'].unique()[0])
                log_error = True
                bad_gids.append(h_pbp['game_id'].unique()[0])
            
    
    return (replace_id)
#function used in lineups_on_court
#takes the unaccounted_game_time dict and the dict about to be pushed to the table with the info of time played for the current lineup
def game_time_accounting(unaccounted_game_time, dict):
    global log_error
    for key in dict:
        if (pd.isnull(dict[key])):
            continue
        if ('h_player_' in key):
            try:
                if (dict[key] > 0): unaccounted_game_time['home'][dict[key]] -= dict['end'] - dict['start']
            except:
                print ('ERROR: Player Id not found in boxscore.', dict['game_id'], dict['h_id'], dict[key])
                log_error = True
        elif ('a_player_' in key):
            try:
                if (dict[key] > 0): unaccounted_game_time['away'][dict[key]] -= dict['end'] - dict['start']
            except:
                print ('ERROR: Player Id not found in boxscore.', dict['game_id'], dict['a_id'], dict[key])
                log_error = True
#Uses play by play data to get data about every unique lineup during the game
#need to come up with a way to validate
#Note: When run for future seasons, must validate all errors in the new season
def lineups_on_court():
    pbp = pd.read_csv(db_path+'play_by_play.csv')
    
    #This is disgusting to have to do, but it's all I could come up with for these cases where the name has additional titles at the end but not the name in the description
    #Although the players start out being referred to without the title, the title gets added eventually, so you must also hard code their id to their full title name in the sub function
    pbp = pbp.replace("Jones III","Jones")
    pbp = pbp.replace("Ennis III","Ennis")
    pbp = pbp.replace("O'Bryant III","O'Bryant")
    pbp = pbp.replace("Drew II","Drew")
    pbp = pbp.replace("Baldwin IV","Baldwin")
    pbp = pbp.replace("Mason III","Mason")

    box_score = pd.read_csv(db_path+'traditional_boxscores_players.csv')
    #pbp = pbp.iloc[0:100000]
    #pbp.to_csv('C:/Users/jackj/OneDrive/Desktop/play_by_play.csv', index = False)
    #return
    games = pd.read_csv("./database/games.csv")
    games_to_exclude = list(games[games["game_type"].str.contains("Pre Season|Pre-Season")]['game_id'])

    game_ids = list(pbp["game_id"].unique())

    #players_df = pd.read_csv(db_path+'players.csv')
    #player_map = dict.fromkeys(players_df[''])

    table = []
    global log_error
    global bad_gids

    for gid in tqdm(game_ids):
        if (gid in games_to_exclude):
            continue
        log_error = False
        game_pbp = pbp.loc[pbp['game_id']==gid,].reset_index(drop=True)

        game_bs = box_score.loc[box_score['game_id']==gid,]
        pd.options.mode.chained_assignment = None
        try:
            game_bs['seconds'] = game_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
        except:
            game_bs['seconds'] = game_bs['minutes'].dropna() * 60

        sub_events = game_pbp.loc[game_pbp['actionType']=="Substitution", ].copy()

        sub_events['sub_id'] = sub_events.groupby(['clock','period']).ngroup()

        #2960032 is missing from boxscore, will just ignore it altogether
        try:
            h_team = game_bs['team_id'].unique()[0]
            a_team = game_bs['team_id'].unique()[1]
        except IndexError:
            continue

        #these lists store the periods where there was a missing player
        #if there is only one item in the list, we know the key that is missing must be x_player_5 across all slots of a certain period
        #if there are 2, for example, it could be x_player_4 and x_player_5 for the same period, or x_player_5 missing for two different entire periods
        #we won't code handling 2 or more for now
        missing_list_h = []
        missing_list_a = []

        #This is used when a player plays the entirety of a period, usually OT, without doing anything
        #We have no name or anything to go off of, so we try to account for all of the game time given by the box score
        #Start each player by their seconds played in the boxscore, subtract time played as we push dict to the table, and record the index and key pairs of the nan in the dict in the table
        #At the end, if we have a record of this happening, we look through unaccounted_game_time for a player with an entire period worth of seconds unaccounted for and put him in the dicts in the table
        unaccounted_game_time = {'home':{},'away':{}}
        for index, row in game_bs.loc[game_bs['team_id'] == h_team].iterrows():
            if (not pd.isnull(row['seconds'])):
                unaccounted_game_time['home'][row['player_id']] = row['seconds']
        for index, row in game_bs.loc[game_bs['team_id'] == a_team].iterrows():
            if (not pd.isnull(row['seconds'])):
                unaccounted_game_time['away'][row['player_id']] = row['seconds']

        h_pbp = game_pbp.loc[game_pbp['teamId']==h_team,]
        a_pbp = game_pbp.loc[game_pbp['teamId']==a_team,]

        dict = {'game_id':gid,'h_id':h_team,'a_id':a_team,'start':0,'end':-1,'h_player_1':'','h_player_2':'',
                'h_player_3':'','h_player_4':'','h_player_5':'','a_player_1':'','a_player_2':'','a_player_3':'','a_player_4':'','a_player_5':''}
    

        new_lineup = False

        for index in range(len(game_pbp.index)):
            if ((game_pbp.at[index, 'actionType'] == 'period' and game_pbp.at[index, 'subType'] == 'start') or game_pbp.at[index, 'actionId'] == 1):
                if (game_pbp.at[index, 'period'] != 1):
                    if (game_pbp.at[index, 'period'] > 4):
                        dict['end'] = 2880 + (game_pbp.at[index, 'period']-5) * 300
                        if (not sub_in_period):
                            for i in range(len(h_players_found), 5):
                                dict['h_player_'+str(i+1)] = -(i+1)
                                missing_list_h.append(game_pbp.at[index,'period'])
                            for i in range(len(a_players_found), 5):
                                dict['a_player_'+str(i+1)] = -(i+1)
                                missing_list_a.append(game_pbp.at[index,'period'])
                        table.append(dict.copy())
                        game_time_accounting(unaccounted_game_time, dict)
                        dict = {'game_id':gid,'h_id':h_team,'a_id':a_team,'start':2880 + (game_pbp.at[index, 'period']-5) * 300,'end':-1} 
                    else:
                        dict['end'] = (game_pbp.at[index, 'period']-1) * 720
                        table.append(dict.copy())
                        game_time_accounting(unaccounted_game_time, dict)
                        dict = {'game_id':gid,'h_id':h_team,'a_id':a_team,'start':(game_pbp.at[index, 'period']-1) * 720,'end':-1}
                h_players_found = []
                a_players_found = []
                #new_lineup is true when 1,2,3,4,OTs periods begin and we do not know who started the quarter on the court; switched to False when we know
                new_lineup = True
                sub_in_period = False
            #bench players can get technical fouls
            if (new_lineup and not (game_pbp.at[index, 'actionType'] == 'Foul' and (game_pbp.at[index, 'subType'] == 'Technical' or game_pbp.at[index, 'subType'] == 'Double Technical'))):
                if (game_pbp.at[index,'teamId'] == h_team and game_pbp.at[index,'personId'] not in h_players_found and game_pbp.at[index,'personId'] in unaccounted_game_time['home']):
                    h_players_found.append(game_pbp.at[index,'personId'])
                    dict['h_player_'+str(len(h_players_found))] = game_pbp.at[index,'personId']
                    if (len(h_players_found) > 5):
                        print ("ERROR: 6 Players Thought to be on court", gid)
                        bad_gids.append(gid)
                        log_error = True
                        
                if (game_pbp.at[index,'teamId'] == a_team and game_pbp.at[index,'personId'] not in a_players_found and game_pbp.at[index,'personId'] in unaccounted_game_time['away']):
                    a_players_found.append(game_pbp.at[index,'personId'])
                    dict['a_player_'+str(len(a_players_found))] = game_pbp.at[index,'personId']
                    if (len(a_players_found) > 5):
                        print ("ERROR: 6 Players Thought to be on court", gid)
                        bad_gids.append(gid)
                        log_error = True
                if (len(a_players_found) == 5 and len(h_players_found) == 5):
                    new_lineup = False

            if (game_pbp.at[index, 'actionType'] == 'Substitution' and not pd.isnull(game_pbp.at[index,'description'])):
                sub_in_period = True

                if (game_pbp.at[index,'period'] <= 4):
                    sec_elapsed_game = 720*(game_pbp.at[index,'period']) - float(game_pbp.at[index,'clock'].split("PT")[1].split("M")[0]) * 60 - float(game_pbp.at[index,'clock'].split("M")[1].split("S")[0])
                else:
                    sec_elapsed_game = 2880 + 300*(game_pbp.at[index,'period']-4) - float(game_pbp.at[index,'clock'].split("PT")[1].split("M")[0]) * 60 - float(game_pbp.at[index,'clock'].split("M")[1].split("S")[0])

                #This handles when we reach the first substitution of the quarter and we still do not know who started the quarter
                #We iterate through every remaining play for the rest of the period to find the starters - we can never find them if they play all quarter without any contribution
                if (new_lineup):
                    h_subbed_on = []
                    a_subbed_on = []

                    # #Sometimes there are duplicate end period events
                    # end_period_indices = list(np.where(game_pbp['subType'] == 'end'))[0]
                    # end_period_dict = {}
                    # for ind in end_period_indices:
                    #     end_period_dict[game_pbp.at[ind,'period']] = ind
                    # end_period_index = end_period_dict[game_pbp.at[index,'period']]

                    cur_period = game_pbp.at[index,'period']
                    end_period_index = index
                    max_index = len(game_pbp.index)
                    while (end_period_index < max_index and game_pbp.at[end_period_index,'period'] == cur_period):
                        end_period_index += 1

                    for j in range(index, end_period_index):
                        if (not (game_pbp.at[j, 'actionType'] == 'Foul' and (game_pbp.at[j, 'subType'] == 'Technical' or game_pbp.at[j, 'subType'] == 'Double Technical'))):
                            if (game_pbp.at[j,'teamId'] == h_team and game_pbp.at[j,'personId'] not in h_players_found and game_pbp.at[j,'personId'] not in h_subbed_on and game_pbp.at[j,'personId'] in unaccounted_game_time['home']):
                                h_players_found.append(game_pbp.at[j,'personId'])
                                dict['h_player_'+str(len(h_players_found))] = game_pbp.at[j,'personId']
                                if (len(h_players_found) > 5):
                                    print ("ERROR: 6 Players Thought to be on court", gid)
                                    bad_gids.append(gid)
                                    log_error = True
                            if (game_pbp.at[j,'teamId'] == a_team and game_pbp.at[j,'personId'] not in a_players_found and game_pbp.at[j,'personId'] not in a_subbed_on and game_pbp.at[j,'personId'] in unaccounted_game_time['away']):
                                a_players_found.append(game_pbp.at[j,'personId'])
                                dict['a_player_'+str(len(a_players_found))] = game_pbp.at[j,'personId']
                                if (len(a_players_found) > 5):
                                    print ("ERROR: 6 Players Thought to be on court", gid)
                                    bad_gids.append(gid)
                                    log_error = True
                            if (len(a_players_found) == 5 and len(h_players_found) == 5):
                                new_lineup = False
                                break
                        
                        if (game_pbp.at[j, 'actionType'] == 'Substitution' and not pd.isnull(game_pbp.at[j,'description'])):
                            replace_name = str(game_pbp.at[j,'description']).split("SUB: ")[1].split(" FOR")[0]
                            replace_team = game_pbp.at[j,'teamId']
                            if (replace_team == h_team):
                                h_subbed_on.append(substitution_player_id(replace_name,replace_team,h_pbp,a_pbp,game_bs))
                            else:
                                a_subbed_on.append(substitution_player_id(replace_name,replace_team,h_pbp,a_pbp,game_bs))
                    for i in range(len(h_players_found), 5):
                        dict['h_player_'+str(i+1)] = -(i+1)
                        missing_list_h.append(game_pbp.at[index,'period'])
                    for i in range(len(a_players_found), 5):
                        dict['a_player_'+str(i+1)] = -(i+1)
                        missing_list_a.append(game_pbp.at[index,'period'])
                    new_lineup = False
                
                #when multiple subs happen consecutively
                if (not (game_pbp.at[index-1, 'actionType'] == 'Substitution' and game_pbp.at[index-1, 'clock'] == game_pbp.at[index, 'clock'])):
                    dict['end'] = sec_elapsed_game
                    table.append(dict.copy())
                    game_time_accounting(unaccounted_game_time, dict)

                    dict['start'] = sec_elapsed_game
                    dict['end'] = -1

                #finds what position in the dict the player being subbed out is
                key_found = False
                for k in dict:
                    if ('_player_' in k):
                        if (dict[k] == game_pbp.at[index,'personId']):
                            key = k
                            key_found = True
                            break
                if (not key_found):
                    #If there is only one unknown player in the dict, then we know it must be them being subbed out
                    #If not, this fails and the rest of the period's data is useless
                    only_one_unknown = False
                    for k in dict:
                        if ('_player_' in k):
                            if (dict[k] < 0 and not only_one_unknown):
                                key = k
                                only_one_unknown = True
                            elif (dict[k] < 0 and only_one_unknown):
                                only_one_unknown = False
                                break
                    
                    #Often this happens because a player was ejected and they do not specify the sub like they usually do
                    #It is very rare that this happens - currently just throwing away the games - example 20300785
                    if (not only_one_unknown):
                        print ("BIG BIG ERROR (Unknown key - data ruined):", gid, game_pbp.at[index,'personId'], sec_elapsed_game)
                        bad_gids.append(gid)
                        log_error = True
                
                #print (str(game_pbp.at[index,'description']))
                replace_name = str(game_pbp.at[index,'description']).split("SUB: ")[1].split(" FOR")[0]
                replace_team = game_pbp.at[index,'teamId']
                
                new_id = substitution_player_id(replace_name,replace_team,h_pbp,a_pbp,game_bs)

                #We mark missing players with a negative id instead of nan so that each position still has a unique value
                if (pd.isnull(new_id)):
                    dict[key] = -int(key.split("player_")[1])
                else:
                    dict[key] = substitution_player_id(replace_name,replace_team,h_pbp,a_pbp,game_bs)
            
        end_period = list(game_pbp['period'])[-1]
        dict["end"] = 2880 + (end_period-4)*300
        table.append(dict.copy())
        game_time_accounting(unaccounted_game_time, dict)
        
        #Going to only code for the case of max one missing per team
        if (len(missing_list_h) == 1):
            for key in unaccounted_game_time['home']:
                if (unaccounted_game_time['home'][key] > 298):
                    for t_dict in table[-100:]:
                        try:
                            if (t_dict['h_player_5'] < 0):
                                t_dict['h_player_5'] = key
                        except:
                            log_error = True
                            print ("ERROR: h_player_5 missing???", gid)
                            bad_gids.append(gid)
        if (len(missing_list_a) == 1):
            for key in unaccounted_game_time['away']:
                if (unaccounted_game_time['away'][key] > 298):
                    for t_dict in table[-100:]:
                        try:
                            if (t_dict['a_player_5'] < 0):
                                t_dict['a_player_5'] = key
                        except:
                            log_error = True
                            print ("ERROR: a_player_5 missing???", gid)
                            bad_gids.append(gid)
            
        if (log_error):
            df = pd.DataFrame(table[-100:])
            df.to_csv('./error_logs/'+str(gid)+'_lineups.csv', index=False)
            game_pbp.to_csv('./error_logs/'+str(gid)+'_pbp.csv', index=False)


    df = pd.DataFrame(table)
    drop_rows = df['game_id'].isin(bad_gids)
    df = df[~drop_rows]
    print ("Games dropped:", bad_gids)
    df.to_csv(db_path+'unique_lineups.csv',index=False)


#helper apply function
#If clock is 0 min 0 sec, we make sure that we translate that to 0.1 sec so that it gets put in the correct bin; otherwise, things at the end of period are included in the next period starters stats
#With subs, there is no problem with the bins because the only time things happen at the same clock time as a sub is when someone is shooting free throws
    #this should be attributed to the players on before the substitution who committed the foul, so we want it included in the previous bin
    #However we must do something somewhere so that we exclude the free throws from the group subbed on
def clock_sec(row):
    if (row['period'] <= 4):
        time = 720*(row['period']) - float(row['clock'].split("PT")[1].split("M")[0]) * 60 - float(row['clock'].split("M")[1].split("S")[0])
    else:
        time = 2880 + 300*(row['period']-4) - float(row['clock'].split("PT")[1].split("M")[0]) * 60 - float(row['clock'].split("M")[1].split("S")[0])
    
    if (float(row['clock'].split("PT")[1].split("M")[0]) == 0 and float(row['clock'].split("M")[1].split("S")[0] == 0)):
        return (time - 0.001)
    else:
        return (time)
#The following are for getting the stats of the players for the particular lineup
def get_lineup_key(row, id, location):
    missing_value_keys = []
    for side in ['h','a']:
        for i in range(1,6):
            if (row[side+'_player_'+str(i)] == id):
                return (side+'_player_'+str(i))
            elif (row[side+'_player_'+str(i)] < 0):
                missing_value_keys.append(side+'_player_'+str(i))
    

    if (len(missing_value_keys) == 1):
        return (side+'_player_'+str(i))
    print ("ERROR: Player ID Not Found in Lineup:", id, row['game_id'], row['start'], row['end'])
    if (location == 'h'):
        return ('h_unknown')
    elif (location == 'v'):
        return ('a_unknown')
#checks if the stat should be tabulated for the lineup before or after the sub
def should_skip(row, index, actionType, pbp, start=-1, end=-1):
    #stats that happen at the same time as subs are counted twice otherwise
    #this checks for whether the event occured before the sub or after the sub and whether it should be ignored for the cur_lineup
    ignore = False
    if (row['clock_sec'] == start):
        sub_happened = False
        for i, r in pbp.loc[pbp['clock_sec']==start,].iterrows():
            if (r['actionType'] == 'Substitution'):
                sub_happened = True
            if (not sub_happened and row['actionType'] == actionType and index == i):
                ignore = True
                break
    elif (row['clock_sec'] == end):
        sub_happened = False
        for i, r in pbp.loc[pbp['clock_sec']==end,].iterrows():
            if (r['actionType'] == 'Substitution'):
                sub_happened = True
            if (sub_happened and row['actionType'] == actionType and index == i):
                ignore = True
                break
    return (ignore)
#Searches through the current range of the pbp for missing shots, made shots, etc 
def tabulate_traditional_stats(cur_pbp, cur_lu, pbp, stats, start, end):
    if (stats == {}):
        stats = {'possessions':0,'h_team_dreb':0,'h_team_oreb':0,'a_team_dreb':0,'a_team_oreb':0}
        cats = ['fgm','fga','3pm','3pa','ftm','fta','oreb','dreb','pts']
        for side in ['h','a']:
            for i in range(1,6):
                for cat in cats:
                    stats[side+'_player_' + str(i) + '_' + cat] = 0
        for x in ['h_unknown','a_unknown']:
            for cat in cats:
                stats[x+'_'+cat] = 0
    
    made_shot = cur_pbp.loc[cur_pbp['actionType'] == 'Made Shot',]
    for index, row in made_shot.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160)):
            return (stats)
        
        if (row['clock_sec'] == start):
            if (should_skip(row, index, 'Made Shot', pbp, start=start)):
                continue
        elif (row['clock_sec'] == end):
            if (should_skip(row, index, 'Made Shot', pbp, end=end)):
                continue
        key = get_lineup_key(cur_lu, row['personId'], row['location'])
        stats[key+'_fgm'] += 1
        stats[key+'_fga'] += 1
        if ("3PT" in row['description']):
            stats[key+'_3pm'] += 1
            stats[key+'_3pa'] += 1
            stats[key+'_pts'] += 3
        else:
            stats[key+'_pts'] += 2

        #We ignore and-one makes
        stats['possessions'] += 1
    
    missed_shot = cur_pbp.loc[cur_pbp['actionType'] == 'Missed Shot',]
    for index, row in missed_shot.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160 or end == 2880)):
            return (stats)
        
        if (row['clock_sec'] == start):
            if (should_skip(row, index, 'Missed Shot', pbp, start=start)):
                continue
        elif (row['clock_sec'] == end):
            if (should_skip(row, index, 'Missed Shot', pbp, end=end)):
                continue
        key = get_lineup_key(cur_lu, row['personId'], row['location'])
        stats[key+'_fga'] += 1
        if ("3PT" in row['description']):
            stats[key+'_3pa'] += 1
    
    free_throws = cur_pbp.loc[cur_pbp['actionType'] == 'Free Throw',]
    for index, row in free_throws.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160)):
            return (stats)
        
        #Free throws that happen around subs are counted twice otherwise
        #Credit should go to the team who fouled to start the free throw shooting, ie. the last lineup
        #There is a situation though where a technical is committed and the player is hurt and subbed off and the entering player takes the free throw
        #If it is a technical or flagrant at the start or end, then we will check if it comes before or after the first sub
        if (('Technical' in row['description'] or 'Flagrant' in row['description']) and row['clock_sec'] == end):
            if (should_skip(row, index, 'Free Throw', pbp, end=end)):
                continue
        if (('Technical' in row['description'] or 'Flagrant' in row['description']) and row['clock_sec'] == start):
            if (should_skip(row, index, 'Free Throw', pbp, start=start)):
                continue
        if (row['clock_sec'] == start):
            continue
        key = get_lineup_key(cur_lu, row['personId'], row['location'])
        stats[key+"_fta"] += 1
        if ("MISS" not in row['description']):
            stats[key+"_ftm"] += 1
            stats[key+"_pts"] += 1
        
        if (not 'Technical' in row['description'] and not 'Flagrant' in row['description'] and not "Clear Path" in row['description']):
            #And-ones have their possession tabulated on the made shot
            if ('1 of 1' not in row['subType']):
                if ("MISS " not in row['description']):
                    stats['possessions'] += 1
    
    rebounds = cur_pbp.loc[cur_pbp['actionType'] == 'Rebound',]
    for index, row in rebounds.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160 or end == 2880)):
            return (stats)
        
        #Sometimes rebounds are given to missed shots at the ends of quarters but they are meaningless, if they even actually occur
        if (row['clock'] == 'PT00M00.00S'):
            continue
        #The rebounds after a missed free throw after a sub should go to the 2nd lineup
        #but, the possession should be given to the 1st lineup because we tabulate possessions AFTER they happen
        #this only applied to free throws
        skip_reb = False
        start_event = False
        if (row['clock_sec'] == start):
            if (should_skip(row, index, 'Rebound', pbp, start=start)):
                skip_reb = True
            start_event = True
        elif (row['clock_sec'] == end):
            if (should_skip(row, index, 'Rebound', pbp, end=end)):
                skip_reb = True  
        if (row['personId'] > 1000000000):
            if (row['location'] == 'h'):
                key = 'h_team'
            elif (row['location'] == 'v'):
                key = 'a_team'
            else:
                print ("ERROR: Team Rebound Unknown Side")
        else:
            key = get_lineup_key(cur_lu, row['personId'], row['location'])
        if (not pd.isnull(pbp.at[index-1,'description']) and "MISS " in pbp.at[index-1,'description']):
            #Sometimes rebounds are improperly given after missed first free throw out of two
            if ("Free Throw" in pbp.at[index-1,'description']):
                if (not 'Technical' in pbp.at[index-1,'description'] and not 'Flagrant' in pbp.at[index-1,'description'] and not "Clear Path" in pbp.at[index-1,'description']):
                    if ('1 of 1' not in pbp.at[index-1,'subType']):
                        if (pbp.at[index-1,'teamId'] == row['teamId']):
                            if (not skip_reb):
                                stats[key+'_oreb'] += 1
                        else:
                            if (not skip_reb):
                                stats[key+'_dreb'] += 1 
                            if (not start_event):
                                stats['possessions'] += 1
            else:
                if (pbp.at[index-1,'teamId'] == row['teamId']):
                    stats[key+'_oreb'] += 1
                else:
                    stats[key+'_dreb'] += 1
                    stats['possessions'] += 1
        elif (not pd.isnull(pbp.at[index-1,'description']) and not pd.isnull(pbp.at[index-2,'description']) and "BLK)" in pbp.at[index-1,'description'] and "MISS " in pbp.at[index-2,'description']):
            if (pbp.at[index-2,'teamId'] == row['teamId']):
                stats[key+'_oreb'] += 1
            else:
                stats[key+'_dreb'] += 1
                stats['possessions'] += 1
        elif (not pd.isnull(pbp.at[index-1,'description']) and "SUB: " in pbp.at[index-1,'description']):
            cur_index = index - 2
            while (not pd.isnull(pbp.at[cur_index,'description']) and "SUB: " in pbp.at[cur_index,'description']):
                cur_index -= 1
            if ("Free Throw" in pbp.at[cur_index,'description']):
                if (not 'Technical' in pbp.at[cur_index,'description'] and not 'Flagrant' in pbp.at[cur_index,'description'] and not "Clear Path" in pbp.at[cur_index,'description']):
                    if ('1 of 1' not in pbp.at[cur_index,'subType']):
                        if (pbp.at[cur_index,'teamId'] == row['teamId']):
                            if (not skip_reb):
                                stats[key+'_oreb'] += 1
                        else:
                            if (not skip_reb):
                                stats[key+'_dreb'] += 1
                            if (not start_event):
                                stats['possessions'] += 1
            else:
                if (pbp.at[cur_index,'teamId'] == row['teamId']):
                    stats[key+'_oreb'] += 1
                else:
                    stats[key+'_dreb'] += 1
                    stats['possessions'] += 1
        elif (not pd.isnull(pbp.at[index-1,'description']) and "REBOUND" in pbp.at[index-1,'description']):
            continue
        elif (not pd.isnull(pbp.at[index-1,'description']) and "Jump Ball" in pbp.at[index-1,'description']):
            if (not pd.isnull(pbp.at[index-2,'description']) and 'MISS ' in pbp.at[index-2,'description']):
                if (pbp.at[index-2,'teamId'] == row['teamId']):
                    stats[key+'_oreb'] += 1
                else:
                    stats[key+'_dreb'] += 1
                    stats['possessions'] += 1
            elif (not pd.isnull(pbp.at[index-2,'description']) and not pd.isnull(pbp.at[index-3,'description']) and 'BLK)' in pbp.at[index-2,'description'] and 'MISS ' in pbp.at[index-3,'description']):
                if (pbp.at[index-3,'teamId'] == row['teamId']):
                    stats[key+'_oreb'] += 1
                else:
                    stats[key+'_dreb'] += 1
                    stats['possessions'] += 1
        else:
            print ("ERROR: Rebound doesn't follow missed shot", row['game_id'], row['clock_sec'])
    
    turnovers = cur_pbp.loc[cur_pbp['actionType'] == 'Turnover',]
    for index, row in turnovers.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160 or end == 2880)):
            return (stats)
        
        if (row['clock_sec'] == start):
            if (should_skip(row, index, 'Turnover', pbp, start=start)):
                continue
        elif (row['clock_sec'] == end):
            if (should_skip(row, index, 'Turnover', pbp, end=end)):
                continue

        stats['possessions'] += 1
    
    goaltends = cur_pbp.loc[cur_pbp['subType'] == 'Defensive Goaltending',]
    for index, row in goaltends.iterrows():
        #Sometimes things happen at the absolute beginning of the quarter which would be considered the end of the last lineup, need to avoid this
        if (row['clock'] == 'PT12M00.00S' and (end == 720 or end == 1440 or end == 2160 or end == 2880)):
            return (stats)
        
        if (row['clock_sec'] == start):
            if (should_skip(row, index, 'Violation', pbp, start=start)):
                continue
        elif (row['clock_sec'] == end):
            if (should_skip(row, index, 'Violation', pbp, end=end)):
                continue

        stats['possessions'] += 1
    
    return (stats)
def boxscores_by_lineup():
    lineups = pd.read_csv(db_path+'unique_lineups.csv')
    play_by_play = pd.read_csv('C:/Users/jackj/OneDrive/Desktop/play_by_play.csv')

    play_by_play['clock_sec'] = play_by_play.apply(clock_sec, axis=1)

    game_ids = list(lineups['game_id'].unique())

    table = []
    for gid in tqdm(game_ids):
        pbp = play_by_play.loc[play_by_play['game_id']==gid,].reset_index(drop=True)
        lu = lineups.loc[lineups['game_id']==gid,].reset_index(drop=True)
        

        for index in range(len(lu.index)):
            stats = {}
            start = lu.at[index,"start"]
            end = lu.at[index,"end"]
            if (start != end):
                cur_pbp = pbp[pbp['clock_sec'].between(start, end)]
                
                stats = tabulate_traditional_stats(cur_pbp, lu.iloc[index], pbp, stats, start, end)
            
            table.append(stats)
    
    df = pd.DataFrame(table)
    lineups = pd.concat([lineups,df],axis=1)
    lineups.to_csv(db_path+'unique_lineup_stats.csv',index=False)


#Must have scraped nowgoal odds and have nowgoal_odds.csv in intermediates
def odds_table():
    odds = pd.read_csv("./intermediates/nowgoal_odds.csv")
    teams = pd.read_csv(db_path+"teams.csv")
    games = pd.read_csv(db_path+"games.csv")
    id_map = {}
    for i in range(len(teams.index)):
        id_map[teams["full_name"].to_list()[i]] = teams["id"].to_list()[i]
    odds["h_team"] = odds["h_team"].transform(lambda x: id_map[x])
    odds["a_team"] = odds["a_team"].transform(lambda x: id_map[x])
    odds = odds.rename(columns={"h_team":"h_team_id","a_team":"a_team_id"})
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds.merge(games, how='inner', left_on=["h_team_id","a_team_id","date"], right_on=["h_team_id","a_team_id","game_date"])
    odds = odds.drop(["nowgoal_id","season","game_type","game_date","matchup"],axis=1)
    odds.insert(0, "game_id", odds.pop("game_id"))
    odds = odds.sort_values("date")
    odds = odds.replace(["-"," - "], np.nan)
    odds.to_csv(db_path+"odds.csv",index=False)


#for uploading to git
def compress_db():
    for file in tqdm(os.listdir("./database")):
        a = pd.read_csv(db_path+file)
        a.to_csv("./compressed_database/"+file+".gz",index=False,compression="gzip")

def extract_db():
    for file in tqdm(os.listdir("./compressed_database")):
        a = pd.read_csv("./compressed_database/"+file, compression="gzip")
        a.to_csv("./database/"+file.split(".gz")[0],index=False)

boxscores_by_lineup()
