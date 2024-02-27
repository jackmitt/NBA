import pandas as pd

#Searches for significant differences in my calculated pace and the actual recorded team pace
#currently only 0.5% of games have a pace diff of >= 3
# 0.02% have a pace diff of >= 5
def lineup_pace(diff_filter,log_pbp=False):
    lu = pd.read_csv('./database/unique_lineup_stats.csv')
    team_bs = pd.read_csv('./database/advanced_boxscores_teams.csv')
    if (log_pbp):
        pbp = pd.read_csv('./database/play_by_play.csv')

    ids = list(lu['game_id'].unique())

    for gid in ids:
        cur_lu = lu.loc[lu['game_id'] == gid,]
        lu_pace = cur_lu['possessions'].sum() * 2880 / list(cur_lu['end'])[-1]
        try:
            team_pace = list(team_bs.loc[team_bs['game_id']==gid,]['pace'])[0]
        except:
            continue
        if (abs(team_pace - lu_pace) >= diff_filter):
            print (gid, team_pace, lu_pace)
            if (log_pbp):
                pbp.loc[pbp['game_id']==gid,].to_csv('./error_logs/'+str(gid)+'.csv', index = False)

lineup_pace(5, True)