import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import pickle
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
import os

def poisson_approx(k, lambda_):
    return (NormalDist(lambda_, math.sqrt(lambda_)).cdf(k+0.5) - NormalDist(lambda_, math.sqrt(lambda_)).cdf(k-0.5))

def poisson(k, lambda_):
    return (math.exp(-lambda_) * lambda_**k / math.factorial(k))

def kelly(p, ret_odds, div):
    return ((p - (1-p)/ret_odds) / div)

#recursive thru 3OTs
def OT_dist(pace, h_rtg, a_rtg, recursive_depth=1, cur_points=0, prior_p = 1, ot_dist=[]):
    if (recursive_depth < 1):
        return
    
    if (recursive_depth == 1):
        ot_dist = np.zeros((75,75))


    for i in range(25):
        for j in range(25):
            p = poisson(i, h_rtg*pace/100/9.6) * poisson(j, a_rtg*pace/100/9.6)
            if (i != j):
                ot_dist[i+cur_points][j+cur_points] += p * prior_p
            else:
                if (i == 0):
                    continue
                OT_dist(pace, h_rtg, a_rtg, recursive_depth-1, i+cur_points, prior_p*p, ot_dist)

    
    return ot_dist

#independent marginal poissons (normal approx)
#OT pace and rtg assumed to be same proportionally to reg time
def score_joint_pmf(pace, h_rtg, a_rtg):
    dist = np.zeros((275,275))
    ot_dist = OT_dist(pace, h_rtg, a_rtg)

    for i in range(200):
        for j in range(200):
            p = poisson_approx(i, h_rtg*pace/100) * poisson_approx(j, a_rtg*pace/100)
            if (i != j or p == 0):
                dist[i][j] += p
            else:
                dist[i][j] = 0
                dist[i:i+75,j:j+75] += ot_dist * p
                
    
    return (dist)

def p_spread(dist, open, close):
    p_open_home = 0
    p_close_home = 0
    p_open_away = 0
    p_close_away = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if (i - j > open):
                p_open_home += dist[i][j]
            elif (i - j < open):
                p_open_away += dist[i][j]
            
            if (i - j > close):
                p_close_home += dist[i][j]
            elif (i - j < close):
                p_close_away += dist[i][j]

    p_open_home = p_open_home / (p_open_home + p_open_away)

    p_close_home = p_close_home / (p_close_home + p_close_away)

    return ((p_open_home, p_close_home))


def p_total(dist, open, close):
    p_open_over = 0
    p_close_over = 0
    p_open_under = 0
    p_close_under = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if (i + j > open):
                p_open_over += dist[i][j]
            elif (i + j < open):
                p_open_under += dist[i][j]
            
            if (i + j > close):
                p_close_over += dist[i][j]
            elif (i + j < close):
                p_close_under += dist[i][j]

    p_open_over = p_open_over / (p_open_over + p_open_under)

    p_close_over = p_close_over / (p_close_over + p_close_under)

    return ((p_open_over, p_close_over))

def result(row, spread_total, open_close):
    if (spread_total == 'spread'):
        if (row[open_close+'_spread_side'] == 'H'):
            if (row['actual_spread'] > row['bet365_'+open_close+'_spread']):
                return ('W')
            elif (row['actual_spread'] < row['bet365_'+open_close+'_spread']):
                return ('L')
            else:
                return ('P')
        else:
            if (row['actual_spread'] < row['bet365_'+open_close+'_spread']):
                return ('W')
            elif (row['actual_spread'] > row['bet365_'+open_close+'_spread']):
                return ('L')
            else:
                return ('P')
    
    if (spread_total == 'total'):
        if (row[open_close+'_total_side'] == 'O'):
            if (row['actual_total'] > row['bet365_'+open_close+'_total']):
                return ('W')
            elif (row['actual_total'] < row['bet365_'+open_close+'_total']):
                return ('L')
            else:
                return ('P')
        else:
            if (row['actual_total'] < row['bet365_'+open_close+'_total']):
                return ('W')
            elif (row['actual_total'] > row['bet365_'+open_close+'_total']):
                return ('L')
            else:
                return ('P')

def edge(row, spread_total, open_close, pred_actual):
    if (row[open_close+'_'+spread_total+'_side'] == 'H' or row[open_close+'_'+spread_total+'_side'] == 'O'):
        book_p = 1 / (row['bet365_h_'+open_close+'_'+spread_total+'_odds'] + 1)
    else:
        book_p = 1 / (row['bet365_a_'+open_close+'_'+spread_total+'_odds'] + 1)

    if (pred_actual == 'pred'):
        if (row[open_close+'_'+spread_total+'_side'] == 'H' or row[open_close+'_'+spread_total+'_side'] == 'O'):
            return (max(row[open_close+'_'+spread_total+'_prob'] - book_p, 0))
        else:
            return (max(row[open_close+'_'+spread_total+'_prob'] - book_p, 0))
    else:
        if (row[open_close+'_'+spread_total+'_result'] == 'W'):
            return (1 - book_p)
        elif (row[open_close+'_'+spread_total+'_result'] == 'L'):
            return (0 - book_p)
        elif (row[open_close+'_'+spread_total+'_result'] == 'P'):
            return (0)

def bankroll_growth_graph(pred, model, start_br=10000, kelly_div = 10):
    if (not os.path.exists('./figures/betting/' + model)):
        os.mkdir('./figures/betting/' + model)
    
    for x in ['open','close']:
        for y in ['spread','total']:
            bankroll = [start_br]
            bet_num = [0]
            total_bet = 0
            net_win = 0
            weight_lm = 0
            for index in range(len(pred.index)):
                if (pred.at[index,'pred_'+x+'_'+y+'_edge'] > 0):
                    bet_size = kelly(pred.at[index,x+'_'+y+'_prob'],
                                    pred.at[index,x+'_'+y+'_odds'],
                                    kelly_div)
                    if (pred.at[index, x+'_'+y+'_result'] == 'W'):
                        bankroll.append(bankroll[-1] + bankroll[-1]*bet_size*pred.at[index,x+'_'+y+'_odds'])
                        bet_num.append(len(bet_num))
                        total_bet += bet_size
                        net_win += bet_size*pred.at[index,x+'_'+y+'_odds']
                    elif (pred.at[index, x+'_'+y+'_result'] == 'L'):
                        bankroll.append(bankroll[-1] - bankroll[-1]*bet_size)
                        bet_num.append(len(bet_num))
                        total_bet += bet_size
                        net_win -= bet_size
                    elif (pred.at[index, x+'_'+y+'_result'] == 'P'):
                        bankroll.append(bankroll[-1])
                        bet_num.append(len(bet_num))
                    if (x=='open'):
                        weight_lm += bet_size * pred.at[index,y+'_line_movement']

            plt.plot(bet_num, bankroll)
            plt.xlabel("Bet Number")
            plt.ylabel("Bankroll $")
            plt.title("Growth of $" + str(start_br) + " on " + y.capitalize() + " Bets at " + x.capitalize() + " From 2007-08 through 2018-19", fontsize=17)
            plt.figtext(0.75,0.85,"Return per Bet: "+str((net_win/total_bet).round(3)), fontsize=15)
            if (x=='open'):
                plt.figtext(0.13,0.85,"Avg Weighted Line Movement: "+str((weight_lm/total_bet).round(3)), fontsize=15)
            plt.gcf().set_size_inches(16,9)
            plt.savefig('./figures/betting/' + model + '/' + x + '_' + y + '_bankroll.png', dpi=100)
            plt.clf()

def calibration_curve_edge(pred, model):
    for x in ['open','close']:
        for y in ['spread','total']:
            pred_edge = list(pred['pred_'+x+'_'+y+'_edge'])
            act_edge = list(pred['actual_'+x+'_'+y+'_edge'])
            pair = {}
            for i in range(len(pred_edge)):
                if (not pd.isnull(pred_edge[i]) and not pd.isnull(act_edge[i])):
                    pair[pred_edge[i]] = act_edge[i]
            sorted_pair = dict(sorted(pair.items()))
            n = len(sorted_pair)
            bins = [(0,n*0.01),(n*0.01,n*0.025),(n*0.025,n*0.05),(n*0.05,n*0.1),(n*0.1,n*0.20),(n*0.20,n*0.35),(n*0.35,n*0.5),
                    (n*0.50,n*0.65),(n*0.65,n*0.80),(n*0.80,n*0.9),(n*0.9,n*0.95),(n*0.95,n*0.975),(n*0.975,n*0.99),(n*0.99,n)]
            sorted_pred = list(sorted_pair.keys())
            sorted_actual = list(sorted_pair.values())
            binned_pred = []
            binned_actual = []
            binned_n = []
            for bin in bins:
                binned_pred.append(np.mean(sorted_pred[math.trunc(bin[0]):math.trunc(bin[1])]))
                binned_actual.append(np.mean(sorted_actual[math.trunc(bin[0]):math.trunc(bin[1])]))
                binned_n.append(math.trunc(bin[1]) - math.trunc(bin[0]))

            fig, ax = plt.subplots()
            ax.scatter(binned_pred, binned_actual, s=binned_n, alpha = 0.5)
            plt.xlabel("Predicted Edge")
            plt.ylabel("Actual Edge")
            plt.title("Calibration Curve for " + y.capitalize() + " Bets at " + x.capitalize() + " From 2007-08 through 2018-19", fontsize=17)
            plt.ylim(-0.1,0.25)
            plt.figtext(0.81,0.86,"Correct Calibration", fontsize=10, c="g")
            plt.figtext(0.81,0.84,"Minimum Breakeven", fontsize=10, c="b")
            plt.figtext(0.81,0.82,"Random Guessing", fontsize=10, c="r")
            ax.axline((0, -0.0263), slope=0, c="r")
            ax.axline((0, 0), slope=0, c="b")
            ax.axline((0, 0), slope=1, c="g")
            plt.gcf().set_size_inches(16,9)
            plt.savefig('./figures/betting/' + model + '/' + x + '_' + y + '_calibration_curve.png', dpi=100)
            plt.clf()



#takes the name of the model; file with the name must exist in /predictions/betting/pre_bet/
#2007-08 thru 2018-19
def backtest_bet(model):
    odds = pd.read_csv('./database/odds.csv')
    pred = pd.read_csv('./predictions/betting/pre_bet/' + model + '.csv')

    drop_cols = []
    for col in odds.columns:
        if ('game_id' not in col and 'score' not in col and 'bet365' not in col):
            drop_cols.append(col)
    odds = odds.drop(columns=drop_cols)

    seasons = ""
    for yr in range(2007, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    pred = pred[pred["season"].str.contains(seasons[:-1])]
    
    pred = pred.merge(odds,how='inner',on='game_id').reset_index(drop=True)

    table = []

    for index in tqdm(range(len(pred.index))):
        pred_dist = score_joint_pmf(pred.at[index, 'pred_pace'],pred.at[index, 'h_rtg_pred'],pred.at[index, 'a_rtg_pred'])

        cur_row = {'pred_h_points':pred.at[index, 'pred_pace']*pred.at[index, 'h_rtg_pred']/100,'pred_a_points':pred.at[index, 'pred_pace']*pred.at[index, 'a_rtg_pred']/100}

        if (not pd.isnull(pred.at[index, 'bet365_open_spread']) and not pd.isnull(pred.at[index, 'bet365_close_spread'])):
            p_open_home, p_close_home = p_spread(pred_dist, pred.at[index,'bet365_open_spread'], pred.at[index,'bet365_close_spread'])
            if (p_open_home > 0.5):
                cur_row['open_spread_side'] = 'H'
                cur_row['open_spread_prob'] = p_open_home
                cur_row['open_spread_odds'] = pred.at[index,'bet365_h_open_spread_odds']
                cur_row['spread_line_movement'] = pred.at[index,'bet365_close_spread'] - pred.at[index,'bet365_open_spread']
            else:
                cur_row['open_spread_side'] = 'A'
                cur_row['open_spread_prob'] = 1 - p_open_home
                cur_row['open_spread_odds'] = pred.at[index,'bet365_a_open_spread_odds']
                cur_row['spread_line_movement'] = pred.at[index,'bet365_open_spread'] - pred.at[index,'bet365_close_spread']
            if (p_close_home > 0.5):
                cur_row['close_spread_side'] = 'H'
                cur_row['close_spread_prob'] = p_close_home
                cur_row['close_spread_odds'] = pred.at[index,'bet365_h_close_spread_odds']
            else:
                cur_row['close_spread_side'] = 'A'
                cur_row['close_spread_prob'] = 1 - p_close_home
                cur_row['close_spread_odds'] = pred.at[index,'bet365_a_close_spread_odds']
        
        if (not pd.isnull(pred.at[index, 'bet365_open_total']) and not pd.isnull(pred.at[index, 'bet365_close_total'])):
            p_open_over, p_close_over = p_total(pred_dist, pred.at[index,'bet365_open_total'], pred.at[index,'bet365_close_total'])
            if (p_open_over > 0.5):
                cur_row['open_total_side'] = 'O'
                cur_row['open_total_prob'] = p_open_over
                cur_row['open_total_odds'] = pred.at[index,'bet365_h_open_total_odds']
                cur_row['total_line_movement'] = pred.at[index,'bet365_close_total'] - pred.at[index,'bet365_open_total']
            else:
                cur_row['open_total_side'] = 'U'
                cur_row['open_total_prob'] = 1 - p_open_over
                cur_row['open_total_odds'] = pred.at[index,'bet365_a_open_total_odds']
                cur_row['total_line_movement'] = pred.at[index,'bet365_open_total'] - pred.at[index,'bet365_close_total']
            if (p_close_over > 0.5):
                cur_row['close_total_side'] = 'O'
                cur_row['close_total_prob'] = p_close_over
                cur_row['close_total_odds'] = pred.at[index,'bet365_h_close_total_odds']
            else:
                cur_row['close_total_side'] = 'U'
                cur_row['close_total_prob'] = 1 - p_close_over
                cur_row['close_total_odds'] = pred.at[index,'bet365_a_close_total_odds']
        
        table.append(cur_row)
    
    bets = pd.DataFrame(table)
    pred = pd.concat([pred, bets], axis=1)
    pred.to_csv("./predictions/betting/post_bet/"+model+".csv", index=False)
    

def backtest_eval(model):
    pred = pd.read_csv('predictions/betting/post_bet/'+model+'.csv')
    pred['actual_spread'] = pred['h_score'] - pred['a_score']
    pred['actual_total'] = pred['h_score'] + pred['a_score']
    for x in ['open','close']:
        for y in ['spread','total']:
            pred[x+'_'+y+'_result'] = pred.apply(result, args=(y,x),axis=1)
    for x in ['open','close']:
        for y in ['spread','total']:
            for z in ['pred','actual']:
                pred[z+'_'+x+'_'+y+'_edge'] = pred.apply(edge, args=(y,x,z),axis=1)
    
    bankroll_growth_graph(pred,model)
    calibration_curve_edge(pred,model)

       
#backtest_bet('base_player_eff_bhm')
#backtest_bet('player_eff_bhm_home_app_twice')
#backtest_bet('player_eff_bhm_b2b')
backtest_eval('base_player_eff_bhm')
backtest_eval('player_eff_bhm_home_app_twice')
backtest_eval('player_eff_bhm_b2b')