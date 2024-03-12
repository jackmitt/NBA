import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import pickle
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
import os

def poisson_approx(k, lambda_, var_factor):
    return (NormalDist(lambda_, math.sqrt(lambda_*var_factor)).cdf(k+0.5) - NormalDist(lambda_, math.sqrt(lambda_*var_factor)).cdf(k-0.5))

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
def score_joint_pmf(pace, h_rtg, a_rtg, var_factor):
    dist = np.zeros((275,275))
    ot_dist = OT_dist(pace, h_rtg, a_rtg)

    for i in range(200):
        for j in range(200):
            p = poisson_approx(i, h_rtg*pace/100, var_factor) * poisson_approx(j, a_rtg*pace/100, var_factor)
            if (i != j or p == 0):
                dist[i][j] += p
            else:
                dist[i][j] = 0
                dist[i:i+75,j:j+75] += ot_dist * p
                
    
    return (dist)

#Adjusts the poisson pmfs with weights based on the distribution of margin of victory which is weird bc of intentional fouling and such
#OT pace and rtg assumed to be same proportionally to reg time
def score_joint_pmf_bandaid(pace, h_rtg, a_rtg, var_factor):
    dist = np.zeros((275,275))
    ot_dist = OT_dist(pace, h_rtg, a_rtg)
    weights = [0.12113788, 0.17177079, 0.16850415, 0.16496529, 0.18674289, 0.186879]
    total_p = 0
    total_w_p = 0

    for i in range(200):
        for j in range(200):
            p = poisson_approx(i, h_rtg*pace/100, var_factor) * poisson_approx(j, a_rtg*pace/100, var_factor)
            if (i != j or p == 0):
                dist[i][j] += p
            else:
                dist[i][j] = 0
                dist[i:i+75,j:j+75] += ot_dist * p
                
    for i in range(275):
        for j in range(275):
            if (abs(i-j) <= 6 and i != j):
                total_p += dist[i][j]
                total_w_p += dist[i][j]*weights[abs(i - j)-1]
                dist[i][j] = dist[i][j]*weights[abs(i - j)-1]
    for i in range(275):
        for j in range(275):
            if (abs(i-j) <= 6 and i != j):
                dist[i][j] = dist[i][j] * total_p / total_w_p
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

def bankroll_growth_graph(pred, model, playoff_reg, start_br=10000, kelly_div = 10):
    if (not os.path.exists('./figures/betting/' + model)):
        os.mkdir('./figures/betting/' + model)
    
    if (playoff_reg == 'reg'):
        pred = pred[pred["game_type"].str.contains('Regular Season')].reset_index(drop=True)
    else:
        pred = pred[pred["game_type"].str.contains('Playoffs')].reset_index(drop=True)
    
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
            plt.title("Growth of $" + str(start_br) + " on " + y.capitalize() + " Bets at " + x.capitalize() + " From 2014-15 through 2018-19", fontsize=17)
            plt.figtext(0.75,0.85,"Return per Bet: "+str((net_win/total_bet).round(3)), fontsize=15)
            if (x=='open'):
                plt.figtext(0.13,0.85,"Avg Weighted Line Movement: "+str((weight_lm/total_bet).round(3)), fontsize=15)
            plt.gcf().set_size_inches(16,9)
            plt.savefig('./figures/betting/' + model + '/' + playoff_reg + '_' + x + '_' + y + '_bankroll.png', dpi=100)
            plt.clf()

def calibration_curve_edge(pred, model, playoff_reg):
    if (playoff_reg == 'reg'):
        pred = pred[pred["game_type"].str.contains('Regular Season')].reset_index(drop=True)
    else:
        pred = pred[pred["game_type"].str.contains('Playoffs')].reset_index(drop=True)

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
            plt.title("Calibration Curve for " + y.capitalize() + " Bets at " + x.capitalize() + " From 2014-15 through 2018-19", fontsize=17)
            plt.ylim(-0.1,0.25)
            plt.figtext(0.81,0.86,"Correct Calibration", fontsize=10, c="g")
            plt.figtext(0.81,0.84,"Minimum Breakeven", fontsize=10, c="b")
            plt.figtext(0.81,0.82,"Random Guessing", fontsize=10, c="r")
            ax.axline((0, -0.0263), slope=0, c="r")
            ax.axline((0, 0), slope=0, c="b")
            ax.axline((0, 0), slope=1, c="g")
            plt.gcf().set_size_inches(16,9)
            plt.savefig('./figures/betting/' + model + '/' + playoff_reg + '_' + x + '_' + y + '_calibration_curve.png', dpi=100)
            plt.clf()

def print_dashboard(pred,player_added_missing=False):
    pred = pred.dropna()
    pred['game_date'] = pd.to_datetime(pred['game_date'])
    o_s_b = pred.loc[pred['pred_open_spread_edge'] > 0.05]
    c_s_b = pred.loc[pred['pred_close_spread_edge'] > 0.05]
    o_t_b = pred.loc[pred['pred_open_total_edge'] > 0.05]
    c_t_b = pred.loc[pred['pred_close_total_edge'] > 0.05]

    h_o = pred.loc[pred['open_spread_side'] == 'H',]
    a_o = pred.loc[pred['open_spread_side'] == 'A',]
    h_c = pred.loc[pred['close_spread_side'] == 'H',]
    a_c = pred.loc[pred['close_spread_side'] == 'A',]
    o_o = pred.loc[pred['open_total_side'] == 'O',]
    u_o = pred.loc[pred['open_total_side'] == 'U',]
    o_c = pred.loc[pred['close_total_side'] == 'O',]
    u_c = pred.loc[pred['close_total_side'] == 'U',]
    print ('------------------ General Results (n = ' + str(len(pred.index)) + ') ------------------')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')

    print ('All Spreads          1                  ' + 
           str((pred['open_spread_result'].value_counts()['W'] / (pred['open_spread_result'].value_counts()['W'] + pred['open_spread_result'].value_counts()['L'])).round(3)) + '               1                  ' +
           str((pred['close_spread_result'].value_counts()['W'] / (pred['close_spread_result'].value_counts()['W'] + pred['close_spread_result'].value_counts()['L'])).round(3)))
    print ('Home Side            ' + str((pred['open_spread_side'].value_counts()['H'] / len(pred.index)).round(3)) + '              ' +
           str((h_o['open_spread_result'].value_counts()['W'] / (h_o['open_spread_result'].value_counts()['W'] + h_o['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str((pred['close_spread_side'].value_counts()['H'] / len(pred.index)).round(3)) + '               ' +
           str((h_c['close_spread_result'].value_counts()['W'] / (h_c['close_spread_result'].value_counts()['W'] + h_c['close_spread_result'].value_counts()['L'])).round(3)))
    print ('Away Side            ' + str((pred['open_spread_side'].value_counts()['A'] / len(pred.index)).round(3)) + '              ' +
           str((a_o['open_spread_result'].value_counts()['W'] / (a_o['open_spread_result'].value_counts()['W'] + a_o['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str((pred['close_spread_side'].value_counts()['A'] / len(pred.index)).round(3)) + '               ' +
           str((a_c['close_spread_result'].value_counts()['W'] / (a_c['close_spread_result'].value_counts()['W'] + a_c['close_spread_result'].value_counts()['L'])).round(3)))
    
    print ('------------------------------------')
    
    print ('All Totals           1                  ' + 
           str((pred['open_total_result'].value_counts()['W'] / (pred['open_total_result'].value_counts()['W'] + pred['open_total_result'].value_counts()['L'])).round(3)) + '                1                  ' +
           str((pred['close_total_result'].value_counts()['W'] / (pred['close_total_result'].value_counts()['W'] + pred['close_total_result'].value_counts()['L'])).round(3)))
    print ('Overs                ' + str((pred['open_total_side'].value_counts()['O'] / len(pred.index)).round(3)) + '               ' +
           str((o_o['open_total_result'].value_counts()['W'] / (o_o['open_total_result'].value_counts()['W'] + o_o['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str((pred['close_total_side'].value_counts()['O'] / len(pred.index)).round(3)) + '              ' +
           str((o_c['close_total_result'].value_counts()['W'] / (o_c['close_total_result'].value_counts()['W'] + o_c['close_total_result'].value_counts()['L'])).round(3)))
    print ('Unders               ' + str((pred['open_total_side'].value_counts()['U'] / len(pred.index)).round(3)) + '               ' +
           str((u_o['open_total_result'].value_counts()['W'] / (u_o['open_total_result'].value_counts()['W'] + u_o['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str((pred['close_total_side'].value_counts()['U'] / len(pred.index)).round(3)) + '              ' +
           str((u_c['close_total_result'].value_counts()['W'] / (u_c['close_total_result'].value_counts()['W'] + u_c['close_total_result'].value_counts()['L'])).round(3)))

    o_s_1 = pred.loc[pred['pred_open_spread_edge'] <= 0.05]
    o_s_2 = pred.loc[(pred['pred_open_spread_edge'] > 0.05) & (pred['pred_open_spread_edge'] <= 0.10)]
    o_s_3 = pred.loc[(pred['pred_open_spread_edge'] > 0.10) & (pred['pred_open_spread_edge'] <= 0.15)]
    o_s_4 = pred.loc[(pred['pred_open_spread_edge'] > 0.15) & (pred['pred_open_spread_edge'] <= 0.20)]
    o_s_5 = pred.loc[pred['pred_open_spread_edge'] > 0.2]
    c_s_1 = pred.loc[pred['pred_close_spread_edge'] <= 0.05]
    c_s_2 = pred.loc[(pred['pred_close_spread_edge'] > 0.05) & (pred['pred_close_spread_edge'] <= 0.10)]
    c_s_3 = pred.loc[(pred['pred_close_spread_edge'] > 0.10) & (pred['pred_close_spread_edge'] <= 0.15)]
    c_s_4 = pred.loc[(pred['pred_close_spread_edge'] > 0.15) & (pred['pred_close_spread_edge'] <= 0.20)]
    c_s_5 = pred.loc[pred['pred_close_spread_edge'] > 0.2]
    print ('------------------ Spread Results by Perceived Edge -------------')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('<= 0.05              ' + str(round(len(o_s_1.index)/len(pred.index),3)) + '               ' +
           str((o_s_1['open_spread_result'].value_counts()['W'] / (o_s_1['open_spread_result'].value_counts()['W'] + o_s_1['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_1.index)/len(pred.index),3)) + '               ' +
           str((c_s_1['close_spread_result'].value_counts()['W'] / (c_s_1['close_spread_result'].value_counts()['W'] + c_s_1['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 0.05 & <= 0.10     ' + str(round(len(o_s_2.index)/len(pred.index),3)) + '              ' +
           str((o_s_2['open_spread_result'].value_counts()['W'] / (o_s_2['open_spread_result'].value_counts()['W'] + o_s_2['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_2.index)/len(pred.index),3)) + '               ' +
           str((c_s_2['close_spread_result'].value_counts()['W'] / (c_s_2['close_spread_result'].value_counts()['W'] + c_s_2['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 0.10 & <= 0.15     ' + str(round(len(o_s_3.index)/len(pred.index),3)) + '              ' +
           str((o_s_3['open_spread_result'].value_counts()['W'] / (o_s_3['open_spread_result'].value_counts()['W'] + o_s_3['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_3.index)/len(pred.index),3)) + '               ' +
           str((c_s_3['close_spread_result'].value_counts()['W'] / (c_s_3['close_spread_result'].value_counts()['W'] + c_s_3['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 0.15 & <= 0.20     ' + str(round(len(o_s_4.index)/len(pred.index),3)) + '              ' +
           str((o_s_4['open_spread_result'].value_counts()['W'] / (o_s_4['open_spread_result'].value_counts()['W'] + o_s_4['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_4.index)/len(pred.index),3)) + '               ' +
           str((c_s_4['close_spread_result'].value_counts()['W'] / (c_s_4['close_spread_result'].value_counts()['W'] + c_s_4['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 0.20               ' + str(round(len(o_s_5.index)/len(pred.index),3)) + '              ' +
           str((o_s_5['open_spread_result'].value_counts()['W'] / (o_s_5['open_spread_result'].value_counts()['W'] + o_s_5['open_spread_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_s_5.index)/len(pred.index),3)) + '               ' +
           str((c_s_5['close_spread_result'].value_counts()['W'] / (c_s_5['close_spread_result'].value_counts()['W'] + c_s_5['close_spread_result'].value_counts()['L'])).round(3)))
    
    o_t_1 = pred.loc[pred['pred_open_total_edge'] <= 0.05]
    o_t_2 = pred.loc[(pred['pred_open_total_edge'] > 0.05) & (pred['pred_open_total_edge'] <= 0.10)]
    o_t_3 = pred.loc[(pred['pred_open_total_edge'] > 0.10) & (pred['pred_open_total_edge'] <= 0.15)]
    o_t_4 = pred.loc[(pred['pred_open_total_edge'] > 0.15) & (pred['pred_open_total_edge'] <= 0.20)]
    o_t_5 = pred.loc[pred['pred_open_total_edge'] > 0.2]
    c_t_1 = pred.loc[pred['pred_close_total_edge'] <= 0.05]
    c_t_2 = pred.loc[(pred['pred_close_total_edge'] > 0.05) & (pred['pred_close_total_edge'] <= 0.10)]
    c_t_3 = pred.loc[(pred['pred_close_total_edge'] > 0.10) & (pred['pred_close_total_edge'] <= 0.15)]
    c_t_4 = pred.loc[(pred['pred_close_total_edge'] > 0.15) & (pred['pred_close_total_edge'] <= 0.20)]
    c_t_5 = pred.loc[pred['pred_close_total_edge'] > 0.2]
    print ('------------------ Total Results by Perceived Edge --------------')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('<= 0.05              ' + str(round(len(o_t_1.index)/len(pred.index),3)) + '              ' +
           str((o_t_1['open_total_result'].value_counts()['W'] / (o_t_1['open_total_result'].value_counts()['W'] + o_t_1['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_1.index)/len(pred.index),3)) + '               ' +
           str((c_t_1['close_total_result'].value_counts()['W'] / (c_t_1['close_total_result'].value_counts()['W'] + c_t_1['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 0.05 & <= 0.10     ' + str(round(len(o_t_2.index)/len(pred.index),3)) + '              ' +
           str((o_t_2['open_total_result'].value_counts()['W'] / (o_t_2['open_total_result'].value_counts()['W'] + o_t_2['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_2.index)/len(pred.index),3)) + '               ' +
           str((c_t_2['close_total_result'].value_counts()['W'] / (c_t_2['close_total_result'].value_counts()['W'] + c_t_2['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 0.10 & <= 0.15     ' + str(round(len(o_t_3.index)/len(pred.index),3)) + '              ' +
           str((o_t_3['open_total_result'].value_counts()['W'] / (o_t_3['open_total_result'].value_counts()['W'] + o_t_3['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_3.index)/len(pred.index),3)) + '               ' +
           str((c_t_3['close_total_result'].value_counts()['W'] / (c_t_3['close_total_result'].value_counts()['W'] + c_t_3['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 0.15 & <= 0.20     ' + str(round(len(o_t_4.index)/len(pred.index),3)) + '              ' +
           str((o_t_4['open_total_result'].value_counts()['W'] / (o_t_4['open_total_result'].value_counts()['W'] + o_t_4['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_4.index)/len(pred.index),3)) + '               ' +
           str((c_t_4['close_total_result'].value_counts()['W'] / (c_t_4['close_total_result'].value_counts()['W'] + c_t_4['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 0.20               ' + str(round(len(o_t_5.index)/len(pred.index),3)) + '              ' +
           str((o_t_5['open_total_result'].value_counts()['W'] / (o_t_5['open_total_result'].value_counts()['W'] + o_t_5['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_5.index)/len(pred.index),3)) + '               ' +
           str((c_t_5['close_total_result'].value_counts()['W'] / (c_t_5['close_total_result'].value_counts()['W'] + c_t_5['close_total_result'].value_counts()['L'])).round(3)))
    
    o_s_o = o_s_b.loc[o_s_b['game_date'].dt.month == 10]
    o_s_n = o_s_b.loc[o_s_b['game_date'].dt.month == 11]
    o_s_d = o_s_b.loc[o_s_b['game_date'].dt.month == 12]
    o_s_j = o_s_b.loc[o_s_b['game_date'].dt.month == 1]
    o_s_f = o_s_b.loc[o_s_b['game_date'].dt.month == 2]
    o_s_m = o_s_b.loc[o_s_b['game_date'].dt.month == 3]
    o_s_a = o_s_b.loc[o_s_b['game_date'].dt.month == 4]
    c_s_o = c_s_b.loc[c_s_b['game_date'].dt.month == 10]
    c_s_n = c_s_b.loc[c_s_b['game_date'].dt.month == 11]
    c_s_d = c_s_b.loc[c_s_b['game_date'].dt.month == 12]
    c_s_j = c_s_b.loc[c_s_b['game_date'].dt.month == 1]
    c_s_f = c_s_b.loc[c_s_b['game_date'].dt.month == 2]
    c_s_m = c_s_b.loc[c_s_b['game_date'].dt.month == 3]
    c_s_a = c_s_b.loc[c_s_b['game_date'].dt.month == 4]
    print ('------------------ Spread Results by Month (Edge > 0.5) ---------')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('October              ' + str(round(len(o_s_o.index)/len(pred.index),3)) + '               ' +
           str((o_s_o['open_spread_result'].value_counts()['W'] / (o_s_o['open_spread_result'].value_counts()['W'] + o_s_o['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_o.index)/len(pred.index),3)) + '               ' +
           str((c_s_o['close_spread_result'].value_counts()['W'] / (c_s_o['close_spread_result'].value_counts()['W'] + c_s_o['close_spread_result'].value_counts()['L'])).round(3)))
    print ('November             ' + str(round(len(o_s_n.index)/len(pred.index),3)) + '               ' +
           str((o_s_n['open_spread_result'].value_counts()['W'] / (o_s_n['open_spread_result'].value_counts()['W'] + o_s_n['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_n.index)/len(pred.index),3)) + '               ' +
           str((c_s_n['close_spread_result'].value_counts()['W'] / (c_s_n['close_spread_result'].value_counts()['W'] + c_s_n['close_spread_result'].value_counts()['L'])).round(3)))
    print ('December             ' + str(round(len(o_s_d.index)/len(pred.index),3)) + '              ' +
           str((o_s_d['open_spread_result'].value_counts()['W'] / (o_s_d['open_spread_result'].value_counts()['W'] + o_s_d['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_d.index)/len(pred.index),3)) + '               ' +
           str((c_s_d['close_spread_result'].value_counts()['W'] / (c_s_d['close_spread_result'].value_counts()['W'] + c_s_d['close_spread_result'].value_counts()['L'])).round(3)))
    print ('January              ' + str(round(len(o_s_j.index)/len(pred.index),3)) + '              ' +
           str((o_s_j['open_spread_result'].value_counts()['W'] / (o_s_j['open_spread_result'].value_counts()['W'] + o_s_j['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_j.index)/len(pred.index),3)) + '               ' +
           str((c_s_j['close_spread_result'].value_counts()['W'] / (c_s_j['close_spread_result'].value_counts()['W'] + c_s_j['close_spread_result'].value_counts()['L'])).round(3)))
    print ('February             ' + str(round(len(o_s_f.index)/len(pred.index),3)) + '              ' +
           str((o_s_f['open_spread_result'].value_counts()['W'] / (o_s_f['open_spread_result'].value_counts()['W'] + o_s_f['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_f.index)/len(pred.index),3)) + '               ' +
           str((c_s_f['close_spread_result'].value_counts()['W'] / (c_s_f['close_spread_result'].value_counts()['W'] + c_s_f['close_spread_result'].value_counts()['L'])).round(3)))
    print ('March                ' + str(round(len(o_s_m.index)/len(pred.index),3)) + '              ' +
           str((o_s_m['open_spread_result'].value_counts()['W'] / (o_s_m['open_spread_result'].value_counts()['W'] + o_s_m['open_spread_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_s_m.index)/len(pred.index),3)) + '               ' +
           str((c_s_m['close_spread_result'].value_counts()['W'] / (c_s_m['close_spread_result'].value_counts()['W'] + c_s_m['close_spread_result'].value_counts()['L'])).round(3)))
    print ('April                ' + str(round(len(o_s_a.index)/len(pred.index),3)) + '              ' +
           str((o_s_a['open_spread_result'].value_counts()['W'] / (o_s_a['open_spread_result'].value_counts()['W'] + o_s_a['open_spread_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_s_a.index)/len(pred.index),3)) + '               ' +
           str((c_s_a['close_spread_result'].value_counts()['W'] / (c_s_a['close_spread_result'].value_counts()['W'] + c_s_a['close_spread_result'].value_counts()['L'])).round(3)))
    
    o_t_o = o_t_b.loc[o_t_b['game_date'].dt.month == 10]
    o_t_n = o_t_b.loc[o_t_b['game_date'].dt.month == 11]
    o_t_d = o_t_b.loc[o_t_b['game_date'].dt.month == 12]
    o_t_j = o_t_b.loc[o_t_b['game_date'].dt.month == 1]
    o_t_f = o_t_b.loc[o_t_b['game_date'].dt.month == 2]
    o_t_m = o_t_b.loc[o_t_b['game_date'].dt.month == 3]
    o_t_a = o_t_b.loc[o_t_b['game_date'].dt.month == 4]
    c_t_o = c_t_b.loc[c_t_b['game_date'].dt.month == 10]
    c_t_n = c_t_b.loc[c_t_b['game_date'].dt.month == 11]
    c_t_d = c_t_b.loc[c_t_b['game_date'].dt.month == 12]
    c_t_j = c_t_b.loc[c_t_b['game_date'].dt.month == 1]
    c_t_f = c_t_b.loc[c_t_b['game_date'].dt.month == 2]
    c_t_m = c_t_b.loc[c_t_b['game_date'].dt.month == 3]
    c_t_a = c_t_b.loc[c_t_b['game_date'].dt.month == 4]
    print ('------------------ Total Results by Month (Edge > 0.5) ----------')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('October              ' + str(round(len(o_t_o.index)/len(pred.index),3)) + '               ' +
           str((o_t_o['open_total_result'].value_counts()['W'] / (o_t_o['open_total_result'].value_counts()['W'] + o_t_o['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_o.index)/len(pred.index),3)) + '               ' +
           str((c_t_o['close_total_result'].value_counts()['W'] / (c_t_o['close_total_result'].value_counts()['W'] + c_t_o['close_total_result'].value_counts()['L'])).round(3)))
    print ('November             ' + str(round(len(o_t_n.index)/len(pred.index),3)) + '               ' +
           str((o_t_n['open_total_result'].value_counts()['W'] / (o_t_n['open_total_result'].value_counts()['W'] + o_t_n['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_n.index)/len(pred.index),3)) + '               ' +
           str((c_t_n['close_total_result'].value_counts()['W'] / (c_t_n['close_total_result'].value_counts()['W'] + c_t_n['close_total_result'].value_counts()['L'])).round(3)))
    print ('December             ' + str(round(len(o_t_d.index)/len(pred.index),3)) + '              ' +
           str((o_t_d['open_total_result'].value_counts()['W'] / (o_t_d['open_total_result'].value_counts()['W'] + o_t_d['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_d.index)/len(pred.index),3)) + '               ' +
           str((c_t_d['close_total_result'].value_counts()['W'] / (c_t_d['close_total_result'].value_counts()['W'] + c_t_d['close_total_result'].value_counts()['L'])).round(3)))
    print ('January              ' + str(round(len(o_t_j.index)/len(pred.index),3)) + '              ' +
           str((o_t_j['open_total_result'].value_counts()['W'] / (o_t_j['open_total_result'].value_counts()['W'] + o_t_j['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_j.index)/len(pred.index),3)) + '               ' +
           str((c_t_j['close_total_result'].value_counts()['W'] / (c_t_j['close_total_result'].value_counts()['W'] + c_t_j['close_total_result'].value_counts()['L'])).round(3)))
    print ('February             ' + str(round(len(o_t_f.index)/len(pred.index),3)) + '              ' +
           str((o_t_f['open_total_result'].value_counts()['W'] / (o_t_f['open_total_result'].value_counts()['W'] + o_t_f['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_f.index)/len(pred.index),3)) + '               ' +
           str((c_t_f['close_total_result'].value_counts()['W'] / (c_t_f['close_total_result'].value_counts()['W'] + c_t_f['close_total_result'].value_counts()['L'])).round(3)))
    print ('March                ' + str(round(len(o_t_m.index)/len(pred.index),3)) + '              ' +
           str((o_t_m['open_total_result'].value_counts()['W'] / (o_t_m['open_total_result'].value_counts()['W'] + o_t_m['open_total_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_t_m.index)/len(pred.index),3)) + '               ' +
           str((c_t_m['close_total_result'].value_counts()['W'] / (c_t_m['close_total_result'].value_counts()['W'] + c_t_m['close_total_result'].value_counts()['L'])).round(3)))
    print ('April                ' + str(round(len(o_t_a.index)/len(pred.index),3)) + '              ' +
           str((o_t_a['open_total_result'].value_counts()['W'] / (o_t_a['open_total_result'].value_counts()['W'] + o_t_a['open_total_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_t_a.index)/len(pred.index),3)) + '               ' +
           str((c_t_a['close_total_result'].value_counts()['W'] / (c_t_a['close_total_result'].value_counts()['W'] + c_t_a['close_total_result'].value_counts()['L'])).round(3)))
    
    o_s_1 = o_s_b.loc[abs(o_s_b['bet365_open_spread']) <= 2]
    o_s_2 = o_s_b.loc[(abs(o_s_b['bet365_open_spread']) > 2) & (abs(o_s_b['bet365_open_spread']) <= 5)]
    o_s_3 = o_s_b.loc[(abs(o_s_b['bet365_open_spread']) > 5) & (abs(o_s_b['bet365_open_spread']) <= 10)]
    o_s_4 = o_s_b.loc[(abs(o_s_b['bet365_open_spread']) > 10) & (abs(o_s_b['bet365_open_spread']) <= 15)]
    o_s_5 = o_s_b.loc[(abs(o_s_b['bet365_open_spread']) > 15)]
    c_s_1 = c_s_b.loc[abs(c_s_b['bet365_close_spread']) <= 2]
    c_s_2 = c_s_b.loc[(abs(c_s_b['bet365_close_spread']) > 2) & (abs(c_s_b['bet365_close_spread']) <= 5)]
    c_s_3 = c_s_b.loc[(abs(c_s_b['bet365_close_spread']) > 5) & (abs(c_s_b['bet365_close_spread']) <= 10)]
    c_s_4 = c_s_b.loc[(abs(c_s_b['bet365_close_spread']) > 10) & (abs(c_s_b['bet365_close_spread']) <= 15)]
    c_s_5 = c_s_b.loc[(abs(c_s_b['bet365_close_spread']) > 15)]
    print ('------------------ Spread Results by Spread Magnitude (Edge > 0.5)')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('<= 2                 ' + str(round(len(o_s_1.index)/len(pred.index),3)) + '               ' +
           str((o_s_1['open_spread_result'].value_counts()['W'] / (o_s_1['open_spread_result'].value_counts()['W'] + o_s_1['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_1.index)/len(pred.index),3)) + '               ' +
           str((c_s_1['close_spread_result'].value_counts()['W'] / (c_s_1['close_spread_result'].value_counts()['W'] + c_s_1['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 2 & <= 5           ' + str(round(len(o_s_2.index)/len(pred.index),3)) + '              ' +
           str((o_s_2['open_spread_result'].value_counts()['W'] / (o_s_2['open_spread_result'].value_counts()['W'] + o_s_2['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_2.index)/len(pred.index),3)) + '               ' +
           str((c_s_2['close_spread_result'].value_counts()['W'] / (c_s_2['close_spread_result'].value_counts()['W'] + c_s_2['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 5 & <= 10          ' + str(round(len(o_s_3.index)/len(pred.index),3)) + '              ' +
           str((o_s_3['open_spread_result'].value_counts()['W'] / (o_s_3['open_spread_result'].value_counts()['W'] + o_s_3['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_3.index)/len(pred.index),3)) + '               ' +
           str((c_s_3['close_spread_result'].value_counts()['W'] / (c_s_3['close_spread_result'].value_counts()['W'] + c_s_3['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 10 & <= 15         ' + str(round(len(o_s_4.index)/len(pred.index),3)) + '              ' +
           str((o_s_4['open_spread_result'].value_counts()['W'] / (o_s_4['open_spread_result'].value_counts()['W'] + o_s_4['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_4.index)/len(pred.index),3)) + '               ' +
           str((c_s_4['close_spread_result'].value_counts()['W'] / (c_s_4['close_spread_result'].value_counts()['W'] + c_s_4['close_spread_result'].value_counts()['L'])).round(3)))
    print ('> 15                 ' + str(round(len(o_s_5.index)/len(pred.index),3)) + '              ' +
           str((o_s_5['open_spread_result'].value_counts()['W'] / (o_s_5['open_spread_result'].value_counts()['W'] + o_s_5['open_spread_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_s_5.index)/len(pred.index),3)) + '               ' +
           str((c_s_5['close_spread_result'].value_counts()['W'] / (c_s_5['close_spread_result'].value_counts()['W'] + c_s_5['close_spread_result'].value_counts()['L'])).round(3)))
    
    o_t_1 = o_t_b.loc[abs(o_t_b['bet365_open_total']) <= 195]
    o_t_2 = o_t_b.loc[(abs(o_t_b['bet365_open_total']) > 195) & (abs(o_t_b['bet365_open_total']) <= 205)]
    o_t_3 = o_t_b.loc[(abs(o_t_b['bet365_open_total']) > 205) & (abs(o_t_b['bet365_open_total']) <= 215)]
    o_t_4 = o_t_b.loc[(abs(o_t_b['bet365_open_total']) > 215) & (abs(o_t_b['bet365_open_total']) <= 225)]
    o_t_5 = o_t_b.loc[(abs(o_t_b['bet365_open_total']) > 225)]
    c_t_1 = c_t_b.loc[abs(c_t_b['bet365_close_total']) <= 195]
    c_t_2 = c_t_b.loc[(abs(c_t_b['bet365_close_total']) > 195) & (abs(c_t_b['bet365_close_total']) <= 205)]
    c_t_3 = c_t_b.loc[(abs(c_t_b['bet365_close_total']) > 205) & (abs(c_t_b['bet365_close_total']) <= 215)]
    c_t_4 = c_t_b.loc[(abs(c_t_b['bet365_close_total']) > 215) & (abs(c_t_b['bet365_close_total']) <= 225)]
    c_t_5 = c_t_b.loc[(abs(c_t_b['bet365_close_total']) > 225)]
    print ('------------------ Total Results by Total Magnitude (Edge > 0.5)')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('<= 195                 ' + str(round(len(o_t_1.index)/len(pred.index),3)) + '               ' +
           str((o_t_1['open_total_result'].value_counts()['W'] / (o_t_1['open_total_result'].value_counts()['W'] + o_t_1['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_1.index)/len(pred.index),3)) + '               ' +
           str((c_t_1['close_total_result'].value_counts()['W'] / (c_t_1['close_total_result'].value_counts()['W'] + c_t_1['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 195 & <= 205           ' + str(round(len(o_t_2.index)/len(pred.index),3)) + '              ' +
           str((o_t_2['open_total_result'].value_counts()['W'] / (o_t_2['open_total_result'].value_counts()['W'] + o_t_2['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_2.index)/len(pred.index),3)) + '               ' +
           str((c_t_2['close_total_result'].value_counts()['W'] / (c_t_2['close_total_result'].value_counts()['W'] + c_t_2['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 205 & <= 215          ' + str(round(len(o_t_3.index)/len(pred.index),3)) + '              ' +
           str((o_t_3['open_total_result'].value_counts()['W'] / (o_t_3['open_total_result'].value_counts()['W'] + o_t_3['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_3.index)/len(pred.index),3)) + '               ' +
           str((c_t_3['close_total_result'].value_counts()['W'] / (c_t_3['close_total_result'].value_counts()['W'] + c_t_3['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 215 & <= 225         ' + str(round(len(o_t_4.index)/len(pred.index),3)) + '              ' +
           str((o_t_4['open_total_result'].value_counts()['W'] / (o_t_4['open_total_result'].value_counts()['W'] + o_t_4['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_4.index)/len(pred.index),3)) + '               ' +
           str((c_t_4['close_total_result'].value_counts()['W'] / (c_t_4['close_total_result'].value_counts()['W'] + c_t_4['close_total_result'].value_counts()['L'])).round(3)))
    print ('> 225                 ' + str(round(len(o_t_5.index)/len(pred.index),3)) + '              ' +
           str((o_t_5['open_total_result'].value_counts()['W'] / (o_t_5['open_total_result'].value_counts()['W'] + o_t_5['open_total_result'].value_counts()['L'])).round(3)) + '                 ' +
           str(round(len(c_t_5.index)/len(pred.index),3)) + '               ' +
           str((c_t_5['close_total_result'].value_counts()['W'] / (c_t_5['close_total_result'].value_counts()['W'] + c_t_5['close_total_result'].value_counts()['L'])).round(3)))

    if (not player_added_missing):
        return 1
    
    o_s_1 = o_s_b.loc[(o_s_b['significant_player_added'] == 0) & (o_s_b['significant_player_removed'] == 0)]
    o_s_2 = o_s_b.loc[(o_s_b['significant_player_added'] == 1) & (o_s_b['significant_player_removed'] == 0)]
    o_s_3 = o_s_b.loc[(o_s_b['significant_player_added'] == 0) & (o_s_b['significant_player_removed'] == 1)]
    o_s_4 = o_s_b.loc[(o_s_b['significant_player_added'] == 1) & (o_s_b['significant_player_removed'] == 1)]
    c_s_1 = c_s_b.loc[(c_s_b['significant_player_added'] == 0) & (c_s_b['significant_player_removed'] == 0)]
    c_s_2 = c_s_b.loc[(c_s_b['significant_player_added'] == 1) & (c_s_b['significant_player_removed'] == 0)]
    c_s_3 = c_s_b.loc[(c_s_b['significant_player_added'] == 0) & (c_s_b['significant_player_removed'] == 1)]
    c_s_4 = c_s_b.loc[(c_s_b['significant_player_added'] == 1) & (c_s_b['significant_player_removed'] == 1)]
    print ('------------------ Spread Results by Lineup Changes (Edge > 0.5)')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('Mostly Same          ' + str(round(len(o_s_1.index)/len(pred.index),3)) + '               ' +
           str((o_s_1['open_spread_result'].value_counts()['W'] / (o_s_1['open_spread_result'].value_counts()['W'] + o_s_1['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_1.index)/len(pred.index),3)) + '               ' +
           str((c_s_1['close_spread_result'].value_counts()['W'] / (c_s_1['close_spread_result'].value_counts()['W'] + c_s_1['close_spread_result'].value_counts()['L'])).round(3)))
    print ('Significant Addition ' + str(round(len(o_s_2.index)/len(pred.index),3)) + '              ' +
           str((o_s_2['open_spread_result'].value_counts()['W'] / (o_s_2['open_spread_result'].value_counts()['W'] + o_s_2['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_2.index)/len(pred.index),3)) + '               ' +
           str((c_s_2['close_spread_result'].value_counts()['W'] / (c_s_2['close_spread_result'].value_counts()['W'] + c_s_2['close_spread_result'].value_counts()['L'])).round(3)))
    print ('Significant Removal  ' + str(round(len(o_s_3.index)/len(pred.index),3)) + '              ' +
           str((o_s_3['open_spread_result'].value_counts()['W'] / (o_s_3['open_spread_result'].value_counts()['W'] + o_s_3['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_3.index)/len(pred.index),3)) + '               ' +
           str((c_s_3['close_spread_result'].value_counts()['W'] / (c_s_3['close_spread_result'].value_counts()['W'] + c_s_3['close_spread_result'].value_counts()['L'])).round(3)))
    print ('Both Significances   ' + str(round(len(o_s_4.index)/len(pred.index),3)) + '              ' +
           str((o_s_4['open_spread_result'].value_counts()['W'] / (o_s_4['open_spread_result'].value_counts()['W'] + o_s_4['open_spread_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_s_4.index)/len(pred.index),3)) + '               ' +
           str((c_s_4['close_spread_result'].value_counts()['W'] / (c_s_4['close_spread_result'].value_counts()['W'] + c_s_4['close_spread_result'].value_counts()['L'])).round(3)))
    
    o_t_1 = o_t_b.loc[(o_t_b['significant_player_added'] == 0) & (o_t_b['significant_player_removed'] == 0)]
    o_t_2 = o_t_b.loc[(o_t_b['significant_player_added'] == 1) & (o_t_b['significant_player_removed'] == 0)]
    o_t_3 = o_t_b.loc[(o_t_b['significant_player_added'] == 0) & (o_t_b['significant_player_removed'] == 1)]
    o_t_4 = o_t_b.loc[(o_t_b['significant_player_added'] == 1) & (o_t_b['significant_player_removed'] == 1)]
    c_t_1 = c_t_b.loc[(c_t_b['significant_player_added'] == 0) & (c_t_b['significant_player_removed'] == 0)]
    c_t_2 = c_t_b.loc[(c_t_b['significant_player_added'] == 1) & (c_t_b['significant_player_removed'] == 0)]
    c_t_3 = c_t_b.loc[(c_t_b['significant_player_added'] == 0) & (c_t_b['significant_player_removed'] == 1)]
    c_t_4 = c_t_b.loc[(c_t_b['significant_player_added'] == 1) & (c_t_b['significant_player_removed'] == 1)]
    print ('------------------ Total Results by Lineup Changes (Edge > 0.5)')
    print ('                 Open Proportion     Open Results     Close Proportion     Close Results')
    print ('Mostly Same          ' + str(round(len(o_t_1.index)/len(pred.index),3)) + '               ' +
           str((o_t_1['open_total_result'].value_counts()['W'] / (o_t_1['open_total_result'].value_counts()['W'] + o_t_1['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_1.index)/len(pred.index),3)) + '               ' +
           str((c_t_1['close_total_result'].value_counts()['W'] / (c_t_1['close_total_result'].value_counts()['W'] + c_t_1['close_total_result'].value_counts()['L'])).round(3)))
    print ('Significant Addition ' + str(round(len(o_t_2.index)/len(pred.index),3)) + '              ' +
           str((o_t_2['open_total_result'].value_counts()['W'] / (o_t_2['open_total_result'].value_counts()['W'] + o_t_2['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_2.index)/len(pred.index),3)) + '               ' +
           str((c_t_2['close_total_result'].value_counts()['W'] / (c_t_2['close_total_result'].value_counts()['W'] + c_t_2['close_total_result'].value_counts()['L'])).round(3)))
    print ('Significant Removal  ' + str(round(len(o_t_3.index)/len(pred.index),3)) + '              ' +
           str((o_t_3['open_total_result'].value_counts()['W'] / (o_t_3['open_total_result'].value_counts()['W'] + o_t_3['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_3.index)/len(pred.index),3)) + '               ' +
           str((c_t_3['close_total_result'].value_counts()['W'] / (c_t_3['close_total_result'].value_counts()['W'] + c_t_3['close_total_result'].value_counts()['L'])).round(3)))
    print ('Both Significances   ' + str(round(len(o_t_4.index)/len(pred.index),3)) + '              ' +
           str((o_t_4['open_total_result'].value_counts()['W'] / (o_t_4['open_total_result'].value_counts()['W'] + o_t_4['open_total_result'].value_counts()['L'])).round(3)) + '               ' +
           str(round(len(c_t_4.index)/len(pred.index),3)) + '               ' +
           str((c_t_4['close_total_result'].value_counts()['W'] / (c_t_4['close_total_result'].value_counts()['W'] + c_t_4['close_total_result'].value_counts()['L'])).round(3)))
    

#takes the name of the model; file with the name must exist in /predictions/betting/pre_bet/
#2014-15 thru 2018-19
def backtest_bet(model, bandaid, var_factor=1):
    odds = pd.read_csv('./database/odds.csv')
    pred = pd.read_csv('./predictions/betting/pre_bet/' + model + '.csv')

    drop_cols = []
    for col in odds.columns:
        if ('game_id' not in col and 'score' not in col and 'bet365' not in col):
            drop_cols.append(col)
    odds = odds.drop(columns=drop_cols)

    seasons = ""
    for yr in range(2014, 2019):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    pred = pred[pred["season"].str.contains(seasons[:-1])]
    
    pred = pred.merge(odds,how='inner',on='game_id').reset_index(drop=True)

    table = []

    for index in tqdm(range(len(pred.index))):
        if (not bandaid):
            pred_dist = score_joint_pmf(pred.at[index, 'pred_pace'],pred.at[index, 'h_rtg_pred'],pred.at[index, 'a_rtg_pred'], var_factor)
        else:
            pred_dist = score_joint_pmf_bandaid(pred.at[index, 'pred_pace'],pred.at[index, 'h_rtg_pred'],pred.at[index, 'a_rtg_pred'], var_factor)

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
    save_name = model
    if (bandaid):
        save_name += "_bandaid"
    if (var_factor != 1):
        save_name += "_varfactor"+str(var_factor)
    
    pred.to_csv("./predictions/betting/post_bet/"+save_name+".csv", index=False)
    

def backtest_eval(model,playoff_reg):
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
    
    bankroll_growth_graph(pred,model,playoff_reg)
    calibration_curve_edge(pred,model,playoff_reg)
    print_dashboard(pred, player_added_missing=False)

#backtest_bet('third_w_best_eff_reg_season_last_lu_test',bandaid=True, var_factor=1.5)
backtest_eval('third_w_best_eff_reg_season_last_lu_test','reg')
#backtest_bet('third_expanding_reg')
#backtest_eval('third_expanding_reg','reg')
#backtest_bet('third_expanding_xgb')
#backtest_eval('third_expanding_xgb','reg')
#backtest_bet('second_xgb')
#backtest_eval('second_xgb','reg')
