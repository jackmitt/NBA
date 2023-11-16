# Model Chronological Experimentation

This file will be used to document the evolution of the model. 

There is so much to mess with that I likely would forget what has been promising and what hasn't.

The goal is to find the best performing models *against the following betting markets*:
* Moneyline - model the probability of each team winning a game
* Spread - model the score difference in the game
* Total - model the sum of scores in the game

Beyond general improvements and exploration concerning features and model types, I want to explore these questions from the start:
1. Is it better to model the spread and total, or is it better to model each team's scores and derive the spread and total? Does it depend on the model type, or is one way always better?
2. Will a significantly better fit translate to significantly better betting results? Expectation: no.
3. Will player-based models outperform team-based models? When they do, will there be anything to gain from considering the team-based model alongside it?

## Beginning

### Baseline

First, I just want to make and evaluate a simple model to use as a baseline.

At the present and until further notice, the data will be split up as follows:
* Training Set: 1998-99 through 2010-11 *note: 3 games missing data in the two preceding seasons where we have boxscore data
* Test Set: 2011-12 through 2019-20 until the Covid bubble
* Validation Set: Covid bubble through 2022-23

#### Model Description

For the simplest model, we will just use as features the average points scored and allowed all season for each team and homefield indicator.

To predict the spread and total, we will try it two different ways to consider question 2 above, both using OLS. For moneyline, we will use logistic regression.

For the first five games of the season, we will use a weighted average of the average points from the previous season and the average points from the current season. Thus one year is lost for the train set.

Each game played raises the current season weight by 0.2 until it is 1 after 5 games. 
