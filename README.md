# NFL-Betting
Finding positive EV betting lines in the NFL season <br>

Game data sourced from https://www.pro-football-reference.com/ <br>
Betting data sourced from http://www.aussportsbetting.com/historical_data/nfl.xlsx <br>
ELO data (unused in the project but sometimes interesting to look at) sourced from https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv <br>

**Objective** <br>
Create a long-term EV-positive betting strategy for the NFL <br>

**Method** <br>
Identify how the difference in a GradientBoostingClassifier's win probability predictions and the win probability implied by the odds translates into real-life win probability. <br>
Then using this real-life win probability and the potential payouts of the bets, calculate the expected value of betting on any given team in any given game. <br>
Bet on a team when they have an expected value greater than 0.15 (in back-testing, a threshold of 0.15 seemed to give a good enough cushion to account for prediction error) and their expected value is greater than that of their opponent's (you cannot bet on both sides). <br>

**Findings** <br>
This unique method (not relying solely on a predictive model for probabilities, but instead using the real-life probabilities that the difference between a predictive model and the book's implied probabilities correlate with) seems to be successful at finding bets with positive EVs.
