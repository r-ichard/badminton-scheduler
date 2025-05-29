# Bad planning calculator
The goal of this script is to calculate in which order, where and when players in a Badminton tournament will compeet agains each other.
The computing is made in a way that the players spend at least 20 minutes of rest between each match but don't wait too much it's optimized to improve planning and time distribution.

It takes as input :
The tournament information :
- Name
- Number of court where people will play
- Start time
- End time
- The number of players in total
In addition to those inside a tournament there can be one or more series.
It also takes as input :
- The name of the series
- The mean duration time for a match in this series
- The number of people in a series
In additon to those inside each series we can have one or more pool.
If there is a single pool it means that each player will play agains each other.
If there are multiple pools it means those describe how people will compeet against each other.
A pool is composed of a number of contestant that wil compeet agains each other (player A , player B, player C) and a number of people winners.

### How it works
First we need to make some basic verification :
1. Is total number of players logic based on series and pool
2. Based on mean time in each series is it possible to play every match the same day with this number of court and configuration.
3. Are number of people in series logical based on pool number.

Then computation takes place:
First we schedule in the time we have when each match of pools should take place.
All players in a pool should compeet agains each other (e.g. a pool of 3 people - player A vs player B, player A vs player C, player B vs player C )
if the number of winner is configured to be  higher then 1 then we count points and the players with the most points go to eliminations.
We devide series and pools to court based on the following constraints :
- Mean duration of match should be taken into account
- Each player must have at least 20 minutes of rest between matches.

### Eliminations
The winners of each pools compeet agains each other in pairs in rounds until there is only 1 winners.

### Edge case
It's possible that some configuration will not allow each player in each pools compeeting against  each other in eliminations because it will create redundant matches
in this case the player(s) with the best score are considered "winners" of a first match and will directly fight the next round of eliminations.

## Output
This script outputs a clear agenda with times for each pool first and then the eliminations rounds.
It also cleary outputs warning based on the provided configuration.
