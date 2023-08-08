# NBA Prediction Modeling
## Luke DiPerna
### August 2023
![basket_splash_image](https://github.com/luke-lite/NBA-Prediction-Modeling/blob/7dd0df0e39e479758dc5b26affdbdd2dffc058ff/basket_splash_image.jpg)

## Project Goal
The purpose of this project is to create a machine learning model that can accurately predict the outcome of NBA games using boxscore statistics from the past 10 seasons. In order to accomplish this, I:

* built a webscraper to gather the raw data
* aggregated and processed the data to prepare it for modeling
* evaluated and iterated upon the models to improve accuracy

## Stakeholder
The stakeholder for this project is Stat-Ball, a sports news and entertainment website. The site plans to have fantasy drafts and competitions for predicting NBA game winners, so they want to build an in-house model they can use as a benchmark for users to compete against. The exact limitations and specifications are discussed in the [Data Overview](#data-overview) section.

## Table of Contents
* [Data Overview](#data-overview)
* [Modeling](#modeling)
* [Results](#results)
* [Next Steps](#next-steps)

The notebook containing the full code for this project: [NBA-Prediction-Modeling](https://github.com/luke-lite/NBA-Prediction-Modeling/blob/c2ee794d54425ebb60a0e02def46109a347b3215/NBA-Prediction-Modeling.ipynb)

## Data Overview
The final versions of the datasets I will be using include team aggregated statistics for the past 10, 20, and 30 games. To see the code used to perform the data aggregation, see the [data-aggregation notebook](https://github.com/luke-lite/NBA-Prediction-Modeling/blob/7dd0df0e39e479758dc5b26affdbdd2dffc058ff/data/data-aggregation.ipynb). The data has a highly normal distribution, so little pre-processing is needed.

### Data Collection
The first task was to determine how much and what kind of data would be needed. Given the timeline for the project, it was decided that I would collect all boxscore data from every NBA regular season game over the past 10 years. Future iterations could include playoff games and data from older seasons, but teams have been shown to perform differently in the playoffs, so it would require additional changes to the models to account for this. Additionally, modern machine learning models often utilize player tracking data as well, but this data is not as publicly available and would require a significant investment.

After the scope was decided, I needed to source the data. The majority of the raw data was gathered from [basketball-reference](https://www.basketball-reference.com/), a leading site for basketball related stats. I created a web-scraper to collect the metadata, player stats, and team stats for each NBA regular season game. The web-scraper can be found on my [Github](https://github.com/luke-lite/NBA-Web-Scraper) and a beginner-friendly demonstration can be found on my [website](https://luke-lite.github.io/blog/web-scraping-basketball-stats-in-python-with-selenium-and-beautiful-soup/).

The resulting dataset was put in a SQLite database that contains:

* 3 tables (game_info, player_stats, team_stats)
* 341,669 observations and 46 columns of data across 11,979 NBA games.

### Data Aggregation
The next step was to decide how best to aggregate the individual game data. Typically, averages of each statistic are calculated by using the stats from the previous *n* number of games. There were 3 main considerations:

* How "responsive" should the data be?
* Is team data or player data more effective for predicting outcomes?
* Which features are the most effective?

#### Responsiveness
"Responsiveness" means how much importance should be put on recent performance. For example, when trying to decide whether team A or team B will win, I can use the team stats from their previous game to make a prediction. But relying on only the previous game's stats will mean the resulting models are far less robust and more susceptible to outliers, since any time a team has a particularly good performance, the models are likely to assume a win in their next game, even if the team's nine previous games were all terrible. After a bit of research, the best performing data tends to account for around the previous 20-30 games. Because of this, I decided to compare model performance when looking at the average of a team's past 10, 20, and 30 games.

#### Team vs. Player Aggregation
For the question of player or team data, my assumption going in was that player data would lead to better performing models. However, it was unclear whether the gains in performance would be worth the extra time and computational costs. I already had the team data for each game, so combining previous game data was straightforward, but in order to aggregate data by player, I needed to get the full team roster, and for each player, find the last *n* games they had appeared in regardless of the team they played for, aggregate each individual player's past performance, then calculate the teams total performance in each statistic. This allows the models to respond more quickly to roster changes, such as injuries and trades, but it is unclear how much improvment the models will make. Due to the project timeline, player aggregated data will remain a possibility for future iterations.

#### Feature Selection
With around 35 numeric columns of data, I needed to decide how to select the most important features. Going in, I knew that some of the data would be colinear, and some would be poorly correlated with outcome prediction. It wouldn't be possible to try every combination of features, so I settled on a few key methods:

* Four Factor Data
* Principal Component Analysis
* Full dataset

The "Four Factors of Basketball Success" were proposed in the early 2000's by Dean Oliver, a leading sports statistician, and include the four most impactful boxscore stats to determine a team's win probability. The stats are effective field goal percentage (eFG%), turnover percentage (TOV%), offensive rebounding percentage (ORB%), and Free Throw rate (FTr). A more detailed description can be found [here](https://www.basketball-reference.com/about/factors.html). This is a relatively simplistic, yet proven and effective method for determining game outcomes, so it is perfect for the task at hand.

However, since I will be using modern machine learning techniques that are capable of finding their own connections in the data, I also wanted to perform tests using the full dataset. Principal Component Analysis (PCA) is a technique for reducing the dimensionality of the data by performing linear transformations that reduce the overall number of features in the dataset while still preserving the ability to describe as much of the variance in the data as possible. I also decided to use the full dataset without PCA when training a Neural Network model to see how it would handle the full list of features.

### Data Limitations
While modern basketball analysis still utilizes boxscore data, most advanced analysis also incorporates player tracking data, such as player movement and spacing. This data, however, is typically not publicly available, and given the time and resource constraints, it is outside the scope of the current project. Future iterations could also include playoff games and data from older seasons, but teams have been shown to perform differently in the playoffs, so it would require additional changes to the model to account for this.

Additionally, there are some limitations regarding team and player data aggregation. In the case of team aggregation, the data does not do a good job of accounting for changes in team rosters. In the event of a trade or injury, where a number of players are moved to new teams or removed from a team's roster, there will obviously be an immediate impact on the team's performance, but team aggregation only considers the team, not the players on the team roster. Likewise, new players entering the league, such as through the NBA draft, will have a varying impact on a team's performance, but the data has no way to account for this.

One option is to create a predictive metric that can approximate the expected contribution (or lack there of) from a player in case of a roster change, but these systems are highly specialized and require a great deal of time and additional computational costs, so they are currently outside the scope of this project.

Finally, there are implications for using only the past 10 seasons of data. Professional basketball has undergone a number of changes through it's history, largely as a result of changes in overall player skill and competitiveness, and rule changes. The modern NBA is far less physical and more offensively focused than, for example, the NBA in the 80's and 90's. Because of this, I suspect that some of the relative weights of the data features will differ from if I had included data from previous NBA eras. It is also possible that 10 seasons is not enough data for the more abstract ML models to shine. So additional data collection is a possibility for future project iterations.

## Modeling
I used a number of different modeling techniques to evaluate the relative effectiveness of each, with a plan to create an ensemble model that uses the best individual models. The modeling techniques I used include:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest (RF)
* Gaussian Naive-Bayes (GNB)
* Support Vector Classifier (SVC)
* Neural Network (NN)
* Elo Rating System

These models provide a mix of weak- and strong-learning binary classifiers. Some, like the logistic regression model, are fairly fast and efficient. But others, like the RF model, are significantly more computationally expensive and time-consuming.

### Baseline Model

The "baseline" model always selects the home team as the winner. There is a notable home-court advantage in the NBA, with the home team typically winning around 60% of their games in a given season. For my dataset, the exact value is 57.2%.

### Target Metric
Traditionally, the NBA has an upset rate of between 28-32%, meaning that the "better" team wins 68-72% of the time. Because of this, it is very challenging to create a model with an accuracy higher than this range. Given the limitations of the data I am using, I hope to achieve an accuracy that approaches the 68% threshold.

### Modeling Process
I began by testing the four factor data using the past 10, 20, and 30 game averages. The 10-game aggregation data underperformed, while the 20 and 30 game aggregation were similar in terms of average accuracy across all models. Ultimately, I decided to focus on the 20-game aggregation when testing the full dataset with all boxscore statistics.

I also evaluated the model error in more detail. Since I used team aggregation data, I know that the models are unable to quickly account for roster changes, which are mainly due to a few factors:

* injuries
* trades
* free-agency
* draft

My hypothesis was that the models will have less error in the second half of each season because of fewer roster changes. In the NBA, once the trade deadline has passed, rosters mostly remain the same outside of injuries and the occasional signing. At the start of a season, however, there will be a lot of uncertainty since the off-season is when we see the vast majority of roster changes. And since the team-aggregated data does not reset between seasons, the models are using data that has been carried over from the end of the previous season even though the rosters may be completely different. I broke down the error stats by taking the average error across all seasons, and splitting the error counts into season quarters. This was performed using the four factor 10-game aggregated data:

![model_error_per_season_quarter](https://github.com/luke-lite/NBA-Prediction-Modeling/blob/f2bdb3fc70f305814df41406ae729a5924e53dde/graphs/model_error_per_season_quarter.png)

It is clear that the models are behaving very similarly, not just in terms of overall accuracy, but also the error distribution over the course of a season. My hypothesis that the second half of each season would be less error-prone seems possible, but the cross-model similarity also suggests there is not enough information in the data to differentiate the models.

I also examined the average model error per season to see if there were any outliers. Since some seasons had fewer overall games than others, I adjusted the results to represent the average error per game for each season:

![average_error_per_game](https://github.com/luke-lite/NBA-Prediction-Modeling/blob/f2bdb3fc70f305814df41406ae729a5924e53dde/graphs/average_error_per_game.png)

Each season had close to 0.35-0.40 errors per game, meaning that for every 10 games in a given season, the models averaged about 3.5 to 4 errors. This matched the average model accuracy of around 60%.

After testing the four factor datasets, which averaged around 61-62% accuracy, I also used the PCA datasets and full datasets. I included the Neural Network (NN) model with these datasets, as I thought the NN model would be able to process the extra features more effectively. A detailed breakdown of the modeling results can be found in the [Results](#Results) section, but none of the models were able to reach the target of 68%. An ensemble model was also unlikely to reach the target since the individual models were all performing very similarly, so I chose to use an alternate method for the final model, an Elo rating system.

### Elo Rating System
A detailed breakdown of the mathematics and code for the Elo rating system can be found in a guide I wrote titled [How to create an NBA Elo Rating System](#). The Elo system I created was based on the [FiveThirtyEight Elo system](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) developed by Nate Silver. The article provides some great insight into the process by which they created their model, and I highly suggest reading it.

The main benefit of an Elo system is its simplicity. All that is required is the team names, their current Elo rating, and the outcome of each competition, and the system can make a prediction on who will win. For each game, the system predicts that the team with the higher current elo will win, and both teams Elo ratings are adjusted up or down depending on if they won or lost, respectively. Surprisingly, the Elo rating system outperformed every individual model with an accuracy of 65.3%, despite not utilizing any boxscore data. Given the relatively low computational costs and lack of a need to train complex machine learning models, it was clearly the best model for the task at hand.

## Results
This is a list of the top 10 models and the relative performance of the top 5 in terms of overall accuracy:

![]()

There were incremental improvements moving from 10-game averages to 20-game averages, but as mentioned earlier, the machine learning models all behaved simiarly, with a tendency to favor the home team (false-positives) and similar error distribution thoughout each season. Because of this, an ensemble model is unlikely to see large improvements in accuracy. Ultimately, my recommendation is for Stat-Ball to use the Elo rating system. It was the most accurate model, while also requiring the least amount of data and computation.

## Next Steps
Overall, the project was a success. I was able to collect the data, create and compare models and datasets, and select a model suitable for the stakeholder. The results did not meet the original goal of 68% accuracy, but there are several promising leads for future iterations of this project:

* more extensive data collection
* player aggregated data
* additional model adjustments

### Data Collection
It is likely that all models would benefit from additional data collection. Machine learning datasets typcially handle extremely large datasets well, so the additional information may lead to improvements in performance, particularly for the more complex models like the Neural Network. I plan to add an additional decade of boxscore data in the future. There are also opportunities for feature engineering that may better capture the information in the raw dataset, such as advanced statistics like PIE (Player Impact Estimate) and BPM (Box Plus/Minus). 

### Player Aggregated Data
As discussed before, player aggregated data has several benefits over team aggregated data. In particular, it has the ability to respond more quickly to roster changes, which could lead to significant improvements in accuracy at the beginning of each season. Player aggregation also allows for the possibility of creating predictive player metrics that can predict how a player will perform based on past performance and current teammates and opponents. For an example of how this would work in practice, see [How Our RAPTOR Metric Works]([How Our RAPTOR Metric Works](https://fivethirtyeight.com/features/how-our-raptor-metric-works/). FiveThrityEight's RAPTOR metric utilizes player tracking data in addition to boxscore data and is highly sophisticated, but it is possible to create a more simplified version based on the data I have available.

### Additional Model Adjustments
Assuming that additional data will help improve the machine learning models, there is a good chance that a custom-built ensemble method that also incorporates the Elo rating system will outperform the Elo system alone. I can also adjust the current Elo system to increase its complexity. The only adjustments it currently makes are home-court advantage and margin of victory. Other possible adjustments include back-to-back games, road trips, and even the elevation at which games are played. These additional features could also further improve the individual machine learning models.

## Repository Structure
```
├── data
├── graphs
├──.gitignore
├── NBA-Prediction-Modeling.ipynb
├── README.md
├── basket_splash_image.jpg
├── nbaenv.yml
```
