# Improving S&P stock prediction with time series stock similarity

## Overview
Stock market prediction with forecasting algorithms is a popular topic these days where most of the forecasting algorithms train only on data collected on a particular stock.
In this work, we enriched the stock data with related stocks just as a professional trader would have done to improve the stock prediction models. 
We tested five different similarities functions and found co-integration similarity to have the best improvement on the prediction model. 
We evaluate the models on seven S&P stocks from various industries over five years period. The prediction model we trained on similar stocks had significantly better results with 0.55 mean accuracy, and 19.782 profit compare to the state of the art model with an accuracy of
0.52  and profit of 6.6.

## Framework
In order to evaluate if stocks similarity improves a baseline model, we conduct two-step experiments (back-testing) to evaluate different types of configurations.
The first experiment goal is to come up with a processing pipeline and a baseline model. 
The second experiment is to evaluate how different stock similarity functions influence the baseline model. 

A configuration tree of all the setup to be optimize and evaluate in workflow pipeline

![Configurations](https://github.com/liorsidi/StockSimilarity/blob/master/configurations.png)

The [S&P dataset](https://github.com/liorsidi/StockSimilarity/tree/master/sandp500) contains daily historical data for all the S&P (Standard & Poor) 500 stock market index companies from 2012 to 2017. 
The features given are date, open price, closing price, highest price, lowest price, volume, and the short name of the stock. 
The S&P is an American stock index of the largest companies listed in NYSE or NASDAQ, maintained by S\&P Dow Jones Indices. 
It covers about 80 percent of the American equity market by capitalization.

We apply the evaluation process on stocks from different industries: 
* Consumer (Disney - DIS, Coca Cola KO)
* Health (Johnson and Johnson - JNJ)
* Industrial (General electric - GE , 3M - MMM)
* Information technology (Google - GOOGL)
* Financial (JP Morgan - JPM)

The validation folds are set to five and prepared for each stock separately. 

## Results
The evaluation metrics are accuracy score and F1 score; we calculate each metric per class (increase/decrease) and average it to one score. 
To evaluate the model profit, we implement a simple Buy & Hold algorithm that applies a long or short position regarding the model price prediction.
We also measure the risk of the strategy with the Sharp ratio.


**Experiment 1 processing parameters results** - transformation function, features and temporal modeling. 
(rows - configuration , columns - metrics and color - profit scale)

![exp1_prep](https://github.com/liorsidi/StockSimilarity/blob/master/exp1_prep.png)

**Experiment 1 prediction parameters results** - prediction model, Horizon and Value (rows - configuration, columns - prediction value with metrics and color - profit scale

![exp1_models](https://github.com/liorsidi/StockSimilarity/blob/master/exp1_models.png)

**Experiment 2 random selection compare** - a profit comparison between SAX and co-integration similarities on top 50 stocks and random stock selection

![exp2_rand_sim](https://github.com/liorsidi/StockSimilarity/blob/master/exp2_rand_sim.png)

**Experiment 2 folds profit per stock** - a profit comparison between top 50 stocks from co-integration similarity(orange) and 100 random stock selection enhancement (Blue) for each stock (x axis) in different folds (y axis)

![exp2_profit_plot](https://github.com/liorsidi/StockSimilarity/blob/master/exp2_profit_plot.png)

for more information on the methods results, and deeper analysis of the stock similarities check our [paper pdf](https://github.com/liorsidi/StockSimilarity/blob/master/full_paper.pdf).
