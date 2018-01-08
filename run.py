

import  pandas as pd
import  numpy as np

data_path = 'C:\\Users\\Lior\\StockSimilarity\\sandp500\\all_stocks_1yr.csv'

all_stocks = pd.read_csv(data_path)
all_stocks['Date'] =  pd.to_datetime(all_stocks['Date'])

stock_names = all_stocks['Name'].unique()
from dtw import dtw

all_stocks=all_stocks.dropna(axis=0)
stock = "GOOG"


def clac_dist(all_stocks, stock_names):
    return

all_stocks = all_stocks[:1]
similarities = \
    [
        [
            dtw(all_stocks[all_stocks['Name'] == stock_name_1][['Open', 'Date']]['Open'].tolist(),
                all_stocks[all_stocks['Name'] == stock_name_2][['Open', 'Date']]['Open'].tolist(),
                dist=lambda x, y: abs(x - y))
            for stock_name_2 in stock_names
            ]
        for stock_name_1 in stock_names
    ]

dist_df = pd.DataFrame()
for i in range(len(stock_names)):

    for j in range(i,len(stock_names)):

        stock0 = all_stocks[all_stocks['Name'] == stock_names[i]][['Open', 'Date']]
        stock1 = all_stocks[all_stocks['Name'] == stock_names[j]][['Open', 'Date']]
        x = (stock0['Open'].tolist())
        y = (stock1['Open'].tolist())

        dist, cost, acc, path = dtw(x, x , dist=lambda x, y: abs(x - y))

window = 10
slide = 1
next_t = np.asarray([1,3,7])

i=0
x = np.array(x)
while i < len(x):
    instance = x[i:i+window]
    next_ti = i + window + next_t
    labeles = x[next_ti]
    i+=slide

