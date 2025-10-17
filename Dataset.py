import tensorflow as tf
import yfinance as yf
import numpy as np
import pandas as pd
import talib

coid = 'PYPL'
history = '10y'
data = yf.Ticker(coid).history(period = history, interval = '1d')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data[:-1]
data['WMA'] = talib.WMA(data['Close'], timeperiod=5)  # 计算30日简单移动平均线
data['EMA'] = talib.EMA(data['Close'], timeperiod=5)  # 计算30日指数移动平均线
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)  # 计算14日相对强弱指数
data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)  # 计算MACD

PAST_WIN_LEN = 10
CLASSES = ["Bull","Bear"]
LABEL_BULL = CLASSES. index ( "Bull")
LABEL_BEAR = CLASSES. index ("Bear")
x, y = [], []
for today_i in range(len(data)) :
# Get day-K in the past 100-day window and the forvard 1-day window
    day_k_past = data[ :today_i+1]
    day_k_forward = data[today_i+1:]
    if len(day_k_past) < PAST_WIN_LEN or len(day_k_forward) < 1:
        continue
    day_k_past_win = day_k_past [-PAST_WIN_LEN: ]
    day_k_forward_win = day_k_forward[:1]
    # Find label
    today_price = day_k_past_win.iloc[-1]["Close"]
    tomorrow_price = day_k_forward_win.iloc[0][ "Close"]
    label = LABEL_BULL if tomorrow_price > today_price else LABEL_BEAR
    x.append(day_k_past_win.values)
    y.append(label)
x , y = np.array(x) , np.array(y)
print(x.shape ,y.shape)

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1
# Take the last portion to be the test dataset
test_split_index = -round(len(x) * TEST_SPLIT)
x_other, x_test = np.split(x, [test_split_index])
y_other, y_test = np.split(y, [test_split_index])
# shuffle the remaining portion and split into training and validation datasets
train_split_index = round(len(x) * TRAIN_SPLIT)
indexes = np. arange(len(x_other) )
np. random. shuffle (indexes)
train_indexes, val_indexes = np.split(indexes, [train_split_index])
x_train, x_val = x_other[train_indexes], x_other[val_indexes]
y_train, y_val = y_other[train_indexes], y_other[val_indexes]

label_distribution = pd.DataFrame([{'Dataset':'Train',
                                    'Bull':np.count_nonzero(y_train == LABEL_BULL),
                                    'Bear':np.count_nonzero(y_train == LABEL_BEAR)},
                                    {'Dataset':'Val',
                                    'Bull':np.count_nonzero(y_train == LABEL_BULL),
                                    'Bear':np.count_nonzero(y_train == LABEL_BEAR)},
                                    {'Dataset':'Test',
                                    'Bull':np.count_nonzero(y_train == LABEL_BULL),
                                    'Bear':np.count_nonzero(y_train == LABEL_BEAR)},])
print(label_distribution)

x_test_bull = x_test[y_test== LABEL_BULL]
x_test_bear = x_test[y_test== LABEL_BEAR]
min_n_labels = min(len(x_test_bull), len(x_test_bear) )
x_test_bull = x_test_bull [np.random.choice(len(x_test_bull), min_n_labels, replace= False),:]
x_test_bear = x_test_bear [np.random.choice(len(x_test_bear), min_n_labels, replace= False),:]
x_test = np.vstack([x_test_bull, x_test_bear])
y_test = np.array([LABEL_BULL] * min_n_labels + [LABEL_BEAR] * min_n_labels)
# Test dataset label distribution
z = pd.DataFrame ([{"Dataset": "test",
                "Bull": np.count_nonzero(y_test == LABEL_BULL),
                "Bear": np.count_nonzero(y_test == LABEL_BEAR)}])
print(z)
np.savez('datasets.npz', x_train = x_train, y_train=y_train,
         x_val = x_val, y_val = y_val, x_test = x_test, y_test = y_test)

