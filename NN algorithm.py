import pandas as pd
from talib.abstract import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:/Users/HP/Desktop/Binary_bots/Untitled Folder/EURUSD1.csv")
df = df.dropna()
df["Future"] = df["Future"].astype(int)

df['MA_2'] = df['close'].rolling(window = 2).mean() #moving average 20
df['MA_5'] = df['close'].rolling(window = 5).mean() #moving average 50
df['MA_20'] = df['close'].rolling(window = 20).mean() #moving average 50

df['EMA_2'] = df['close'].ewm(span = 2, adjust = False).mean() #exponential moving average
df['EMA_5'] = df['close'].ewm(span = 5, adjust = False).mean()
df['EMA_20'] = df['close'].ewm(span = 20, adjust = False).mean()
df

#########################################################################################
#Momentum indicator
df['adx'] = pd.Series(ADX(df['high'], df['low'], df['close'], timeperiod=14))
df['adxr'] = pd.Series(ADXR(df['high'], df['low'], df['close'], timeperiod=14))
df['apo'] = pd.Series(APO(df['close'], fastperiod=12, slowperiod=26, matype=0))

aroondown, aroonup = AROON(df['high'], df['low'], timeperiod=14)
df['aroondown'] = pd.Series(aroondown)
df['aroonup'] = pd.Series(aroonup)
del aroondown
del aroonup

df['aroonosc'] = pd.Series(AROONOSC(df['high'], df['low'], timeperiod=14))
df['bop'] = pd.Series(BOP(df['open'], df['high'], df['low'], df['close']))
df['cci'] = pd.Series(CCI(df['high'], df['low'], df['close'], timeperiod=14))
df['cmo'] = pd.Series(CMO(df['close'], timeperiod=14))
df['dx'] = pd.Series(DX(df['high'], df['low'], df['close'], timeperiod=14))

macd, macdsignal, macdhist = MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'] = pd.Series(macd)
df['macdsignal'] = pd.Series(macdsignal)
df['macdhist'] = pd.Series(macdhist)
del macd
del macdsignal
del macdhist

df['mfi'] = pd.Series(MFI(df['high'], df['low'], df['close'],  df['volume'], timeperiod=14))
df['minus_di'] = pd.Series(MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14))
df['minus_dm'] = pd.Series(MINUS_DM(df['high'], df['low'], timeperiod=14))
df['mom'] = pd.Series(MOM(df['close'], timeperiod=10))
df['roc'] = pd.Series(ROC(df['close'], timeperiod=10))
df['rocp'] = pd.Series(ROCP(df['close'], timeperiod=10))
df['rocr'] = pd.Series(ROCR(df['close'], timeperiod=10))
df['rsi'] = pd.Series(RSI(df['close'], timeperiod=14))

slowk, slowd = STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df['slowk'] = pd.Series(slowk)
df['slowd'] = pd.Series(slowd)
del slowk
del slowd

fastk, fastd = STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
df['fastk'] = pd.Series(fastk)
df['fastd'] = pd.Series(fastd)
del fastk
del fastd

df['ultosc'] = pd.Series(ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28))
#########################################################################################

# #Pattern Recognition
# df['cdl2crows'] = pd.Series(CDL2CROWS(df['open'], df['high'], df['low'], df['close']))
# df['cdl3blackcrows'] = pd.Series(CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close']))
# df['cdl3inside'] = pd.Series(CDL3INSIDE(df['open'], df['high'], df['low'], df['close']))
# df['cdl3linestrike'] = pd.Series(CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close']))
# df['cdl3outside'] = pd.Series(CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close']))
# df['cdl3starsinsouth'] = pd.Series(CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close']))
# df['cdl3whitesoldiers'] = pd.Series(CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close']))
# df['cdlabandonedbaby'] = pd.Series(CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close']))
# df['cdladvanceblock'] = pd.Series(CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close']))
# df['CDLBELTHOLD'] = pd.Series(CDLBELTHOLD(df['open'], df['high'], df['low'], df['close']))
# df['CDLBREAKAWAY'] = pd.Series(CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close']))
# df['CDLCLOSINGMARUBOZU'] = pd.Series(CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close']))
# df['CDLCONCEALBABYSWALL'] = pd.Series(CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close']))
# df['CDLCOUNTERATTACK'] = pd.Series(CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close']))
# df['CDLDARKCLOUDCOVER'] = pd.Series(CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close']))
# df['CDLDOJI'] = pd.Series(CDLDOJI(df['open'], df['high'], df['low'], df['close']))
# df['CDLDRAGONFLYDOJI'] = pd.Series(CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close']))
# df['CDLENGULFING'] = pd.Series(CDLENGULFING(df['open'], df['high'], df['low'], df['close']))
# df['CDLEVENINGDOJISTAR'] = pd.Series(CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close']))
# df['CDLEVENINGSTAR'] = pd.Series(CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']))
# df['CDLGAPSIDESIDEWHITE'] = pd.Series(CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close']))
# df['CDLGRAVESTONEDOJI'] = pd.Series(CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close']))
# df['CDLHAMMER'] = pd.Series(CDLHAMMER(df['open'], df['high'], df['low'], df['close']))
# df['CDLHANGINGMAN'] = pd.Series(CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close']))
# df['CDLHARAMI'] = pd.Series(CDLHARAMI(df['open'], df['high'], df['low'], df['close']))
# df['CDLHARAMICROSS'] = pd.Series(CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close']))
# df['CDLHIGHWAVE'] = pd.Series(CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close']))
# df['CDLHIKKAKE'] = pd.Series(CDLHIKKAKE(df['open'], df['high'], df['low'], df['close']))
# df['CDLHIKKAKEMOD'] = pd.Series(CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close']))
# df['CDLHOMINGPIGEON'] = pd.Series(CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close']))
# df['CDLIDENTICAL3CROWS'] = pd.Series(CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close']))
# df['CDLINNECK'] = pd.Series(CDLINNECK(df['open'], df['high'], df['low'], df['close']))
# df['CDLINVERTEDHAMMER'] = pd.Series(CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close']))
# df['CDLKICKING'] = pd.Series(CDLKICKING(df['open'], df['high'], df['low'], df['close']))
# df['CDLKICKINGBYLENGTH'] = pd.Series(CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close']))
# df['CDLLADDERBOTTOM'] = pd.Series(CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close']))
# df['CDLLONGLEGGEDDOJI'] = pd.Series(CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close']))
# df['CDLLONGLINE'] = pd.Series(CDLLONGLINE(df['open'], df['high'], df['low'], df['close']))

#########################################################################################
df['LINEARREG'] = pd.Series(LINEARREG(df['close'], timeperiod=14))
df['LINEARREG_ANGLE'] = pd.Series(LINEARREG_ANGLE(df['close'], timeperiod=14))
df['LINEARREG_INTERCEPT'] = pd.Series(LINEARREG_INTERCEPT(df['close'], timeperiod=14))
df['LINEARREG_SLOPE'] = pd.Series(LINEARREG_SLOPE(df['close'], timeperiod=14))
df['TSF'] = pd.Series(TSF(df['close'], timeperiod=14))
#########################################################################################

x_train, x_test, y_train, y_test = train_test_split(df.drop('Future',axis=1),  df['Future'], test_size=0.3)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.array(x_train)
y_train = np.asarray(y_train)

x_test = np.array(x_test)
y_test = np.asarray(y_test)

x_train = x_train[:45000]
y_train =y_train[:45000]

x_test = x_test[:19400]
y_test = y_test[:19400]

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], -1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],-1)
#########################################################################################

LEARNING_RATE = 0.001 #isso mesmo
EPOCHS = 20  # how many passes through our data #20 was good
BATCH_SIZE = 128  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.

earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())

model.add(LSTM(128))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=5e-5)
    
# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
model.summary()

# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    
)
