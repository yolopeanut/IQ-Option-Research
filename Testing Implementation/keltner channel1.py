from iqoptionapi.stable_api import IQ_Option
import logging, sys, time, configparser
from talib.abstract import ATR, EMA
import numpy as np
import pandas as pd
from datetime import date
from csv import writer

def login(verbose = False, iq = None, checkConnection = False):
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

    if iq == None:
      print("Trying to connect to IqOption")
      iq=IQ_Option('Username','Pass') # YOU HAVE TO ADD YOUR USERNAME AND PASSWORD
      iq.connect()

    if iq != None:
      while True:
        if iq.check_connect() == False:
          print('Error when trying to connect')
          print(iq)
          print("Retrying")
          iq.connect()
        else:
          if not checkConnection:
            print('Successfully Connected!')
          break
        time.sleep(3)

    iq.change_balance("PRACTICE") #or real
    return iq

def higher(iq,Money,Actives):
    
    done,id = iq.buy(Money,Actives,"call",1)
    print(Money, id)
    if not done:
        print('Error call')
        print(done, id)
        exit(0)
    
    return id


def lower(iq,amount,pair):
    
    done,id = iq.buy(amount,pair,"put",1)
    print(amount, id)
    
    if not done:
        print('Error put')
        print(done, id)
        exit(0)
    
    return id

def atr(values, period):
    return ATR(values['high'], values['low'], values['close'],period)

def keltner(values, period, offset):
    ema = EMA(values['close'], timeperiod=period) - 0.00004
    upperb = ema + offset * atr(values, period)
    lowerb = ema - offset * atr(values, period)
    return upperb, lowerb, ema

def get_candles(pair, candle_freq):
    candles = API.get_realtime_candles(pair, candle_freq)
    values = {'open': np.array([]), 
                  'high': np.array([]), 
                  'low': np.array([]), 
                  'close': np.array([]), 
                  'volume': np.array([]) }
        
        
    for x in candles:
        values['open'] = np.append(values['open'], candles[x]['open'])
        values['high'] = np.append(values['high'], candles[x]['max'])
        values['low'] = np.append(values['low'], candles[x]['min'])
        values['close'] = np.append(values['close'], candles[x]['close'])
        values['volume'] = np.append(values['volume'], candles[x]['volume'])

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in values.items() ]))
    return df, values

logging.disable(level=(logging.DEBUG))

user_file = configparser.RawConfigParser()
user_file.read('config_user.txt')    
API = IQ_Option(user_file.get('USER', 'user'), user_file.get('USER', 'password'))
API.connect()

API.change_balance('PRACTICE') # PRACTICE / REAL

if API.check_connect():
    print('\n\nSuccessful connection')
else:
    print('\n Error connecting')
    sys.exit()


pair = 'EURUSD-OTC'
amount = 10

period_EMA2 = 25
candle_freq = 300
API.start_candles_stream(pair, candle_freq, 280)

mid_way = None
upperb = None

GupperB = False
LlowerB = False

trade = False

while True:
    if int(time.strftime("%S", time.localtime())) < 5:
    # if True:
        df, values = get_candles(pair, candle_freq) 
        upperb, lowerb, ema = keltner(values, 25, 2)
        close = df.at[df.index[-2],'close']
        op = df.at[df.index[-2],'open']
        upperb = round(upperb[-2],5)
        ema = round(ema[-2],5)
        lowerb = round(lowerb[-2],5)
        print(f"     Close: {close}")        
        print(f"Upper Band: {upperb}")
        print(f"  EMA Band: {ema}")
        print(f"Lower Band: {lowerb}")
        
        #If in either top or bottom bound
        if close > ema:
            upperb = True
        elif close <ema:
            upperb = False
            
        #If crosses the midway 
        if upperb == True:
            if close<ema:
                mid_way = True
        elif upperb == False:
            if close>ema:
                mid_way = True
        
        #If greater than upperbound
        if mid_way == True and upperb == True:
            if close > upperb:
                GupperB = True
            elif close < upperb:
                GupperB = False

        #If less than lowerbound
        if mid_way == True and upperb == False:
            if close < lowerb:
                LlowerB = True
            elif close > lowerb:
                LlowerB = False            
        
        #If green candle
        if GupperB == True:
            if close>op:
                lower(API, amount, pair)
                mid_way = False
                trade = True
                
        #If red candle
        if LlowerB == True:
            if close<op:
                higher(API, amount, pair)
                mid_way = False
                trade = True
        
        print(f"{upperb= }")
        print(f"{mid_way= }")
        print(f"{GupperB= }")
        print(f"{LlowerB= }")
        if trade == True:
            
            sec = int(time.strftime("%S", time.localtime()))
            while(sec != 1): #wait till 1 to see if win or lose
                sec = int(time.strftime("%S", time.localtime()))
                
            betsies = API.get_optioninfo_v2(1)
            betsies = betsies['msg']['closed_options']
            
            bets = []
            for bt in betsies:
                bets.append(bt['win'])
            outcome = bets[-1:]
            print(outcome)
        
            #Appending history into csv file
            csvInsert = [date.today().strftime("%d/%m/%Y"),time.strftime("%H:%M:%S", time.localtime()), outcome, amount, API.get_balance()]
            with open('C:/Users/HP/Desktop/Binary_bots/binary-bot-master/Records/historyKC1.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(csvInsert)
                f_object.close()
        
        # time.sleep(5)
        print("------------------------------------------------------------------")
        time.sleep(10)
