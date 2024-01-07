import math
from time import sleep
import datetime
import sys

import pandas as pd

import userdata
#from iqoption_api import IQOption as IQOptionAPI
from iqoptionapi.api import IQOptionAPI
import iqoptionapi.constants as api_constants


class IQOption():

    def __init__(self, user=userdata.mainUser, active_id="EURUSD"):
        self.username = user['username']
        self.password = user['password']

        self.real = user['real']
        self.practice = user['practice']

        self.active_id = active_id

        self.api = IQOptionAPI("iqoption.com", self.username, self.password)
        self.api.connect()
        #self.api.start_socket_connection()

        self.setExpirationTime()
        self.setActives(self.active_id)
        self.api.changebalance(self.practice)

    def openPosition(self, direction, amount=1):
        self.api.buy(amount, api_constants.ACTIVES[self.active_id], "turbo", direction)

    def getBalance(self):
        return self.api.profile.balance

    def setExpirationTime(self, time=1):
        self.api.timesync.expiration_time = time

    def setActives(self, activeID):
        self.active_id = activeID
        self.api.setactives([api_constants.ACTIVES[self.active_id]])

    def getCandles(self, duration=1, amount=25):
        self.api.getcandles(api_constants.ACTIVES[self.active_id], duration, amount)
        candles = self.api.candles.candles_data
        while not candles:
            sleep(0.1)
            self.api.getcandles(api_constants.ACTIVES[self.active_id], duration, amount)
            candles = self.api.candles.candles_data

        if candles:
            # print 'Timestamp: ', datetime.datetime.fromtimestamp(int(candles[-1][0])).strftime('%Y-%m-%d %H:%M:%S')
            # print 'Open: ', candles[-1][1]
            # print 'Close: ', candles[-1][2]
            # print 'High: ', candles[-1][3]
            # print 'Low: ', candles[-1][4]

            return candles[-3], candles[-2], candles[-1]
        else:
            print('Invalid Candle!')
            return None

    def getServerDateTime(self):
        print(str(self.api.timesync.server_datetime))
        return str(self.api.timesync.server_datetime)

    def getExpirationDateTime(self):
        print(str(self.api.timesync.expiration_datetime))
        return str(self.api.timesync.expiration_datetime)

    def getResult(self):
        try:
            result = self.api.listinfodata.current_listinfodata.win
            #print('Result: ' + result)
            return result
        except:
            print('Result Error!')
            return None

    def changeBalance(self, real=False):
        # To enable the real account:
        if real:
            self.api.changebalance(self.real)
        else:
            self.api.changebalance(self.practice)
            #self.api.change_balance('practice')

    def getDataFrame(self, duration=1):
        candles = self.getCandles(duration)
        if candles:
            try:
                # candlesDF -> dict type
                candlesDF = {
                    'DateTime': [candles[-3][0], candles[-2][0]],  # candles[-1][0]],
                    'Open': [candles[-3][1], candles[-2][1]],  # candles[-1][1]],
                    'Close': [candles[-3][2], candles[-2][2]],  # candles[-1][2]],
                    'High': [candles[-3][3], candles[-2][3]],  # candles[-1][3]],
                    'Low': [candles[-3][4], candles[-2][4]]  # , candles[-1][4]]
                }

            except:
                return None

        return candlesDF


def round_up(tm, nearest=1):
    upmins = math.ceil(float(tm.minute) / nearest) * nearest
    diffmins = upmins - tm.minute
    newtime = tm + datetime.timedelta(minutes=diffmins)
    newtime = newtime.replace(second=0)
    return newtime
