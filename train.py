# from env.FutureEnv import FutureTradingEnv
from PPO_model.PPO import PPO_
from stable_baselines3.common.logger import configure
import pandas as pd

from callback.CustomBack import CustromCallback
from util import add_features
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    data = pd.read_csv('./甲醛.csv')
    data = add_features(data)
    data = data.sort_values('date')
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month / 12 - 0.5
    data['day'] = data['date'].dt.day / 30 - 0.5
    data['weekday'] = data['date'].dt.weekday / 6 - 0.5
    data['hour'] = data['date'].dt.hour / 23 - 0.5
    data['minute'] = data['date'].dt.minute / 59 - 0.5

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    model = PPO_(data,log_dir)

    callback = CustromCallback(log_dir)

    model.learn(total_timesteps=10000,callback=callback)



