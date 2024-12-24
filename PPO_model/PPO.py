from stable_baselines3.common.vec_env import DummyVecEnv
from callback.CustomBack import CustromCallback
from env.FutureEnv import FutureTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from PPO_model.CustomActorCriticPolicy import CustomActorCriticPolicy,CustomRNN

def PPO_(data,log_dir):
    policy_kwargs = dict(
        features_extractor_class=CustomRNN,
        features_extractor_kwargs=dict(features_dim=128)
    )

    env = make_vec_env(lambda: FutureTradingEnv(data), n_envs=4,monitor_dir=log_dir)

    model = PPO(CustomActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs, batch_size=512)

    return model
