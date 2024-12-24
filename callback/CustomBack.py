import os.path

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


class CustromCallback(BaseCallback):
    def __init__(self,log_dir:str,verbose=1,check_freq=10):
        super().__init__(verbose)
        self.check_freq=check_freq
        self.log_dir=log_dir
        self.save_path=os.path.join(log_dir,'best_model')
        self.best_mean_reward=-np.inf
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path,exist_ok=True)
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x,y=ts2xy(load_results(self.log_dir),'episodes')
            if len(x)>0:
                mean_reward=np.mean(y[-50:])
                if self.verbose>0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                    # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True

