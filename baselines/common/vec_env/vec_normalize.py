from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import gym

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, mask = None):
        VecEnvWrapper.__init__(self, venv)

        if type(self.observation_space) == gym.spaces.Tuple:
            self.ob_rms = tuple(RunningMeanStd(shape=i.shape) for i in self.observation_space.spaces) if ob else None
            if not mask:
                mask = [True]*len(self.observation_space.spaces)
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask = mask

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if type(self.observation_space) == gym.spaces.Tuple:
                for obs_i, ob_rms_i in zip(obs, self.ob_rms):
                    ob_rms_i.update(obs_i)
                obs = tuple(
                    np.clip(
                        (obs_i - ob_rms_i.mean) / np.sqrt(ob_rms_i.var + self.epsilon), -self.clipob, self.clipob
                    ) if mask_i else obs_i for obs_i, ob_rms_i, mask_i in zip(obs, self.ob_rms, self.mask)
                )
                return obs
            else:
                self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)
