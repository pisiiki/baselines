from typing import *

import gym
import numpy as np
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env import VecEnvWrapper, VecEnv


class VecNormalize(VecEnvWrapper):
    def __init__(
            self,
            venv: VecEnv,
            training: bool,
            norm_clip_ob: Optional[float] = 10.,  # None means not normalizing/clipping observations.
            norm_clip_rew: Optional[float] = 10.,  # None means not normalizing/clipping rewards.
            gamma: float = 0.99, # Used to track per env. discounted returns. Should match RL algo. gamma.
            epsilon: float = 1e-8,
            ob_tuple_norm_mask: Optional[Tuple[bool, ...]] = None  # Selectively normalize tuple observation items.
    ):
        assert norm_clip_ob is not None or norm_clip_rew is not None
        assert norm_clip_ob is None or norm_clip_ob > 0.
        assert norm_clip_rew is None or norm_clip_rew > 0.
        assert gamma > 0.
        assert epsilon > 0.

        VecEnvWrapper.__init__(self, venv)

        self._training = training

        self.ob_rms = None
        self.ret_rms = None
        self.ob_tuple_norm_mask = None

        if type(self.observation_space) == gym.spaces.Tuple:
            if norm_clip_ob is not None:
                self.ob_rms = tuple(RunningMeanStd(shape=i.shape) for i in self.observation_space.spaces)
            if ob_tuple_norm_mask is None:
                ob_tuple_norm_mask = (True,) * len(self.observation_space.spaces)
        else:
            if norm_clip_ob is not None:
                self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)

        if norm_clip_rew is not None:
            self.ret_rms = RunningMeanStd(shape=())

        self.norm_clip_ob = norm_clip_ob
        self.norm_clip_rew = norm_clip_rew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ob_tuple_norm_mask = ob_tuple_norm_mask

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, terminals)
        """
        obs, rews, terminals, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            if self._training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.norm_clip_rew, self.norm_clip_rew)
        self.ret[terminals] = 0.
        return obs, rews, terminals, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if type(self.observation_space) == gym.spaces.Tuple:
                if self._training:
                    for obs_i, ob_rms_i in zip(obs, self.ob_rms):
                        ob_rms_i.update(obs_i)
                obs = tuple(
                    np.clip(
                        (obs_i - ob_rms_i.mean) / np.sqrt(ob_rms_i.var + self.epsilon),
                        -self.norm_clip_ob,
                        +self.norm_clip_ob
                    ) if norm_i else obs_i for obs_i, ob_rms_i, norm_i in zip(obs, self.ob_rms, self.ob_tuple_norm_mask)
                )

            else:
                if self._training:
                    self.ob_rms.update(obs)
                obs = np.clip(
                    (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                    -self.norm_clip_ob,
                    +self.norm_clip_ob
                )

        return obs

    def reset(self):
        """
        Reset all environments
        """
        self.ret[:] = 0.
        obs = self.venv.reset()
        return self._obfilt(obs)
