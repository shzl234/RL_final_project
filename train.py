#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
from reduction.PCA import PCA_RED  
import dmc2gym
import hydra


def make_env(cfg):
    """Helper function to create dm_control environment"""
    # print(cfg.domain_name)
    # assert False
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        # from utils import cfg_env
        utils.cfg_env=cfg.env
        # from utils import cfg_env
        # print(cfg_env)
        # assert False,utils.cfg_env
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        # cfg.agent.params.action_dim =3
        cfg.agent.params.action_dim=self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        action_pca=PCA_RED(3,'halfcheeth_acs')
        action_pca.load_weight()
        import warnings
        warnings.filterwarnings("ignore")
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                    # print(type(action))
                    if False:
                        # print(action.shape)
                        # print(action)
                        actionnn=action_pca.pca.inverse_transform(action[None,:])[0]
                        # print(action.shape)
                        # action=torch.tensor(action)
                        # print(action)
                        # print(self.env.action_space,'action_space')
                        # print(float(self.env.action_space.low.min()),'!!!',float(self.env.action_space.high.max()))
                        actionnn=actionnn.clip(float(self.env.action_space.low.min()),
                        float(self.env.action_space.high.max()))
                        # print(action)
                        # print(type(action))
                    else:
                        actionnn=action
                obs, reward, done, _ = self.env.step(actionnn)
                # self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        # assert False,3
        action_pca=PCA_RED(3,'halfcheeth_acs')
        # action_pca.load_weight()
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                print(episode_reward)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
               
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    if False:
                
                        # print(action.shape)
                        # print(action)
                        actionnn=action_pca.pca.inverse_transform(action[None,:])[0]
                        # print(action.shape)
                        # action=torch.tensor(action)
                        # print(action)
                        # print(self.env.action_space,'action_space')
                        # print(float(self.env.action_space.low.min()),'!!!',float(self.env.action_space.high.max()))
                        actionnn=actionnn.clip(float(self.env.action_space.low.min()),
                        float(self.env.action_space.high.max()))  
                    else:
                        actionnn=action
            # run training update
            if False:
                if self.step < self.cfg.num_seed_steps:
                    actionnn=action
            else:
                actionnn=action
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(actionnn)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            if False:
                if self.step < self.cfg.num_seed_steps:
                    actionnn=action
                    action=action_pca.pca.transform(action[None,:])
                    action=action[0]
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
        
    #     dataset_file='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/dataset/sac_large_halfcheeth.pkl'
    # # # dataset_file = os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl")
    #     with open(dataset_file, "wb") as f:
    #         pkl.dump(self.replay_buffer, f)
    #         print("Saved dataset to", dataset_file)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
