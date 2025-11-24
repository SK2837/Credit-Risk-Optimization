#!/usr/bin/env python
# coding: utf-8

# ## Optimizing Acceptance Threshold in Credit Scoring using Reinforcement Learning

# ### External libraries used

# In[ ]:


import numpy as np                         # to work with vectors and matrices
import pandas as pd                        # to work with data 
import datetime as dt                      # to work with dates
import joblib                              # to store python objects as files
import matplotlib.pyplot as plt            # to vizualize


# ### Custom classes used

# In[ ]:


# simulation environment of the micro-loan business
from simulation import SimulationEnv
# the reinforcement learning agent
from agent import Agent
# models used by the RL agent to interact with and learn from environment 
from model import FeatureTransformer, Model, EnvironmentModel
# policy that the RL agent follows when interacting with the environment
from policy import Policy
# manager to hide a couple hundreds rows of code and to keep the presentation neat
from manager import Manager


# ### Setting up the environment and initializing the RL agent

# In[ ]:


# micro-loan business simulation environment instance
env = SimulationEnv()
# feature transformer instance to convert numerous outputs of environment into simple numeric variables understood by the RL agent
ft = FeatureTransformer(env)
# value function model instance - the brain of the RL agent. Approximates value of each action in every state of environment
lr = 0.0001                               # learning rate defines how adaptive the value function is to new input
model = Model(env, ft, lr)
# environment model instance - the planning center of the agent. Predicts future environment states based on the current one
env_model = EnvironmentModel(env, lr)
# policy instance - includes different kinds of behaviors the agent can use to interact with the environment
policy = Policy(env)
# RL agent instance - the guy that uses all of the above in order to optimize whatever you need
eps = 1                                   # exploration rate defines how much randomness to use to explore the environment
gamma = 0.95                              # discounting rate defines how quick the agent forgets his previous experience
agent = Agent(env, model, env_model, policy, eps, gamma, gamma)
# manager instance - a class to manage the experiment
manager = Manager(agent)


# ### Setting up the RL experiment

# In[ ]:


# define train and test episode numbers
train_episodes = 100                        # number of train episodes, where agent learns the environment and value function
test_episodes = 10                          # number of test episodes in a row to evaluate the current agent
test_frequency = 10                         # frequency of testing to track the progress of the agent
distorted_episodes = 100                    # number of test episodes in a distorted environment to evaluate ability to adapt

# define variables to store the experiment history
name = 'baseline_final'                     # name of experiment
bookkeeping_directory = 'bookkeeping'       # directory to store history
bookkeeping_frequency = 10                  # frequency of storing

# initialize the experiment
manager.initExperiment(train_episodes = train_episodes, test_episodes = test_episodes, test_frequency = test_frequency, experiment_name = name, bookkeeping_directory = bookkeeping_directory, bookkeeping_frequency = bookkeeping_frequency)


# ### Initial agent

# In[ ]:


# run one episode with initial agent having no knowledge of environment
test_episode_progress = manager.runTestEpisode()


# In[ ]:



# visualize episode progress
manager.plotEpisode(test_episode_progress)


# ### First episode of learning

# In[ ]:


# run one episode with initial agent starting to learn from its interaction with the environment
episode_progress = manager.runTrainEpisode()


# In[ ]:



# visualize episode progress
manager.plotEpisode(episode_progress)


# In[ ]:



# visualize value function
manager.plot_q_values(episode = 0)
manager.progress


# ### 100 more episodes of training

# In[ ]:



# run 100 episodes of training
weekly_progress, progress = manager.train()


# In[ ]:



# visualize training progress
manager.plotRun(weekly_progress, progress)


# In[ ]:



# visualize value function
manager.plot_q_values(episode = 100)


# ### Introducing dynamics into the environment

# In[ ]:



# add distortions to the simulated environment: decrease in the average predicted probability to repay
distortions = {'e': 1, 
               'news_positives_score_bias': -2,
               'repeats_positives_score_bias': -1,
               'news_negatives_score_bias': 2,
               'repeats_negatives_score_bias': 1,
               'news_default_rate_bias': 0,
               'repeats_default_rate_bias': 0, 
               'late_payment_rate_bias': 0, 
               'ar_effect': 0}
env = SimulationEnv(distortions = distortions)

# adjust learning and discount rates enabling the agent to adapt more efficiently
lr = 0.001
gamma = 0.5
eps = 0.5
model = joblib.load('bookkeeping/' + name + '/episode_' + str(train_episodes) + '/model.pkl')
model.set_learning_rate(lr)

# pass adjustments to the current agent
agent.env = env
agent.model = model
agent.gamma1 = agent.gamma2 = gamma
agent.eps = eps

# pass adjusted agent to the manager
manager.agent = agent


# ### First distorted episode

# In[ ]:



# run one episode of distorted environment
distorted_episode_progress = manager.runDistortedEpisode()


# In[ ]:



# visualize episode progress
manager.plotEpisode(distorted_episode_progress)


# In[ ]:



# Visualize value function
manager.plot_q_values(episode = 'distorted')


# ### 100 more episodes of adaptation

# In[ ]:



# run 100 distorted episodes
distorted_weekly_progress, progress = manager.simulateDistortedEpisodes(distortions, lr, gamma)


# In[ ]:



# visualize RL agent's performance
manager.plotDistortedEpisodes(distorted_weekly_progress, progress)

