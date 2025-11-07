#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas

'''
This file containes the code used for training a turtlebot3 robot in autonomous navigation and obstacle avoidance behavior on gazebo using DQN

This file is part of the submission for ENPM690 Final Project 

The code used is majorly referenced from https://github.com/tomasvr/turtlebot3_drlnav

'''

import copy
import os
import sys
import time
import numpy as np
import torch
import xml.etree.ElementTree as ET
import numpy as np
Infinity = np.inf
import random
from collections import deque
from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import time
import io
import pickle
import torch.nn.functional as torchf
import torch.nn as nn
from abc import ABC
pg.setConfigOptions(antialias=False)

# Lazy import matplotlib to avoid NumPy 2.0 compatibility issues
_matplotlib = None
_plt = None
_MaxNLocator = None

def _import_matplotlib():
    global _matplotlib, _plt, _MaxNLocator
    if _matplotlib is None:
        try:
            # Try to import matplotlib with NumPy 2.0 compatibility
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                import matplotlib
                matplotlib.use('TkAgg', force=True)
                import matplotlib.pyplot as plt
                from matplotlib.ticker import MaxNLocator
                _matplotlib = matplotlib
                _plt = plt
                _MaxNLocator = MaxNLocator
        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"Warning: matplotlib import failed (NumPy 2.0 compatibility issue): {e}")
            print("Graph plotting will be disabled. Consider upgrading matplotlib or downgrading NumPy.")
            # Create dummy objects if matplotlib fails to import
            class DummyMatplotlib:
                pass
            _matplotlib = DummyMatplotlib()
            _plt = None
            _MaxNLocator = None
    return _matplotlib, _plt, _MaxNLocator
import torch.optim as optim

#Class to manage the storage of the model, weights, replay buffer and graph data
class DqnAgent(Node):
    def __init__(self, training , load_session="", load_episode=0):
        super().__init__('dqn' + '_agent') 
        try:
            with open('/tmp/drlnav_current_stage.txt', 'r') as f:
                self.stage = int(f.read())
        except FileNotFoundError:
            print("Launch the gazebo simulation first")
            
        #Get the simulation speed from the world file
        tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage' + str(self.stage) + '/burger.model')
        root = tree.getroot()
        self.sim_speed = int(root.find('world').find('physics').find('real_time_factor').text)
        
        #Initialise the training and testing flags
        self.training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        #Check if the model is to be loaded for testing
        if (not self.training and not self.load_session):
            quit("\033[1m" + "\033[93m" + "Invalid command: Testing but no model to load specified, see readme for correct format" + "\033[0m}")
        print("GPU available: ", torch.cuda.is_available())
        if (torch.cuda.is_available()):
            print("Name: ", torch.cuda.get_device_name(0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"{'training' if (self.training) else 'testing' } on stage: {self.stage}")
        
        #Initialise the total steps
        self.total_steps = 0
        
        #Initialise the observation steps, this is the number of steps the agent will take before training starts
        self.observe_steps = 25000
        
        #Initialise the model, replay buffer and graph
        self.model = DQN(self.device, self.sim_speed)
        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()
        
        #Initialise the visualisation and stacking flags, and the algorithm
        self.enable_visual = False
        self.enable_stacking = False
        self.algorithm = 'dqn'
        
        #Defining the action set for the mobile robot, the action set is a combination of linear and angular velocities
        self.possible_actions = [[0.0,-0.5],[0.6, -1.0], [0.6, -0.5], [0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.5, 0.0], [0.3, 0.5], [0.3, 1.0], [0.6, 0.5], [0.6, 1.0],[0.0,0.5]]
        
        #Initialise the replay buffer memory
        self.sm = StorageManager(self.algorithm, self.load_session, self.episode, self.device, self.stage)
        
        #Initialise the logger, visualisation and clients for communication with the environment
        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            if self.training:
                self.replay_buffer.buffer = self.sm.load_replay_buffer(self.model.buffer_size, os.path.join(self.load_session, 'stage'+str(self.sm.stage)+'_latest_buffer.pkl'))
            self.total_steps = self.graph.set_graphdata(self.sm.load_graphdata(), self.episode)
            print(f"global steps: {self.total_steps}")
            print(f"loaded model {self.load_session} (eps {self.episode}): {self.model.get_model_parameters()}")
        else:
            self.sm.new_session_dir(self.stage)
            self.sm.store_model(self.model)
        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(self.training, self.sm.machine_dir, self.sm.session_dir, self.sm.session, self.model.get_model_parameters(), self.model.get_model_configuration(), str(self.stage), self.algorithm, self.episode)
        if self.enable_visual:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)
        #Initialise the clients for communication with the environment
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        self.gazebo_pause = self.create_client(Empty, '/pause_physics')
        self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')
        self.process()

    #Function to start the training process, this function will pause the simulation, wait for a new goal, initialise the episode and start the training process
    def process(self):
        self.pause_simulation()
        while (True):
            self.wait_new_goal()
            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = self.init_episode()

            if self.enable_stacking:
                frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
                state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)
                next_state = [0.0] * (self.model.state_size * self.model.stack_depth)

            self.unpause_simulation()
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                if self.training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, self.training, step, self.enable_visual)

                action_current = self.model.possible_actions[action]

                # Take a step
                next_state, reward, episode_done, outcome, distance_traveled = self.step(action_current, action_past)
                action_past = copy.deepcopy(action_current)
                reward_sum += reward

                if self.enable_stacking:   
                    frame_buffer = frame_buffer[self.model.state_size:] + list(next_state)      # Update big buffer with single step
                    next_state = []                                                         # Prepare next set of frames (state)
                    for depth in range(self.model.stack_depth):
                        start = self.model.state_size * (self.model.frame_skip - 1) + (self.model.state_size * self.model.frame_skip * depth)
                        next_state += frame_buffer[start : start + self.model.state_size]

                # Train
                if self.training == True:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        loss_c, loss_a, = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a

                # Update state, step and time for next iteration
                if self.enable_visual:  
                    self.visual.update_reward(reward_sum)
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            # Finish episode, save model, update graph and log
            self.pause_simulation()
            self.total_steps += step
            duration = time.perf_counter() - episode_start
            self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor)
    
    #Function to pause the simulation
    def pause_simulation(self):
        while not self.gazebo_pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pause gazebo service not available, waiting again...')
        future = self.gazebo_pause.call_async(Empty.Request())
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                return
    #This function waits for a new goal to be set in the environment        
    def wait_new_goal(self):
        while(self.get_goal_status() == False):
            print("Waiting for new goal")
            time.sleep(1.0)
    #Function to get the status of the goal in the environment   
    def get_goal_status(self):
        req = Goal.Request()
        while not self.goal_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('new goal service not available, waiting again...')
        future = self.goal_comm_client.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return res.new_goal
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("Goal Service Error")
    #Function to initialise the episode in the environment
    def init_episode(self):
        state, _, _, _, _ = self.step([], [0.0, 0.0])
        return state
    #Function to take a step in the environment, this function sends the action to the environment and gets the next state, reward, done status, success status and distance traveled
    def step(self, action, previous_action):
        req = DrlStep.Request()
        req.action = action
        req.previous_action = previous_action

        while not self.step_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env step service not available, waiting again...')
        future = self.step_comm_client.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return res.state, res.reward, res.done, res.success, res.distance_traveled
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("Step Service Error")
    #Function to unpause the simulation                
    def unpause_simulation(self):
        while not self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('unpause gazebo service not available, waiting again...')
        future = self.gazebo_unpause.call_async(Empty.Request())
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                return
    #Function to finish the episode, this function saves the model, updates the graph and logs the outcomes of the episode
    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, lost_actor):
            if self.total_steps < self.observe_steps:
                print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
                return

            self.episode += 1
            print(f"Epi: {self.episode:<5}R: {reward_sum:<8.0f}outcome: {self.translate_outcome(outcome):<13}", end='')
            print(f"steps: {step:<6}steps_total: {self.total_steps:<7}time: {eps_duration:<6.2f}")

            if (not self.training):
                self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
                return

            self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, lost_actor)
            self.logger.file_log.write(f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, {self.total_steps}, \
                                            {self.replay_buffer.get_length()}, {loss_critic / step}, {lost_actor / step}\n")

            if (self.episode % 100 == 0) or (self.episode == 1):
                self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
                self.logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
            if (self.episode % 10 == 0) or (self.episode == 1):
                self.graph.draw_plots(self.episode)
    #Function to translate the outcome of the episode            
    def translate_outcome(self,outcome):
        if outcome == 1:
            return "SUCCESS"
        elif outcome == 2:
            return "COLL_WALL"
        elif outcome == 3:
            return "COLL_OBST"
        elif outcome == 4:
            return "TIMEOUT"
        elif outcome == 5:
            return "TUMBLE"
        else:
            return f"UNKNOWN: {outcome}"
#Actor Class to define the Actor Network, the architecture of the network is defined in this class
class Actor(nn.Module, ABC):
    def __init__(self, name, state_size, action_size, hidden_size, visual=None):
        super(Actor, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

        # Define the layers of the Neural Network, this is a Fully Connected Neural Network
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)

        # Initialize weights of the Neural Network
        self.apply(self.init_weights)
    
    #Function to forward propagate the input through the network
    def forward(self, states, visualize=False):
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))
        action = self.fa3(x2)

        if visualize and self.visual:
            action_index = action.argmax().item()
            action_tensor = torch.from_numpy(np.asarray(self.possible_actions[action_index], np.float32))
            self.visual.update_layers(states, action_tensor, [x1, x2], [self.fa1.bias, self.fa2.bias])

        return action
    #Function to initialise the weights of the network, Xavier Uniform Initialisation is used as the weights initialisation technique. The bias is initialised to 0.01
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
#DQN class to define the Deep Q Network
class DQN(ABC):
    def __init__(self, device, sim_speed):
        #Initialise the device and simulation speed
        self.device = device
        self.simulation_speed = sim_speed
        #Get the number of scan samples from the model.sdf file
        tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf')
        root = tree.getroot()
        for link in root.find('model').findall('link'):
            if link.get('name') == 'base_scan':
                NUM_SCAN_SAMPLES = int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)
                
        # Define the parameters of the DQN model
        self.state_size         = NUM_SCAN_SAMPLES + 4
        self.action_size        = 12
        self.hidden_size        = 1024
        self.input_size         = self.state_size
        self.batch_size         = 128
        self.buffer_size        = 1000000
        self.discount_factor    = 0.99
        self.learning_rate      = 0.003
        self.tau                = 0.003
        self.step_time          = 0.01
        self.loss_function      = torchf.smooth_l1_loss
        self.epsilon            = 1.0
        self.epsilon_decay      = 0.9995
        self.epsilon_minimum    = 0.075
        self.backward_enabled   = True
        self.stacking_enabled   = False
        self.stack_depth        = 3
        self.frame_skip         = 4
        
        #Check if stacking is enabled and update the input size
        if self.stacking_enabled:
            self.input_size *= self.stack_depth
        self.networks = []
        self.iteration = 0
        
        #Define the possible actions for the mobile robot, the action set is a combination of linear and angular velocities
        self.possible_actions = [[0.0,-0.5],[0.6, -1.0], [0.6, -0.5], [0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.5, 0.0], [0.3, 0.5], [0.3, 1.0], [0.6, 0.5], [0.6, 1.0],[0.0,0.5]]
        #Initialise the target update frequency for hard update of the target network
        self.target_update_frequency = 500
        #Call the initialise function to create the network, optimiser, target network and target optimiser
        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)
        self.hard_update(self.actor_target, self.actor)
    #Function to get the action, this function returns the action to be taken by the agent
    def get_action(self, state, is_training, step=0, visualize=False):
        if is_training and np.random.random() < self.epsilon:
            return self.get_action_random()
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        Q_values = self.actor(state, visualize).detach().cpu()
        action = Q_values.argmax().tolist()
        return action
    #Function to get a random action, this function selects a random action from the action set
    def get_action_random(self):
        return np.random.randint(0, self.action_size)
    
    #Function to train the network, this function trains the network using the replay buffer
    def train(self, state, action, reward, state_next, done):
        #Get the Q values for the current state and the next state, this is done using the actor network
        action = torch.unsqueeze(action, 1)
        Q_next = self.actor_target(state_next).amax(1, keepdim=True)
        Q_target = reward + (self.discount_factor * Q_next * (1 - done))
        Q = self.actor(state).gather(1, action.long())
        loss = torchf.mse_loss(Q, Q_target)
        
        #Backward pass, this is done to update the weights of the network
        self.actor_optimizer.zero_grad()
        loss.backward()
        
        #Clip the gradients, this is done to prevent the gradients from exploding
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0, norm_type=1)
        
        #Update the weights of the network, this is done using the Adam Optimiser
        self.actor_optimizer.step()
        
        #When the iteration is a multiple of the target update frequency, a hard update is done to update the target network, otherwise a soft update is done for every 10 iterations
        if self.iteration % self.target_update_frequency == 0:
            self.hard_update(self.actor_target, self.actor)
        elif self.iteration % 10 == 0:
            self.soft_update(self.actor_target,self.actor,self.tau)
        return 0, loss.mean().detach().cpu()
    #Function to train the network, this function trains the network using the replay buffer
    def _train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        sample_s, sample_a, sample_r, sample_ns, sample_d = batch
        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)
        
        #This function trains the network using the replay buffer
        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d)
        self.iteration += 1
        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        return result
    #Function to create the network
    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network
    #Function to create the optimiser, Adam Optimiser is used for training the network
    def create_optimizer(self, network):
        return torch.optim.AdamW(network.parameters(), self.learning_rate)
    #Function to update the target network using hard update
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    #Function to update the target network using soft update
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    #Function to get the saved model parameters
    def get_model_configuration(self):
        configuration = ""
        for attribute, value in self.__dict__.items():
            if attribute not in ['actor', 'actor_target', 'critic', 'critic_target']:
                configuration += f"{attribute} = {value}\n"
        return configuration
    # Function to get the model parameters
    def get_model_parameters(self):
        parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size,
                            self.discount_factor, self.learning_rate, self.tau, self.step_time ,
                            self.backward_enabled, self.stacking_enabled, self.stack_depth, self.frame_skip]
        parameter_string = ', '.join(map(str, parameters))
        return parameter_string
    #Function to attach the visualisation to the network
    def attach_visual(self, visual):
        self.actor.visual = visual
#Class for Replay Buffer, this class is used to store the experiences of the agent
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size
    #Function to sample the replay buffer
    def sample(self, batchsize):
        batch = []
        batchsize = min(batchsize, self.get_length())
        batch = random.sample(self.buffer, batchsize)
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array
    #Function to get the length of the replay buffer
    def get_length(self):
        return len(self.buffer)
    #Function to add a sample to the replay buffer
    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.buffer.append(transition)
#Class to Log the outcomes of the episodes   
class Logger():
    def __init__(self, training, machine_dir, session_dir, session, hyperparameters, model_config, stage, algorithm, load_episode):
        self.test_entry = 0
        self.test_outcome = [0] * 6
        self.test_distance = []
        self.test_duration = []
        self.test_swerving = []
        self.is_training = training

        self.session = session
        self.hyperparameters = hyperparameters
        self.model_config = model_config
        self.stage = stage
        self.algorithm = algorithm

        self.highest_reward = -Infinity
        self.best_episode_reward = 0
        self.highest_success = 0
        self.best_episode_success = 0

        datetime = time.strftime("%Y%m%d-%H%M%S")
        self.file_comparison = self.init_comparison_file(datetime, machine_dir, stage, hyperparameters, algorithm, session, load_episode)
        if self.is_training:
            self.file_log = self.init_training_log(datetime, session_dir, stage, model_config)
        else:
            self.file_log = self.init_testing_log(datetime, session_dir, stage, load_episode)

    #Function to update the test results
    def update_test_results(self, step, outcome, distance_traveled, episode_duration, swerving_sum):
        self.test_entry += 1
        self.test_outcome[outcome] += 1
        if outcome == 1:
            self.test_distance.append(distance_traveled)
            self.test_duration.append(episode_duration)
            self.test_swerving.append(swerving_sum/step)
        success_count = self.test_outcome[1]

        self.file_log.write(f"{self.test_entry}, {outcome}, {step}, {episode_duration}, {distance_traveled}, {self.test_outcome[1]}/{self.test_outcome[2]}/{self.test_outcome[3]}/{self.test_outcome[4]}/{self.test_outcome[5]}\n")
        if self.test_entry > 0 and self.test_entry % 100 == 0:
            self.update_comparison_file(self.test_entry, self.test_outcome[1] / (self.test_entry / 100), 0)
            self.file_log.write(f"Successes: {self.test_outcome[1]} ({self.test_outcome[1]/self.test_entry:.2%}), "
            f"collision (wall): {self.test_outcome[2]} ({self.test_outcome[2]/self.test_entry:.2%}), "
            f"collision (obs): {self.test_outcome[3]} ({self.test_outcome[3]/self.test_entry:.2%}), "
            f"timeouts: {self.test_outcome[4]}, ({self.test_outcome[4]/self.test_entry:.2%}), "
            f"tumbles: {self.test_outcome[5]}, ({self.test_outcome[5]/self.test_entry:.2%}), ")
            if success_count > 0:
                self.file_log.write(f"distance: {sum(self.test_distance)/success_count:.3f}, "
                                    f"swerving: {sum(self.test_swerving)/success_count:.3f}, "
                                    f"duration: {sum(self.test_duration)/success_count:.3f}\n")
        if self.test_entry > 0:
            print(f"Successes: {self.test_outcome[1]} ({self.test_outcome[1]/self.test_entry:.2%}), "
            f"collision (wall): {self.test_outcome[2]} ({self.test_outcome[2]/self.test_entry:.2%}), "
            f"collision (obs): {self.test_outcome[3]} ({self.test_outcome[3]/self.test_entry:.2%}), "
            f"timeouts: {self.test_outcome[4]}, ({self.test_outcome[4]/self.test_entry:.2%}), "
            f"tumbles: {self.test_outcome[5]}, ({self.test_outcome[5]/self.test_entry:.2%}), ")
            if success_count > 0:
                print(f"distance: {sum(self.test_distance)/success_count:.3f}, "
                      f"swerving: {sum(self.test_swerving)/success_count:.3f}, "
                      f"duration: {sum(self.test_duration)/success_count:.3f}")

    #Function to initialise the training log
    def init_training_log(self, datetime, path, stage, model_config):
        file_log = open(os.path.join(path, "_train_stage" + stage + "_" + datetime + '.txt'), 'w+')
        file_log.write("episode, reward, success, duration, steps, total_steps, memory length, avg_critic_loss, avg_actor_loss\n")
        with open(os.path.join(path, '_model_configuration_' + datetime + '.txt'), 'w+') as file_model_config:
            file_model_config.write(model_config + '\n')
        return file_log
    #Function to initialise the testing log
    def init_testing_log(self, datetime, path, stage, load_episode):
        file_log = open(os.path.join(path, "_test_stage" + stage + "_eps" + str(load_episode) + "_" + datetime + '.txt'), 'w+')
        file_log.write(f"episode, outcome, step, episode_duration, distance, s/cw/co/t\n")
        return file_log
    #Function to initialise the comparison file
    def init_comparison_file(self, datetime, path, stage, hyperparameters, algorithm, session, episode):
        prefix = "_training" if self.is_training else "_testing"
        with open(os.path.join(path, "__" + algorithm + prefix + "_comparison.txt"), 'a+') as file_comparison:
            file_comparison.write(datetime + ', ' + session + ', ' + str(episode) + ', ' + stage + ', ' + hyperparameters + '\n')
        return file_comparison
    #Function to update the comparison file
    def update_comparison_file(self, episode, success_count, average_reward=0):
        if average_reward > self.highest_reward and episode != 1:
            self.highest_reward = average_reward
            self.best_episode_reward = episode
        if success_count > self.highest_success and episode != 1:
            self.highest_success = success_count
            self.best_episode_success = episode
        datetime = time.strftime("%Y%m%d-%H%M%S")
        with open(self.file_comparison.name, 'a+') as file_comparison:
            file_comparison.seek(0)
            lines = file_comparison.readlines()
            file_comparison.seek(0)
            file_comparison.truncate()
            file_comparison.writelines(lines[:-1])
            file_comparison.write(datetime + ', ' + self.session + ', ' + self.stage + ', ' + self.hyperparameters)
            if self.is_training:
                file_comparison.write(', results, ' + str(episode) + ', ' + str(self.best_episode_success) + ': ' + str(self.highest_success) + '%, ' + str(self.best_episode_reward) + ': ' + str(self.highest_reward) + '\n')
            else:
                file_comparison.write(', results, ' + str(episode) + ', ' + str(self.best_episode_success) + ', ' + str(self.highest_success) + '%\n')
#Class to define the visualisation of the network
class DrlVisual(QtWidgets.QWidget):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.setWindowTitle('DQN Visualization')
        self.show()
        self.resize(1980, 1200)

        self.state_size = state_size
        self.hidden_sizes = [hidden_size, hidden_size]

        # Use GraphicsLayoutWidget instead of GraphicsWindow (deprecated in newer pyqtgraph)
        try:
            # Try to use GraphicsLayoutWidget (newer pyqtgraph API)
            if hasattr(pg, 'GraphicsLayoutWidget'):
                self.glw = pg.GraphicsLayoutWidget()
            elif hasattr(pg, 'GraphicsWindow'):
                # Fallback for older versions
                self.glw = pg.GraphicsWindow()
            else:
                raise AttributeError("Neither GraphicsLayoutWidget nor GraphicsWindow available")
        except (AttributeError, Exception) as e:
            print(f"Warning: Failed to initialize pyqtgraph visualization: {e}")
            print("Visualization will be disabled. Please upgrade pyqtgraph or use compatible version.")
            # Create a dummy widget to prevent crashes
            self.glw = QtWidgets.QWidget()
            self.visualization_available = False
            return
        
        self.visualization_available = True
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.glw)

        # States
        self.plot_item_states = self.glw.addPlot(title="States", colspan=3)
        self.plot_item_states.setXRange(-1, self.state_size, padding=0)
        self.plot_item_states.setYRange(-1, 1, padding=0)

        self.bar_graph_states = pg.BarGraphItem(x=range(self.state_size), width=1)
        self.plot_item_states.addItem(self.bar_graph_states)

        self.hidden_plot_items = []
        self.hidden_bar_graphs = []
        self.hidden_line_plots = []
        i = 0
        for hidden_size in self.hidden_sizes:
            self.glw.nextRow()
            plot_item = self.glw.addPlot(title=f"Hidden layer {i}", colspan=3)
            plot_item.setXRange(-1, hidden_size, padding=0)
            plot_item.setYRange(-0.2, 1.3, padding=0)

            bar_graph = pg.BarGraphItem(x=range(hidden_size), width=0.8)
            plot_item.addItem(bar_graph)

            line_plot = plot_item.plot(x=range(hidden_size), brush='r', symbol='x', symbolPen='r')
            line_plot.setPen(style=QtCore.Qt.NoPen)
            line_plot.setSymbolSize(5)

            self.hidden_bar_graphs.append(bar_graph)
            self.hidden_plot_items.append(plot_item)
            self.hidden_line_plots.append(line_plot)
            i += 1

        # Output layers
        self.glw.nextRow()
        self.plot_item_action_linear = self.glw.addPlot(title="Action Linear")
        self.plot_item_action_linear.setXRange(-20, 20, padding=0)
        self.plot_item_action_linear.setYRange(-1, 1, padding=0)
        self.bar_graph_action_linear = pg.BarGraphItem(x=range(1), width=0.5)
        self.plot_item_action_linear.addItem(self.bar_graph_action_linear)
        self.plot_item_action_angular = self.glw.addPlot(title="Action Angular")
        self.plot_item_action_angular.setXRange(-1, 1, padding=0)
        self.plot_item_action_angular.setYRange(-1.5, 1.5, padding=0)
        self.bar_graph_action_angular = pg.BarGraphItem(x=range(1), width=0.5)
        self.plot_item_action_angular.addItem(self.bar_graph_action_angular)
        self.bar_graph_action_angular.rotate(90)
        self.plot_item_reward = self.glw.addPlot(title="Accumlated Reward")
        self.plot_item_reward.setXRange(-1, 1, padding=0)
        self.plot_item_reward.setYRange(-3000, 5000, padding=0)
        self.bar_graph_reward = pg.BarGraphItem(x=range(1), width=0.5)
        self.plot_item_reward.addItem(self.bar_graph_reward)

        self.iteration = 0
    #Function to prepare the data for visualisation
    def prepare_data(self, tensor):
        return tensor.squeeze().flip(0).detach().cpu()
    #Function to update the layers of the network
    def update_layers(self, states, actions, hidden, biases):
        if not hasattr(self, 'visualization_available') or not self.visualization_available:
            return  # Skip if visualization is not available
        self.bar_graph_states.setOpts(height=self.prepare_data(states))
        actions = actions.detach().cpu().numpy().tolist()
        self.bar_graph_action_linear.setOpts(height=[actions[0]])
        self.bar_graph_action_angular.setOpts(height=[actions[1]])
        for i in range(len(hidden)):
            self.hidden_bar_graphs[i].setOpts(height=self.prepare_data(hidden[i]))
        pg.QtGui.QApplication.processEvents()
        if self.iteration % 100 == 0:
            self.update_bias(biases)
        self.iteration += 1
    #Function to update the bias of the network
    def update_bias(self, biases):
        for i in range(len(biases)):
            self.hidden_line_plots[i].setData(y=self.prepare_data(biases[i]))
    #Function to update the reward
    def update_reward(self, acc_reward):
        self.bar_graph_reward.setOpts(height=[acc_reward])
        if acc_reward > 0:
            self.bar_graph_reward.setOpts(brush='g')
        else:
            self.bar_graph_reward.setOpts(brush='r')
#Class to define the graph plotting to show the outcomes of the episodes 
class Graph():
    def __init__(self):
        matplotlib, plt, MaxNLocator = _import_matplotlib()
        if plt is None:
            print("Warning: matplotlib not available, graph plotting will be disabled")
            self.plt_available = False
            self.fig = None
            self.ax = None
        else:
            self.plt_available = True
            self.plt = plt
            self.MaxNLocator = MaxNLocator
            plt.show()
            self.fig, self.ax = plt.subplots(2, 2)
            self.fig.set_size_inches(18.5, 10.5)

            titles = ['outcomes', 'avg critic loss over episode', 'avg actor loss over episode', 'avg reward over 10 episodes']
            for i in range(4):
                ax = self.ax[int(i/2)][int(i%2!=0)]
                ax.set_title(titles[i])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.GRAPH_AVERAGE_REWARD = 10
        self.GRAPH_DRAW_INTERVAL =10
        self.session_dir = ""
        self.legend_labels = ['Unknown', 'Success', 'Collision Wall', 'Collision Dynamic', 'Timeout', 'Tumble']
        self.legend_colors = ['b', 'g', 'r', 'c', 'm', 'y']

        self.outcome_histories = []

        self.global_steps = 0
        self.data_outcome_history = []
        self.data_rewards = []
        self.data_loss_critic = []
        self.data_loss_actor = []
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
        self.legend_set = False
    #This function sets the graph data
    def set_graphdata(self, graphdata, episode):
        self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor = [graphdata[i] for i in range(len(self.graphdata))]
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
        self.draw_plots(episode)
        return self.global_steps
    #Function to update the data for the graph
    def update_data(self, step, global_steps, outcome, reward_sum, loss_critic_sum, loss_actor_sum):
        self.global_steps = global_steps
        self.data_outcome_history.append(outcome)
        self.data_rewards.append(reward_sum)
        self.data_loss_critic.append(loss_critic_sum / step)
        self.data_loss_actor.append(loss_actor_sum / step)
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
    #Function to draw the plots
    def draw_plots(self, episode):
        if not self.plt_available:
            return  # Skip plotting if matplotlib is not available
        
        xaxis = np.array(range(1, episode + 1))

        # Plot outcome history
        for idx in range(len(self.data_outcome_history)):
            if idx == 0:
                self.outcome_histories = [[0],[0],[0],[0],[0],[0]]
                self.outcome_histories[self.data_outcome_history[0]][0] += 1
            else:
                for outcome_history in self.outcome_histories:
                    outcome_history.append(outcome_history[-1])
                self.outcome_histories[self.data_outcome_history[idx]][-1] += 1

        if len(self.data_outcome_history) > 0:
            i = 0
            for outcome_history in self.outcome_histories:
                self.ax[0][0].plot(xaxis, outcome_history, color=self.legend_colors[i], label=self.legend_labels[i])
                i += 1
            if not self.legend_set:
                self.ax[0][0].legend()
                self.legend_set = True

        # Plot critic loss
        y = np.array(self.data_loss_critic)
        self.ax[0][1].plot(xaxis, y)

        # Plot actor loss
        y = np.array(self.data_loss_actor)
        self.ax[1][0].plot(xaxis, y)

        # Plot average reward
        count = int(episode / self.GRAPH_AVERAGE_REWARD)
        if count > 0:
            xaxis = np.array(range(self.GRAPH_AVERAGE_REWARD, episode+1, self.GRAPH_AVERAGE_REWARD))
            averages = list()
            for i in range(count):
                avg_sum = 0
                for j in range(self.GRAPH_AVERAGE_REWARD):
                    avg_sum += self.data_rewards[i * self.GRAPH_AVERAGE_REWARD + j]
                averages.append(avg_sum / self.GRAPH_AVERAGE_REWARD)
            y = np.array(averages)
            self.ax[1][1].plot(xaxis, y)

        self.plt.draw()
        self.plt.pause(0.2)
        self.plt.savefig(os.path.join(self.session_dir, "_figure.png"))
    #Function to get the success count
    def get_success_count(self):
        suc = self.data_outcome_history[-self.GRAPH_DRAW_INTERVAL:]
        return suc.count(1)
    #Function to get the average reward per episode
    def get_reward_average(self):
        rew = self.data_rewards[-self.GRAPH_DRAW_INTERVAL:]
        return sum(rew) / len(rew)
#Class to define the storage manager for the model      
class StorageManager:
    def __init__(self, name, load_session, load_episode, device, stage):
        if load_session and name not in load_session:
            print(f"ERROR: wrong combination of command and model! make sure command is: {name}_agent")
            while True:
                pass
        self.machine_dir = (os.getenv('DRLNAV_BASE_PATH') + '/src/model/enpm690_dqn')
        self.name = name
        self.stage = load_session[-1] if load_session else stage
        self.session = load_session
        self.load_episode = load_episode
        self.session_dir = os.path.join(self.machine_dir, self.session)
        self.map_location = device
    #Function to create a new session directory
    def new_session_dir(self, stage):
        i = 0
        session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}_stage_{stage}")
        while(os.path.exists(session_dir)):
            i += 1
            session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}_stage_{stage}")
        self.session = f"{self.name}_{i}"
        print(f"making new model dir: {session_dir}")
        os.makedirs(session_dir)
        self.session = self.session
        self.session_dir = session_dir
    #Function to delete a file
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
    #Function to save the weights of the network
    def network_save_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
        print(f"saving {network.name} model for episode: {episode}")
        torch.save(network.state_dict(), filepath)
    #Function to save the session
    def save_session(self, episode, networks, pickle_data, replay_buffer):
        print(f"saving data for episode: {episode}, location: {self.session_dir}")
        for network in networks:
            self.network_save_weights(network, self.session_dir, self.stage, episode)

        # Store graph data
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        # Store latest buffer (can become very large, multiple gigabytes)
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_latest_buffer.pkl'), 'wb') as f:
            pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)

        # Delete previous iterations (except every 1000th episode)
        if (episode % 1000 == 0):
            for i in range(episode, episode - 1000, 100):
                for network in networks:
                    self.delete_file(os.path.join(self.session_dir, network.name + '_stage'+str(self.stage)+'_episode'+str(i)+'.pt'))
                self.delete_file(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(i)+'.pkl'))
    #Function to store the model
    def store_model(self, model):
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_agent.pkl'), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    #Function to store the replay buffer
    def network_load_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
        print(f"loading: {network.name} model from file: {filepath}")
        network.load_state_dict(torch.load(filepath, self.map_location))
    #Function to load the graph data
    def load_graphdata(self):
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.pkl'), 'rb') as f:
            return pickle.load(f)
    #Function to load the replay buffer
    def load_replay_buffer(self, size, buffer_path):
        buffer_path = os.path.join(self.machine_dir, buffer_path)
        if (os.path.exists(buffer_path)):
            with open(buffer_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"buffer does not exist: {buffer_path}")
            return deque(maxlen=size)
    #Function to load the model
    def load_model(self):
        model_path = os.path.join(self.session_dir, 'stage'+str(self.stage)+'_agent.pkl')
        try :
            with open(model_path, 'rb') as f:
                return CpuUnpickler(f, self.map_location).load()
        except FileNotFoundError:
            quit(f"The specified model: {model_path} was not found. Check whether you specified the correct stage {self.stage} and model name")
    #Function to load the weights of the network
    def load_weights(self, networks):
        for network in networks:
            self.network_load_weights(network, self.session_dir, self.stage, self.load_episode)

class CpuUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        self.map_location = map_location
        super(CpuUnpickler, self).__init__(file)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)
#Main function to create a DQN Agent
def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DqnAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()
#Function to set the code run to training
def main_train(args=sys.argv[1:]):
    args = ['1'] + args
    main(args)
#Function to set the code run to testing
def main_test(args=sys.argv[1:]):
    args = ['0'] + args
    main(args)
    
#Initialise Main Function
if __name__ == '__main__':
    main()