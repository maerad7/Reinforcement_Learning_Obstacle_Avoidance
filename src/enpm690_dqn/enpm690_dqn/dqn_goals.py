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

import os
import random
import math
import numpy
import time
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from turtlebot3_msgs.srv import RingGoal
import xml.etree.ElementTree as ET

class DQNGoals(Node):
    def __init__(self):
        super().__init__('dqn_goals')
        
        ## Getting the path to goal box and opening it
        self.entity_dir_path = (os.path.dirname(os.path.realpath(__file__))).replace(
            'enpm690_dqn/lib/python3.10/site-packages/enpm690_dqn',
            'turtlebot3_gazebo/share/turtlebot3_gazebo/models/turtlebot3_drl_world/goal_box')
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'

        ## Initializing the parameters used in the code
        self.prev_x, self.prev_y = -1, -1
        self.goal_x, self.goal_y = 0.5, 0.0
        self.no_goal_spawn_margin = 0.3 # meters away from any wall
        self.enable_random_goals = False
        self.arena_length = 4.2
        self.arena_width = 4.2
        self.enable_dynamic_goals = True
        
        with open('/tmp/drlnav_current_stage.txt', 'r') as f:
            self.stage = int(f.read())
        print(f"Stage: {self.stage}, dynamic goals enabled: {self.enable_dynamic_goals}")
        
        # Initialising publishers, clients and services
        self.goal_pose_pub = self.create_publisher(Pose, 'goal_pose', QoSProfile(depth=10))
        self.delete_entity_client       = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client        = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client    = self.create_client(Empty, 'reset_simulation')
        self.gazebo_pause               = self.create_client(Empty, '/pause_physics')
        self.task_succeed_server    = self.create_service(RingGoal, 'task_succeed', self.task_succeed_callback)
        self.task_fail_server       = self.create_service(RingGoal, 'task_fail', self.task_fail_callback)

        self.obstacle_coordinates   = self.get_obstacle_coordinates()
        self.delete_entity()
        self.reset_simulation()
        self.publish_callback()
        print("Goal:", self.goal_x, self.goal_y)
        time.sleep(1)

    ## Function to publish the goal values
    def publish_callback(self):
        # Publish goal pose
        goal_pose = Pose()
        goal_pose.position.x = self.goal_x
        goal_pose.position.y = self.goal_y
        self.goal_pose_pub.publish(goal_pose)
        self.spawn_entity()

    ## If the episode is successful, this function is called to reset the simulation and generate a new goal
    def task_succeed_callback(self, request, response):
        self.delete_entity()
        if self.enable_random_goals:
            self.generate_random_goal()
            print(f"Task Successful: Reached Goal: generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}")
        elif self.enable_dynamic_goals:
            self.generate_dynamic_goal_pose(request.robot_pose_x, request.robot_pose_y, request.radius)
            print(f"Task Successful: Reached Goal: generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}, radius: {request.radius:.2f}")
        else:
            self.generate_goal_pose()
            print(f"Task Successful: Reached Goal: generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}")
        return response
    
    ## If the episode is failure, this function is called to reset the simulation and generate a new goal
    def task_fail_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        if self.enable_random_goals:
            self.generate_random_goal()
            print(f"Task Failed: Didn't Reach Goal: Resetting the environment and generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}")
        elif self.enable_dynamic_goals:
            self.generate_dynamic_goal_pose(request.robot_pose_x, request.robot_pose_y, request.radius)
            print(f"Task Failed: Didn't Reach Goal: Resetting the environment and generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}, radius: {request.radius:.2f}")
        else:
            self.generate_goal_pose()
            print(f"Task Failed: Didn't Reach Goal: Resetting the environment and generating a new goal: {self.goal_x:.2f}, {self.goal_y:.2f}")
        return response

    ## Check the validity of the new node generated
    def goal_is_valid(self, goal_x, goal_y):
        if goal_x > self.arena_length/2 or goal_x < -self.arena_length/2 or goal_y > self.arena_width/2 or goal_y < -self.arena_width/2:
            return False
        for obstacle in self.obstacle_coordinates:
            if goal_x < obstacle[0][0] and goal_x > obstacle[2][0]:
                if goal_y < obstacle[0][1] and goal_y > obstacle[2][1]:
                    return False
        return True

    ## Function to genarate random goals
    def generate_random_goal(self):
        self.prev_x = self.goal_x
        self.prev_y = self.goal_y
        tries = 0
        while (((abs(self.prev_x - self.goal_x) + abs(self.prev_y - self.goal_y)) < 4) or (not self.goal_is_valid(self.goal_x, self.goal_y))):
            self.goal_x = random.randrange(-25, 25) / 10.0
            self.goal_y = random.randrange(-25, 25) / 10.0
            tries += 1
            if tries > 200:
                print("Cannot find valid new goal, resestting!")
                self.delete_entity()
                self.reset_simulation()
                self.generate_goal_pose()
                break
        self.publish_callback()

    ## Function to genarate dynamic goals
    def generate_dynamic_goal_pose(self, robot_pose_x, robot_pose_y, radius):
        tries = 0
        while(True):
            ring_position = random.uniform(0, 1)
            origin = radius + numpy.random.normal(0, 0.1) # in meters
            goal_offset_x = math.cos(2 * math.pi * ring_position) * origin
            goal_offset_y = math.sin(2 * math.pi * ring_position) * origin
            goal_x = robot_pose_x + goal_offset_x
            goal_y = robot_pose_y + goal_offset_y
            if self.goal_is_valid(goal_x, goal_y):
                self.goal_x = goal_x
                self.goal_y = goal_y
                break
            if tries > 100:
                print("Cannot find valid new goal, resestting!")
                self.delete_entity()
                self.reset_simulation()
                self.generate_goal_pose()
                return
            tries += 1
        self.publish_callback()

    ## Function to select from a predefined goal values
    def generate_goal_pose(self):   
        self.prev_x = self.goal_x
        self.prev_y = self.goal_y
        tries = 0

        while ((abs(self.prev_x - self.goal_x) + abs(self.prev_y - self.goal_y)) < 2):
            if self.stage == 11:
                # --- Define static goal positions here ---
                goal_pose_list = [[0.0, 0.0], [0.0, 6.5], [5.0, 5.5], [-2.5, -6.0], [3.0, -4.0], [6.0, -1.0]]
                index = random.randrange(0, len(goal_pose_list))
                self.goal_x = float(goal_pose_list[index][0])
                self.goal_y = float(goal_pose_list[index][1])
            elif self.stage == 8 or self.stage == 9 or self.stage == 12:
                # --- Define static goal positions here ---
                goal_pose_list = [[2.0, 2.0], [2.0, 1.5], [2.0, -0.5], [2.0, -1.0], [2.0, -2.0], [1.3, 1.0],
                                    [1.0, 0.3], [1.0, -2.0], [0.3, -1.0],  [0.0, 2.0], [0.0, -1.0], [-1.0, 1.0],
                                        [-1.0, -1.2], [-2.0, 1.0], [-2.2, 0.0], [-2.0, -2.2], [-2.4, 2.4]]
                index = random.randrange(0, len(goal_pose_list))
                self.goal_x = float(goal_pose_list[index][0])
                self.goal_y = float(goal_pose_list[index][1])
            elif self.stage not in [4, 5, 7]:
                self.goal_x = random.randrange(-15, 16) / 10.0
                self.goal_y = random.randrange(-15, 16) / 10.0
            else:
                # --- Define static goal positions here ---
                goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                                  [-1.9, 1.9], [-1.9,  0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0],
                                  [1.5, -1.0], [-0.5, 1.0], [-1.0, -2.0], [1.8, -0.2], [1.0, -1.9]]
                index = random.randrange(0, len(goal_pose_list))
                self.goal_x = float(goal_pose_list[index][0])
                self.goal_y = float(goal_pose_list[index][1])
            tries += 1
            if tries > 100:
                print("ERROR: distance between goals is small")
                break
        self.publish_callback()

    ## Function to reset the simulation by using a client
    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')
        self.reset_simulation_client.call_async(req)

    ## Function to delete the goal entity by using a client
    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.delete_entity_client.call_async(req)

    ## Function to spawn the goal entity
    def spawn_entity(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_x
        goal_pose.position.y = self.goal_y
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)

    ## Get the pose of the obstacles
    def get_obstacle_coordinates(self):
        tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/inner_walls/model.sdf')
        root = tree.getroot()
        obstacle_coordinates = []
        for wall in root.find('model').findall('link'):
            pose = wall.find('pose').text.split(" ")
            size = wall.find('collision').find('geometry').find('box').find('size').text.split()
            rotation = float(pose[-1])
            pose_x = float(pose[0])
            pose_y = float(pose[1])
            if rotation == 0:
                size_x = float(size[0]) + self.no_goal_spawn_margin * 2
                size_y = float(size[1]) + self.no_goal_spawn_margin * 2
            else:
                size_x = float(size[1]) + self.no_goal_spawn_margin * 2
                size_y = float(size[0]) + self.no_goal_spawn_margin * 2
            point_1 = [pose_x + size_x / 2, pose_y + size_y / 2]
            point_2 = [point_1[0], point_1[1] - size_y]
            point_3 = [point_1[0] - size_x, point_1[1] - size_y ]
            point_4 = [point_1[0] - size_x, point_1[1] ]
            wall_points = [point_1, point_2, point_3, point_4]
            obstacle_coordinates.append(wall_points)
        return obstacle_coordinates

## Main function to create a node object
def main():
    rclpy.init()
    drl_gazebo = DQNGoals()
    rclpy.spin(drl_gazebo)

    drl_gazebo.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
