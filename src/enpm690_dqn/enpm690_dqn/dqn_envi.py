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

import math
import numpy
import sys
import copy
import numpy as np
Infinity = np.inf
import os
from geometry_msgs.msg import Pose, Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlStep, Goal, RingGoal
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

## Environment class
class ENPM690Environment(Node):
    def __init__(self):
        super().__init__('enpm690_environment')
        
        ## Get the number of lidar scan samples from the turtlebot .sdf file
        tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf')
        root = tree.getroot()
        for link in root.find('model').findall('link'):
            if link.get('name') == 'base_scan':
                self.num_scan_samples = int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)
        
        ## Delaring parameters used in the code eand initialiing class variables
        self.linear = 0
        self.angular = 1
        self.episode_timeout = 50 
        self.arena_length = 4.2   # meters
        self.arena_width = 4.2   # meters
        self.linear_max_speed = 0.22  # m/s
        self.angular_max_speed = 2.0   # rad/s
        self.lidar_dist = 3.5   # meters
        self.collision_dist = 0.14  # meters
        self.goal_threshold = 0.22  # meters
        self.obstacle_radius = 0.16  # meters
        self.max_obstacles = 6
        self.max_goal_dist = math.sqrt(self.arena_length**2 + self.arena_width**2) 
        self.enable_dynamic_goals = True
        self.enable_backward = True
        self.scan_topic = 'scan'
        self.velo_topic = 'cmd_vel'
        self.odom_topic = 'odom'
        self.goal_topic = 'goal_pose'
        self.goal_dist_initial = 0
        self.goal_x, self.goal_y = 0.0, 0.0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0
        self.done = False
        self.succeed = 0
        self.episode_deadline = Infinity
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
        self.obstacle_distances = [Infinity] * self.max_obstacles
        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = self.max_goal_dist
        self.initial_distance_to_goal = self.max_goal_dist
        self.scan_ranges = [self.lidar_dist] * self.num_scan_samples
        self.obstacle_distance = self.lidar_dist
        self.difficulty_radius = 1
        self.local_step = 0
        self.time_sec = 0
        
        ## Read the map stage
        with open('/tmp/drlnav_current_stage.txt', 'r') as f:
            self.stage = int(f.read())
        print(f"Stage: {self.stage}")
        
        ## Create publishers, subscribers and clients
        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1)
        self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_profile_sensor_data)
        self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)
        self.task_succeed_client = self.create_client(RingGoal, 'task_succeed')
        self.task_fail_client = self.create_client(RingGoal, 'task_fail')
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)
        
    ## Callback function to get the goal position from 'goal_pose' topic
    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"New goal is x: {self.goal_x} y: {self.goal_y}")

    ## A service callback to check if new goal is recieved or not
    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response

    ## Get dynamic obstacle pose from odometry callback
    def obstacle_odom_callback(self, msg):
        # child_frame_id는 ROS Odometry 메시지 안에 포함된 string 필드로,
        # 해당 데이터가 어떤 프레임(child link)에서 측정된 것인지를 나타냅니다.
        # 동적 장애물 시뮬레이션에서 각 장애물 오도메트리 메시지의 child_frame_id가
        # 예를 들어 "obstacle1", "obstacle2"와 같이 publish됩니다.
        # 즉, 각 오도메트리 publisher(동적 장애물)가 Odometry msg의 child_frame_id 필드를 셋팅해서 송신합니다.
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 1
            ## Get distance from robot to dynamic obstacle
            diff_x = self.robot_x - robot_pos.x
            diff_y = self.robot_y - robot_pos.y
            self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)
        else:
            print("ERROR: received odom was not from obstacle!")

    ## Get robot pose from odometry callback
    def odom_callback(self, msg):
        # Odometry 콜백 함수 (로봇 위치 및 자세 정보를 받아 업데이트한다)
        # 기능: 
        #  - 로봇의 현재 위치(x, y), 헤딩(orientation yaw), 틸트값 저장
        #  - 목표점까지의 거리와 각도를 계산하여 저장
        #  - 특정 주기마다 이동한 누적 거리 계산 (로깅용)
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # 쿼터니언 → 이차원 yaw(heading) 변환
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        self.robot_heading = numpy.arctan2(siny_cosp, cosy_cosp)

        # 로봇의 틸트(roll/pitch 중 y축 값 직접 사용, 추후 전복 판단 등에 활용 가능)
        self.robot_tilt = msg.pose.pose.orientation.y

        # 주기적으로 로봇이 이동한 누적 거리 계산 (32 step마다)
        if self.local_step % 32 == 0:
            dx = self.robot_x_prev - self.robot_x
            dy = self.robot_y_prev - self.robot_y
            self.total_distance += math.sqrt(dx ** 2 + dy ** 2)
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        # 목표점까지의 (평면) 거리 계산
        diff_x = self.goal_x - self.robot_x
        diff_y = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)

        # 목표점까지의 방향(heading) 차이 계산
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        # goal_angle 값을 -pi~pi로 정규화
        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle

    ## Lidar callback function
    def scan_callback(self, msg):
        if len(msg.ranges) != self.num_scan_samples:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {self.num_scan_samples}")
        # normalize laser values
        self.obstacle_distance = 1
        for i in range(self.num_scan_samples):
                self.scan_ranges[i] = numpy.clip(float(msg.ranges[i]) / self.lidar_dist, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    ## Extract the distance to the closest obstacle
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= self.lidar_dist

    ## Clock callback for getting time data
    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec
        if not self.reset_deadline:
            return
        self.clock_msgs_skipped += 1
        # Wait a few message for simulation to reset clock
        if self.clock_msgs_skipped <= 10: 
            return
        episode_time = self.episode_timeout
        if self.enable_dynamic_goals:
            episode_time = numpy.clip(episode_time * self.difficulty_radius, 10, 50)
        self.episode_deadline = self.time_sec + episode_time
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

    
    def stop_reset_robot(self, success):
        # stop the robot
        self.cmd_vel_pub.publish(Twist())
        self.episode_deadline = Infinity
        self.done = True
        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = numpy.clip(self.difficulty_radius, 0.5, 4)
        if success:
            ## Increse the goal difficulty radius for every successful episode
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            ## Decrease the goal difficulty radius for every unsuccessful episode
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)

    ## Function to get the current state. 
    ## It is a combination of laser scan data, realative goal pose and the previous actions 
    ## This will be the parameters that we pass as input to the DQN network
    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.scan_ranges)                                            
        state.append(float(numpy.clip((self.goal_distance / self.max_goal_dist), 0, 1)))    
        state.append(float(self.goal_angle) / math.pi)                                     
        state.append(float(action_linear_previous))                                         
        state.append(float(action_angular_previous))                                       
        self.local_step += 1

        # self.local_step는 get_state가 호출될 때마다 1씩 증가합니다. 즉, 매 스텝마다 올라갑니다.
        if self.local_step <= 30: # Grace period to wait for simulation reset
            return state
        # If the robot reaches the goal: Success
        if self.goal_distance < self.goal_threshold:
            self.succeed = 1
        # If the robot collides with an obstacle: Collision
        elif self.obstacle_distance < self.collision_dist:
            dynamic_collision = False
            ## Check if the robot collided with static or dynamic obstacle
            for obstacle_distance in self.obstacle_distances:
                if obstacle_distance < (self.collision_dist + self.obstacle_radius + 0.05):
                    dynamic_collision = True
            if dynamic_collision:
                self.succeed = 3
            else:
                self.succeed = 2
        # If Timeout
        elif self.time_sec >= self.episode_deadline:
            self.succeed = 4
        # If robot tumbled
        elif self.robot_tilt > 0.06 or self.robot_tilt < -0.06:
            self.succeed = 5
        
        ## Printing the episode outcome
        if self.succeed != 0:
            if self.succeed==5:
                print('\n\n',"Robot Tumbled",'\n\n')
                self.stop_reset_robot(0)
            if self.succeed==4:
                print('\n\n',"Timeout",'\n\n')
                self.stop_reset_robot(0)
            if self.succeed==3:
                print('\n\n',"collision with dynamic obstacle",'\n\n')
                self.stop_reset_robot(0)
            if self.succeed==2:
                print('\n\n',"collision with static obstacle",'\n\n')
                self.stop_reset_robot(0)
            if self.succeed==1:  
                self.stop_reset_robot(1)
        return state

    ## Function to initialize a new episode
    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        self.reward_initalize(self.initial_distance_to_goal)
        return response

    ## A service callback for stepping 
    def step_comm_callback(self, request, response):
        """
        서비스 콜백 함수: 외부 에이전트에서 step 명령이 들어오면 실행.
        - 입력: request.action (정규화된 선형/각속도), request.previous_action (이전 액션)
        - 처리:
            * 액션 언정규화(denormalize) 후 Twist로 변환하여 cmd_vel에 발행
            * 상태(state), 보상(reward), 종료(done), 성공 여부(success), 이동 거리(distance_traveled) 업데이트
            * 에피소드 종료 시 내부 상태, 카운터 등 리셋
            * 200스텝마다 정보 출력
        - 출력: response (각 필드 업데이트)
        """
        if len(request.action) == 0:
            # 액션 벡터가 비어 있으면 에피소드 초기화 (reset)
            return self.initalize_episode(response)

        # 1. 액션 언정규화 (정방향/후진 허용 여부에 따라 다르게 변환)
        if self.enable_backward:
            # 선형 속도는 -1~1을 그대로 곱해서 -max~+max 허용
            action_linear = request.action[self.linear] * self.linear_max_speed
        else:
            # 선형 속도 0~1 → 0~max 변환
            action_linear = (request.action[self.linear] + 1) / 2 * self.linear_max_speed

        action_angular = request.action[self.angular] * self.angular_max_speed

        # 2. Twist 메시지 작성 및 발행 (로봇 제어)
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # 3. 환경 상태 및 보상 계산
        response.state = self.get_state(request.previous_action[self.linear], request.previous_action[self.angular])
        response.reward = self.get_reward_ENPM690(
            self.succeed, 
            action_linear, 
            action_angular, 
            self.goal_distance,
            self.goal_angle, 
            self.obstacle_distance)

        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0

        # 4. 에피소드가 끝났을 때 변수 리셋 및 최종 거리 저장
        if self.done:
            response.distance_traveled = self.total_distance
            self.succeed = 0
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
            self.reset_deadline = True

        # 5. 상태 표시: 200 step마다 진행 상황 프린트
        if self.local_step % 200 == 0:
            print(
                f"Rtot: {response.reward:<8.2f}GD: {self.goal_distance:<8.2f}GA: {math.degrees(self.goal_angle):.1f}°\t", 
                end=''
            )
            print(
                f"MinD: {self.obstacle_distance:<8.2f}Alin: {request.action[self.linear]:<7.1f}Aturn: {request.action[self.angular]:<7.1f}"
            )

        return response
    
    ## initializing the parameters used in the reward function
    def reward_initalize(self,init_distance_to_goal):
        self.goal_dist_initial = init_distance_to_goal
    
    ## Custom reward function for the project  
    def get_reward_ENPM690(self,succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):

        r_yaw = - 0.5 * abs(goal_angle)
        

        r_distance = (5 * self.goal_dist_initial) / (self.goal_dist_initial + goal_dist) 
        # r_distance = (2 * self.goal_dist_initial) / (self.goal_dist_initial + goal_dist) - 1

        if min_obstacle_dist < 0.25:
            r_obstacle = -25
        else:
            r_obstacle = 0

        # reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1
        reward = r_yaw + r_distance + r_obstacle - 1

        if succeed == 1:
            reward += 3500
        elif succeed == 2 or succeed == 3:
            reward -= 2500
        return float(reward)

## Main function for Node object creation and spinning 
def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    if len(args) == 0:
        enpm690_environment = ENPM690Environment()
    else:
        rclpy.shutdown()
        quit("ERROR: wrong number of arguments!")
    rclpy.spin(enpm690_environment)
    enpm690_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
