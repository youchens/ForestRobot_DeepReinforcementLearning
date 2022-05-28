#!/usr/bin/env python
# coding=utf-8


##----------------INIT----------------##
##----------------Python Package Import----------------##
from turtle import done, st
import rospy
import numpy as np
import math
import rospy
import random
import os
import time
from math import pi
##----------------ROS Message Import----------------##
from std_msgs.msg import Float64
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel, SpawnModelRequest, DeleteModelRequest
from geometry_msgs.msg import Pose
##----------------Define the Environment----------------##
class Env():

    def __init__(self):
        self.start_point_x = 20
        self.start_point_y = 20
        self.start_point_z = 1.2
        self.goal_point_x_list = []
        self.goal_point_y_list = []
        self.goal_point_x = 0
        self.goal_point_y = 0
        self.train_radius = 5
        self.initGoal = True
        self.height_array =[]
        self.position_x = 0
        self.position_y = 0
        self.position_z = 0
        self.angular_roll = 0
        self.angular_pitch = 0
        self.euler_yaw = 0
        self.check_model = False
        self.marker_array={}
        self.indexRobot=[]
        # ROS Initialize
        self.pubVel = rospy.Publisher('rig_vel_controller/cmd_vel', Twist, queue_size=1)
        self.pubJoint = rospy.Publisher('joint_position_controller/command', Float64, queue_size=1)
        self.pubJoint2 = rospy.Publisher('joint_position_controller2/command', Float64, queue_size=1)

        self.sub_pose = rospy.Subscriber('/my_odom', Odometry, self.get_odom)
        self.sub_markerArray_map = rospy.Subscriber('height_array', Float32MultiArray, self.update_marker)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        # Respawn the model
        self.modelName = "testrig"
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace("controller_dqn/src","hktest/urdf/urdf_suspension/rig.urdf_dyn.xacro")
        self.f = open(self.modelPath, "r")
        self.model = self.f.read()   
    
    def generateGoal(self):
        # Generate goal list
        self.theta = 20
        goal_num = round(360/self.theta)
        for i in range(0, goal_num):
            self.goal_point_x_list.append(self.start_point_x + self.train_radius * math.cos(self.theta * i))
            self.goal_point_y_list.append(self.start_point_y + self.train_radius * math.sin(self.theta * i))
    
    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "testrig":
                self.check_model = True

    def get_odom(self,msg):                                                         # Get the position and roll, pitch angular
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        self.angular_roll = msg.twist.twist.angular.x
        self.angular_pitch = msg.twist.twist.angular.y
    
    def update_marker(self, heighArray_msg): #updates the obstacles
        global occupancy_map
        tmp_map = {}
        self.marker_array = heighArray_msg
        self.indexRobot = [self.marker_array.data[0], self.marker_array.data[1]]
        for i in range(0, int(self.marker_array.layout.dim[0].size)):
            for j in range(0, int(self.marker_array.layout.dim[1].size)):
                tmp_map[(i,j)] = self.marker_array.data[self.marker_array.layout.data_offset + j + self.marker_array.layout.dim[1].stride*i]
        occupancy_map = tmp_map
        for i in range(int(self.marker_array.layout.dim[0].size)):
                for j in range(int(self.marker_array.layout.dim[1].size)):
                    self.height_array.append(occupancy_map[i,j])

    def IndexToGlobal(self, x, y):
        grid_resolution = 0.2
        global_position_x = self.position_x + (self.indexRobot[0] - x) * grid_resolution
        global_position_y = self.position_y + (self.indexRobot[1] - y) * grid_resolution
        return [round(global_position_x,2), round(global_position_y,4)]

    def coord_local_to_global(self):
        grid_map_list = []
        time.sleep(0.1)
        for i in range(int(self.marker_array.layout.dim[0].size)):
            for j in range(int(self.marker_array.layout.dim[1].size)):
                grid_map_list.append(self.IndexToGlobal(i, j))
        return grid_map_list

    def getState(self):                                                   # Get the scan information
        min_range = 0.3
        done = False
        scan_range = []
        grid_map_list = self.coord_local_to_global()
        state = []
        
        for i in range(625):
            if self.height_array[i] == float('nan'):
                scan_range.append(0)
            elif np.isnan(self.height_array[i]):
                scan_range.append(0)
            else:
                scan_range.append(round(self.height_array[i],4))
        
        state_position_x = self.position_x
        state_position_y = self.position_y
        state_angular_roll = self.angular_roll
        state_angular_pitch = self.angular_pitch
    
        if np.sqrt((state_position_x-self.goal_point_x)**2 + (state_position_y-self.goal_point_y)**2) < min_range:
            done = True
        
        gesture = [state_position_x, state_position_y, state_angular_roll, state_angular_pitch]
        
        #for k in range(len(grid_map_list)):
            #state.append([grid_map_list[k][0], grid_map_list[k][1], round(scan_range[k],4)])
        state = scan_range
        abs_dist = np.sqrt((self.position_x - self.goal_point_x)**2 + (self.position_y - self.goal_point_y)**2)
        state.append(abs_dist)
        return state, gesture, done

    def get_drift_distance(self):                          # Calculate the drift distance
        A = self.goal_point_y - self.start_point_y
        B = self.start_point_x - self.goal_point_x
        C = self.start_point_x * (self.start_point_y - self.goal_point_y) + self.start_point_y * (self.goal_point_x - self.start_point_x)
        drift_distance = - np.abs(A * self.position_x + B * self.position_y + C) / (np.sqrt(A**2 + B**2))*0.1
        #print(drift_distance)
        return drift_distance

    #def step_reward(self):
        abs_dist = np.sqrt((self.position_x - self.goal_point_x)**2 + (self.position_x - self.goal_point_y)**2)
        if abs_dist > 5:
            step_reward_value = 0
        elif 5 >= abs_dist > 4:
            step_reward_value = 0.1
        elif 4 >= abs_dist > 3:
            step_reward_value = 0.2
        elif 3 >= abs_dist > 2:
            step_reward_value = 0.3
        elif 2 >= abs_dist > 1:
            step_reward_value = 0.4
        elif 1 >= abs_dist >0:
            step_reward_value = 0.5
        else:
            print('step reward value error!')
            step_reward_value  = 0
        
        return step_reward_value
    
    def step_reward(self):
        abs_dist = np.sqrt((self.position_x - self.goal_point_x)**2 + (self.position_y - self.goal_point_y)**2)

        if abs_dist > 5:
            step_reward_value  = - (abs_dist - 5) * 0.1
        elif abs_dist <= 5 :
            step_reward_value = (5/abs_dist) * 0.2
        else:
            print('step reward value error!')
            step_reward_value  = 0            
        #print(step_reward_value)
        return step_reward_value

    def setReward(self, gesture, done, drift_distance, step_reward_value):

        roll_reward = -round((np.abs(gesture[-2])),5)
        pitch_reward = -round((np.abs(gesture[-1])),5)
        drift_dist_reward = drift_distance
        consumption = 0

        #reward = roll_reward + pitch_reward + drift_dist_reward + step_reward_value
        reward = drift_dist_reward + step_reward_value - consumption

        if gesture[-2] > 2:
            rospy.loginfo("Collision!!")
            reward -= 100
            self.pubVel.publish(Twist())
            self.resetWorld()
            time.sleep(0.3)

        if done:
            rospy.loginfo("Goal!!")
            reward += 200
            self.pubVel.publish(Twist())
            self.resetWorld()
            time.sleep(0.3)

        return reward

    def step(self, act):
        action = act
        vel_cmd = Twist()
        joint_cmd = Float64
        joint2_cmd = Float64

        vel_cmd.linear.x = 0.2
        joint_cmd = 0
        joint2_cmd = 0

        #if action == 0:
            #vel_cmd.linear.x = 0
            #print('linear 0')
        #elif action == 1:
            #vel_cmd.linear.x = 0.2
            #print('linear 1')
        if action == 0:
            joint_cmd = 0
            joint2_cmd = 0
        elif action == 1:
            joint_cmd = 0.4
            joint2_cmd = -0.28
        elif action == 2:
            joint_cmd = -0.4
            joint2_cmd = 0.28
        #elif action == 3:
            #joint2_cmd = 0
        #elif action == 4:
            #joint2_cmd = 0.4
        #elif action == 5:
            #joint2_cmd = -0.4
        else:
            print('action error')
            joint_cmd = 0
            joint2_cmd = 0

        self.pubVel.publish(vel_cmd)
        self.pubJoint.publish(joint_cmd)
        self.pubJoint2.publish(joint2_cmd)
        
        data = None
        while data is None:
            try:
                self.sub_markerArray_map
                data = self.height_array
            except:
                pass

        data = None
        state, gesture, done = self.getState()
        drift_distance = self.get_drift_distance()
        step_reward_value = self.step_reward()
        reward = self.setReward(gesture, done, drift_distance, step_reward_value)
        return np.array(state, dtype=np.float32), reward, done

    def reset(self):
        data =None
        while data is None:
            try:
                self.sub_markerArray_map
                data = self.height_array
            except:
                pass
        if self.initGoal:
            n = random.randint(0,19)
            #self.goal_point_x= round(self.goal_point_x_list[n],3)
            #self.goal_point_y = round(self.goal_point_y_list[n],3)
            self.goal_point_x = 23.416
            self.goal_point_y = 16.349
            self.initGoal = False

            #self.deleteModel()
        self.resetWorld()
        time.sleep(0.3)
        print(self.goal_point_x, self.goal_point_y)
        state, gesture, done = self.getState()
        return np.array(state, dtype=np.float32)
    
    #def respawnModel(self, robot_num):
        while True:
            if not self.check_model:
                initial_pose = Pose()
                initial_pose.position.x = 20
                initial_pose.position.y = 20
                initial_pose.position.z = 1.2

                robot_namespace = "testrig_" + str(robot_num)

                rospy.wait_for_service('gazebo/spawn_urdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, robot_namespace, initial_pose, "empty_world")
                rospy.loginfo("Robot Namespace: %s, Goal position : %.1f, %.1f", robot_namespace, self.goal_point_x,
                              self.goal_point_y)
                break
            else:
                pass
    
    #def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    #def deleteModel(self):
        rospy.wait_for_service('gazebo/delete_model')
        delete_model_service = rospy.ServiceProxy('gazebo/delete_model',DeleteModel)
        objstate = DeleteModelRequest()
        objstate.model_name = "testrig"
        delete_model_service(objstate)
        print('delete model successfully')

    #def respawnModel(self, robot_num):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_service = rospy.ServiceProxy('gazebo/spawn_sdf_model',SpawnModel)
        model_Path = os.path.dirname(os.path.realpath(__file__))
        model_Path = self.modelPath.replace("controller_dqn/src","hktest/urdf/urdf_suspension/rig.urdf_dyn.xacro")
        f_ = open(model_Path, "r")
        model_xml= f_.read()   
        #model_name model_xml robot_namespace initial_pose reference_frame
        objstate = SpawnModelRequest()
        objstate.model_name = 'testrig'
        objstate.model_xml = model_xml
        robot_ns = 'testrig_' + str(robot_num)
        objstate.robot_namespace = robot_ns
        
        pose = Pose()
        pose.position.x = 20 
        pose.position.y = 20
        pose.position.z = 1.2
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        objstate.initial_pose = pose
        objstate.reference_frame = 'ouachitaforest_0'
        try:
            #spawn_model_service("asphalt1_plane",model_xml,"",pose,"world")
            spawn_model_service(objstate)
            print('spawn model successfully')
        except Exception as e:
            print('spawn model failed') 

    def resetWorld(self):
        rospy.wait_for_service('gazebo/reset_world')
        reset_world = rospy.ServiceProxy('gazebo/reset_world', Empty)
        reset_world()
        print('reset successfully')
