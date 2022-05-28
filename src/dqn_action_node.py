#!/usr/bin/env python3
# coding=utf-8


##----------------INIT----------------##
##----------------Python Package Import----------------##
import rospy
import numpy as np
import rospy
from math import pi
from environment_stage import Env
import time
##----------------Pytorch Package Import----------------##
import torch                                    # import torch
import torch.nn as nn                           # import torch.nn
import torch.nn.functional as F                 # import torch.nn.functional
##----------------ROS Message Import----------------##
from std_msgs.msg import Float32MultiArray
##----------------Deep Q Learning Parameter----------------##
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #Hyperparameters
BATCH_SIZE = 512 #128                                # Batch Size
LR = 0.001                                      # Learning Rate
EPSILON = 1.0                                   # Greedy Policy weight
GAMMA = 0.99                                     # Reward Discound
MEMORY_CAPACITY = 50000 #5000                          # Memory Capacity
TARGET_REPLACE_ITER = 50                        #MEMORY_CAPACITY // BATCH_SIZE   # Target Network Update Frequency
N_ACTIONS = 3                                   # Action number
N_STATES =  626 #??                                  # State number          
env = Env()

rospy.init_node('controller_dqn', anonymous=False)
rate = rospy.Rate(2)
pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

device="cuda"   
##----------------Function Defination----------------##
##----------------Define the Neural Network----------------##
class Net(nn.Module):                               ##Define the Network Class
    def __init__(self):                             # Define the net
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1,8,5, padding=1)     # Convolutional layer 1
        #self.conv2 = nn.Conv2d(8,16,5, padding=1)    # Convolutional layer 2
        #self.maxpool = nn.MaxPool2d(2,2)             # Pooling layer
        self.fc1 = nn.Linear(N_STATES, 1024)         # Full connection layer, from input layer to hidden layer
        #self.fc1.weight.data.normal_(0, 0.1)        # Weight Initialization (Gaussian distribution)
        self.fc2 = nn.Linear(1024, 256)              
        #self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, N_ACTIONS)        # Full connection layer, from hidden layer to output layer
        #self.out.weight.data.normal_(0, 0.1)        # Weight Initialization (Gaussian distribution)

    def forward(self, x):                           # Define the forward function (x is the state)
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.maxpool(x)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))                     # Use Rectified Linear Unit (ReLU) to process the value in hidden layer
        x = F.relu(self.fc2(x))
        #x = F.dropout(self.fc2(x))
        actions_value = self.out(x)                 # Output the action value by output layer                       
        return actions_value    
##----------------Define the Agent----------------##
class DQN(object):                                                              ##Define the DQN class
    def __init__(self):                                                         # Define DQN 
        self.eval_net, self.target_net = Net().to(device), Net().to(device)                           # Define Evaluation Network and Target Network
        self.learn_step_counter = 0                                             # For target updating
        self.memory_counter = 0                                                 # For storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # Initialize memory 
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # Use Adam optimizer
        self.loss_func = nn.MSELoss().to(device)                                           # Use meansquare loss function (loss(xi, yi) = (xi-yi)^2)
        self.start_epoch = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05 #0.05
        self.load_models = False
        self.load_ep = 0
        self.loss =0
        self.q_eval=0
        self.q_target=0
        if self.load_models:
            self.epsilon= 0.05
            self.start_epoch=self.load_ep
            checkpoint = torch.load("/home/youchen/dqn_model/"+str(self.load_ep)+".pt")
            #print(checkpoint.keys())
            #print(checkpoint['epoch'])
            #print(checkpoint)
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = 90
            print("loadmodel")

    def choose_action(self, x):                                                 # Define choose action function (x is the state)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # transfer x into 32-bit floating point tensor (or use torch.cuda.FloatTensor)
                                                                                # and returns a new tensor with a dimension of size one inserted at the specified position
        if np.random.uniform() > self.epsilon:                                  # Generate a random number in (0.1). Use greedy policy
            actions_value = self.eval_net.forward(x.to(device))                            # Input x into Evaluation Network, get action value with forward
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()                # Output the index of max value in each line, and transfer into numpy ndarray
            action = action[0]                                                  # Putput first number of action

        else:                                                                   # or choose random action
            action = np.random.randint(0, N_ACTIONS)                                    # 0,1:vel   2,3:joint1   4,5:joint2
        return action

    def store_transition(self, s, a, r, s_):                                    # Define memory function (input is a transition)
        transition = np.hstack((s, [a, r], s_))                                 # Stack arrays in sequence horizontally (column wise)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY                           
        self.memory[index, :] = transition                                      
        self.memory_counter += 1 
    
    def learn(self):                                                            # Define learning function
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # Update every 10 times 
            self.target_net.load_state_dict(self.eval_net.state_dict())         # Give parameter in evaluation network to target network
        self.learn_step_counter += 1                                            # learn step +1

        # sample batch transitions in memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # random choose 128 numbers in [0,2000]
        b_memory = self.memory[sample_index, :]                                 # put the 128 related transitions in b_memory
        
        # Get 128 s,a,r,s_ from b_memory and transfer into 32/64-bit floating point and save in b_
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)                         # 128 line 4 row
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)  # Use LongTensor for using in torch.gather, 128 line 1 row
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).to(device)         # 128 line 1 row
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)                       # 128 line 4 row
        b_d = b_memory[:, -1].to(device)

        # Get evaluation value and target value of 128 transitions and update Evaluation Network by loss function and optimizer
        q_eval = self.eval_net(b_s).gather(1, b_a)                              # gather q value of b_a of related 128 b_s, and shape (batch, 1)
        q_next = self.target_net(b_s_).detach()                                 # detach from graph, don't backpropagate
        d_list = torch.unsqueeze(torch.tensor(((np.asarray(b_d).astype(int) + 1) % 2)), dim=1)
        q_target = b_r + d_list * GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1).to(device)  # return max value in each row, and shape (batch, 1)
        
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()                                              # Clear residual parameters
        loss.backward()                                                         # Backpropogate error
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.)       # Clip gradient norm to 1
        self.optimizer.step()                                                   # Update parameters in Evaluation Network

    def save_model(self,dir):
        state = {'target_net':self.target_net.state_dict(),'eval_net':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':dir}
        torch.save(state,"/home/youchen/dqn_model/"+ dir+".pt")



def main_loop():
    dqn = DQN()
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    e = dqn.start_epoch
    env.generateGoal()
    reward_list = []
    for e in range(5000):
        print('episode '+ str(e))
        s = env.reset()
        episode_reward_sum = 0
        done = False
        episode_stop = 1000
        for t in range(episode_stop):
            a = dqn.choose_action(s)
            s_, r, done = env.step(a)
            dqn.store_transition(s, a, r, s_)
            episode_reward_sum += r
            s = s_
            pub_get_action.publish(get_action)

            if dqn.memory_counter > 5000: #MEMORY_CAPACITY:
                dqn.learn()

            if t >=200:
                rospy.loginfo("time out!")
                done = True

            if done:
                result.data =[episode_reward_sum,float(dqn.loss),float(dqn.q_eval),float(dqn.q_target)]
                pub_result.publish(result)
                reward_list.append(round(episode_reward_sum,2))
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f',e, episode_reward_sum, dqn.memory_counter, dqn.epsilon)

                param_keys = ['epsilon']
                param_values = [dqn.epsilon]    
                param_dictionary = dict(zip(param_keys, param_values))
                break

            if dqn.epsilon > dqn.epsilon_min :
                    dqn.epsilon =dqn.epsilon - 5e-5
            
            rate.sleep()

        if e % 10 == 0:
            dqn.save_model(str(e))

        print(reward_list)

def main():
    while not rospy.is_shutdown():
        main_loop()

##----------------MAINNNNN----------------##

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass