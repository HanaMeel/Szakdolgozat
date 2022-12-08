import numpy as np
import random
import math

import os
import sys
import optparse
from turtle import color, pos
from random import sample
import argparse

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

#states
#1-getting closer to car in front of me
#2-getting farther from the car in front of me
#3-distance is constant from the car in front of me
observation_space = [1,2,3,4]

#actions
#1-accelerating
#2-slowing
#3-keep speed
action_space = [1,2,3,4]
#actions
n_observations = len(observation_space)
n_actions = len(action_space)

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))

#number of episode we will run
n_episodes = 250
#initialize the exploration probability to 1
exploration_proba = 1
#exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001
# minimum of exploration proba
min_exploration_proba = 0.001
#discounted factor
gamma = 0.99
#learning rate
lr = 0.1
#szimulacioban levo jarmuvek ID-jai minden meresi idopillanatban
IDList = []
total_rewards_episode = list()

def step(action, mingap, currentVehicleID): 
    
    try:     
        speed = traci.vehicle.getSpeed(currentVehicleID)
        leader = traci.vehicle.getLeader(currentVehicleID, mingap)
        if leader != None:
            leaderSpeed = traci.vehicle.getSpeed(leader[0])

        reward = 0
        nextState = 0
        done = False
        
        # action
        if action == 1:
            traci.vehicle.setSpeed(currentVehicleID, speed+5)
        if action == 2:
            traci.vehicle.setSpeed(currentVehicleID, speed+3)
        if action == 3:
            traci.vehicle.setSpeed(currentVehicleID, speed-3)
        
        # reward calculation
        if leader == None or (leader != None and leaderSpeed > speed and leader[1] >= 50):
            reward = speed * 1.5
            nextState = 1
        if leader != None and leaderSpeed > speed and leader[1] >= mingap and leader[1] < 50:
            reward = speed
            nextState = 2
        if leader != None and leader[1] >= mingap and leaderSpeed < speed:
            reward = speed - (speed - leaderSpeed)
            nextState = 3
        if leader != None and leader[1] < mingap:
            reward = speed * 0.5
            nextState = 4
    except:
        done = True
        nextState = 0
        reward = 0
    
    return [nextState, reward, done]

def QLearning():
    global exploration_proba, exploration_decreasing_decay, min_exploration_proba, gamma, lr, Q_table
    #we iterate over episodes
    rewards_per_episode = list()
    episodeCount = 0
    for e in range(n_episodes):
        traci.start([sumoBinary, "-c", "mySimulationDistributed.sumocfg",
                             "--collision.mingap-factor", "0",
                             "--start",
                             "--Q"
                             ])
        episodeCount = episodeCount + 1
        print(episodeCount)
        
        #we initialize the first state of the episode
        #at first we keep the current speed
        current_state = 1
        done = False

        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0
        stepCount = 0
        currentVehicleID = 0.0
        mingap = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled flaot is less than the exploration proba
            #     the agent selects a random action
            # else
            #     he exploits his knowledge using the bellman equation 
            traci.simulationStep()
            
            if stepCount == 50:
                IDList = traci.vehicle.getIDList()
                currentVehicleID = random.choice(IDList)
                traci.vehicle.setMinGap(currentVehicleID, 5)
                mingap = traci.vehicle.getMinGap(currentVehicleID)
            if stepCount >= 50:
                if np.random.uniform(0,1) < exploration_proba:
                    action = random.choice(action_space)
                else:
                    action = np.argmax(Q_table[current_state-1,:])

                next_state, reward, done = step(action, mingap, currentVehicleID)
                
                if done:
                    break
                # We update our Q-table using the Q-learning iteration
                Q_table[current_state-1, action-1] = (1-lr) * Q_table[int(current_state)-1, int(action)-1] +lr*(reward + gamma*max(Q_table[int(next_state)-1,:]))
                
                total_episode_reward = total_episode_reward + reward

                # If the episode is finished, we leave the for loop#
                current_state = next_state
            stepCount = stepCount +1
        #We update the exploration proba using exponential decay formula 
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
        rewards_per_episode.append(total_episode_reward)
        traci.close()
        sys.stdout.flush()

    f = open("rewards_per_episode4states9.txt", "w")
    for reward in rewards_per_episode:
        f.write("%s\n" % reward)
    f.close()
    Q_table.tofile("QTable4states9.txt", sep=' ', format='%s')

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

# main entry point
if __name__ == "__main__":

    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    QLearning()