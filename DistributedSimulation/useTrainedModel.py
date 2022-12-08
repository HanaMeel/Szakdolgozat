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

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def run():
    QTable = open("QTable4states9.txt", "r")
    strQTable = (QTable.read()).split(" ")
    table = np.zeros((n_observations,n_actions))
    indexes = []
    
    for i in range(0,4):
        for j in range(0,4):
            if [i , j] not in indexes:
                indexes.append([i, j])
    k = 0
    for t in strQTable:
        table[indexes[k][0], indexes[k][1]] = t
        k = k + 1
    print(table)
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        IDList = traci.vehicle.getIDList()
        for ID in IDList:
            state = 1
            traci.vehicle.setMinGap(ID, 10)
            mingap = traci.vehicle.getMinGap(ID)
            speed = traci.vehicle.getSpeed(ID)
            leader = traci.vehicle.getLeader(ID, mingap)
            if leader != None:
                leaderSpeed = traci.vehicle.getSpeed(leader[0])
            
            if leader == None or (leader != None and leaderSpeed > speed and leader[1] >= 50):
                state = 1
            if leader != None and leaderSpeed > speed and leader[1] >= mingap and leader[1] < 50:
                state = 2
            if leader != None and leader[1] >= mingap and leaderSpeed < speed:
                state = 3
            if leader != None and leader[1] < mingap:
                state = 4
            
            action = np.argmax(table[state-1,:])

            if action == 1:
                    traci.vehicle.setSpeed(ID, speed+5)
            if action == 2:
                traci.vehicle.setSpeed(ID, speed+3)
            if action == 3:
                traci.vehicle.setSpeed(ID, speed-3)

# main entry point
if __name__ == "__main__":

    options = get_options()

    #argparser for the input variables

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    #run()
    #QLearning()
    #Q_table.tofile("QTable.txt", sep=' ', format='%s')
    #reading input from the given file with argparse
    traci.start([sumoBinary, "-c", "mySimulationDistributed.sumocfg", 
                             "--tripinfo-output", 
                             "tripinfoWithDistributedControl4States9.xml",
                             "--collision.mingap-factor", "0",
                             "--collision-output",
                             "collisionOut4States9.xml",
                             "--start"])
    
    run()