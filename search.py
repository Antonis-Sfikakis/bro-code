# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #using stack to implement
    from util import Stack
    dfssearch=Stack() #this stack will dfs 
    visited=set() #in oder to avoid not needed loops  
    startin_position=problem.getStartState()
    path=[] #tracking the path of the grapth 
    dfssearch.push((startin_position,path)) #creatin a tuple with the state and list with the path 
    while not dfssearch.isEmpty():
        current_pos,current_path=dfssearch.pop() #current pos will have the current state and current path
        if problem.isGoalState(current_pos):
            return current_path #found the correct paththe the 
        if current_pos not in visited:
            visited.add(current_pos) #mark it as visited 
            succs=problem.getSuccessors(current_pos) #successors
            for next_position, move , _ in succs:
                if next_position not in visited:
                    new_path=current_path+ [move] #we add the non visisted successor if not marked 
                    dfssearch.push((next_position, new_path)) #we add the new path to the list 
    return []# dfs , nothing found return an empty list 
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #implement using queue in order to use fifo method as we wont search in depth :except that is similar to dfs 
    from util import Queue 
    bfssearch= Queue() #the stack that will dfs 
    visited=set() #in order to avoid not needed loops 
    startin_position=problem.getStartState()
    path=[]#tracking the path
    bfssearch.push((startin_position,path))
    while not bfssearch.isEmpty():
        current_pos, current_path=bfssearch.pop()
        if problem.isGoalState(current_pos):
            return current_path #return the result
        if current_pos not in visited:
            visited.add(current_pos)
            succs=problem.getSuccessors(current_pos)
            for next_position , move , _ in succs:
                if next_position not in visited:
                    new_path= current_path+[move]
                    bfssearch.push((next_position,new_path))
    return [] #bfs not worked; return an empty list

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #implement the unifrom search with priority queue 
    from util import PriorityQueue
    uniformsearch=PriorityQueue()
    path=[]
    visited=set() #avoid extra loops
    uni_dict={} #useful to stores cost of each node 
    startin_position=problem.getStartState()
    uniformsearch.push((startin_position,path),0) #pushing to the queue a tuple with its priority 
    uni_dict[startin_position]=0 #starting position cost is 0
    while not uniformsearch.isEmpty() :
        current_pos,current_path=uniformsearch.pop() 
        if problem.isGoalState(current_pos):
            return current_path
        succs=problem.getSuccessors(current_pos) #adding every succesor to the dictionary with its cost 
        for states in succs:
                new_path=current_path+[states[1]]
                if states[0] not in uni_dict or problem.getCostOfActions(new_path) < uni_dict[states[0]]: #check if the key exits or if its cost is smaller than the exitsting cost of the path
                    cost=problem.getCostOfActions(new_path)
                    uniformsearch.push((states[0],new_path),cost) #pushing to the queue 
                    uni_dict[states[0]]=cost
                    visited.add(states[0])

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
