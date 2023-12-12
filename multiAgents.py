# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        ### return float inf mmeans that pacman should defintely go there (encourage pacamn)
        ### - inf  means that pacman should leave(discourage pacman) 

        score=successorGameState.getScore()
        ghosts_distances=[manhattanDistance(newPos,Ghost.getPosition()) for Ghost in newGhostStates]
        
        for pos,ghostStates in enumerate(newGhostStates):
            distance_of_ghost=ghosts_distances[pos]

            if(distance_of_ghost)<2 : #a ghost is reaching out  
                #got to leave
                return -float("inf")
        min_distance_of_food=[]#to find the minimum distance of food
        dots=newFood.asList() #get the nearby dots 
        if action==Directions.STOP: #discourages pacman to stay still #.STOP returns from getAction function 
            return -float("inf")
        for food in dots: #finding the nearest dot to eat
             min_distance_of_food.append(manhattanDistance(newPos,food)) #adding the distance of new pacman and the dots 
        if len(min_distance_of_food)==0: #empyt list.it should move(prevent loosing points from being stucked) 
            return float('inf')
        return score + 10/min(min_distance_of_food)#enoucrage pacman to eat the nearest dot 
    ## we add to score 10/min_distnace and not just min distance because we want to give 
    #to this distance higher value that the other distances in order to go to that dot 
    #the shorter distance the higher the near value 
    #the division with ten does not really matter it could be any number(i used 10 cause is the value of food dots )

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def MiniMax( gameState): #there is explamation of the pseudocode given 
            return max_evaluation(gameState,0)[0] #return a tuple (move , evaluation ) , need to return the move 
    
        def max_evaluation(gameState,depth):  
            legal_actions= gameState.getLegalActions(0) #get legal action of pacman agent 
            if len(legal_actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth: #terminal test -> return utility 
                return (None , self.evaluationFunction(gameState))#IF TERRMINAL TEST RETURN UTITLITY STATE 
            max_eval= -float("inf")#v inintialize infinity
            best_move=None 
            for action in legal_actions: #for each a state in actions(state ) do 
                eval=min_evaluation(gameState.generateSuccessor(0,action),1,depth)  #recursive call v=max_evaliation(v,min-value(result(state,a)))
                eval=eval[1] #get the value ot the tuple 
                if eval> max_eval: #serach max value 
                    max_eval=eval
                    best_move=action
            return (best_move, max_eval)  #return v
     
        def min_evaluation(gamestate,agent_number , depth):  #agent number : 1)first ghost  , 2)second ghost.... 
            legal_actions= gamestate.getLegalActions(agent_number) #get the legal action of the ghost 
            if len(legal_actions)==0: 
                return (None, self.evaluationFunction(gamestate) ) #terminal test -> return utility 
            min_eval= float("inf") #initialize v=infinity
            best_move=None
            for action in legal_actions: #for each a state in actions(state ) do 
               if agent_number==gamestate.getNumAgents()-1: #there is one ghost -> call max
                #gamsestate.getnumagents()-1 because all agents are 3 with pacman so in order
                #to get the numbner of ghosts we abstract 1 
                #if there are 2 ghosts  we should call the again min and not max for the second ghost 
                    eval=max_evaluation(gamestate.generateSuccessor(agent_number,action),depth+1) 
               else :  
                    eval=min_evaluation(gamestate.generateSuccessor(agent_number,action),agent_number+1,depth) 
               eval=eval[1]#recursive call v=max_evaliation(v,min-value(result(state,a)))
               if eval<min_eval :
                    min_eval=eval
                    best_move= action                    
            return (best_move , min_eval)
            

        return MiniMax(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(gameState): #there is explanation of the pesudocode given 
            return maxEvaluation_AB(gameState,0,-float("inf"),float("inf"))[0] #setting a=-inf and b=inf 
        def maxEvaluation_AB(gameState,depth,maxvalue,minvalue):
            legal_actions= gameState.getLegalActions(0) #get legal action of pacman agent 
            if len(legal_actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth: #terminal test -> return utility 
                return (None , self.evaluationFunction(gameState))
            max_eval= -float("inf")#initialize v=infinity
            best_move=None
            for action in legal_actions: #for each successor state:
                eval=minEvaluation_AB(gameState.generateSuccessor(0,action),depth,1,maxvalue,minvalue) #v = max(v,value(successor,a,b))
                eval=eval[1] 
                if eval>=max_eval:
                    max_eval=eval
                    best_move=action
                if eval>minvalue: #if v>b return v 
                    return (action,eval)
                if eval>maxvalue : #a=max(a , v )
                    maxvalue=eval
            return(best_move , max_eval) #return v  
       
       
        def minEvaluation_AB(gameState , depth , agent_number ,  maxvalue , minvalue ):
            legal_actions= gameState.getLegalActions(agent_number) #get the legal action of the ghost 
            if len(legal_actions)==0: 
                return (None, self.evaluationFunction(gameState) ) #terminal test -> return utility 
            min_eval= float("inf")#initialize v=inf
            best_move=None
            for action in legal_actions: #for each successor of state
                if agent_number==gameState.getNumAgents()-1: #v=max(v,value(successor,a , b)) 
                    #if its pacman turn 
                    eval=maxEvaluation_AB(gameState.generateSuccessor(agent_number,action),depth+1,maxvalue,minvalue) 
                else :  
                    #second agent turn
                    eval=minEvaluation_AB(gameState.generateSuccessor(agent_number,action),depth,agent_number+1,maxvalue, minvalue) 
                eval=eval[1] #take the value of the tuple 
                if eval<= min_eval: 
                    min_eval=eval
                    best_move=action
                if eval< maxvalue: #if v<a return v
                    return (action, eval)
                if eval<minvalue: #switch value to the b
                    minvalue=eval # b=min(b,v)
            return (best_move,min_eval)#rutern v 
        return alphaBeta(gameState)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def ExpectMinimax(gameState):
            return max_evaluation_expect_minimax(gameState,0)[0]
        def max_evaluation_expect_minimax(gameState,depth): 
            legal_actions= gameState.getLegalActions(0) #get legal action of pacman agent 
            if len(legal_actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth: #terminal test -> return utility 
                return (None , self.evaluationFunction(gameState))
            max_eval= -float("inf")#v initialize - inf
            best_move=None
            for action in legal_actions: #recursive call
                eval=min_eval_expect_minimax(gameState.generateSuccessor(0,action),1,depth,0)
                 #get the value ot the tuple 
                if eval> max_eval: #serach max value 
                    max_eval=eval
                    best_move=action
            return (best_move, max_eval) 
        
        
        
        def min_eval_expect_minimax(gameState,agent_number , depth,eval):  #agent number : 1)first ghost  , 2)second ghost.... 
            legal_actions= gameState.getLegalActions(agent_number) #get the legal action of the ghost 
            if len(legal_actions)==0 or self.depth==depth: 
                return self.evaluationFunction(gameState)  #terminal test -> return utility 
            successors=[] #this list will help is to reuturn the average heuristic of the successors
            for action in legal_actions: 
                successors.append(gameState.generateSuccessor(agent_number,action))
                if agent_number==gameState.getNumAgents()-1: 
                    #counting the probability of the successor states
                    eval+=max_evaluation_expect_minimax(gameState.generateSuccessor(agent_number,action),depth+1)[1]
                else :  
                    eval+=min_eval_expect_minimax(gameState.generateSuccessor(agent_number,action),agent_number+1,depth,eval)  
            return eval/len(successors) #return average eval value(chance) 
        return ExpectMinimax(gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    we use for evaluation function point by:
    -DISTANCE OF FOOD(food dots)
    -CAPSULES(big dots)
    -DINSTANCE OF GHOST(if they are eatable encourage pacamn to eat )
    if the game is win we encourage pacman to move 
    We add to score 10/ghost_point(or food points) and not just food points because we want to give 
    to this distance higher value that the other distances in order to go to that dot  
    example:if ghost_points>food_points and i want to prioritize food then 
    1/food_points>1/ghost_points
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance #using manhattan distance to find  ghost distance 
    pacman_position=currentGameState.getPacmanPosition()  #current pacman position
    GameScore=scoreEvaluationFunction(currentGameState) #current gamescorfe
    GhostList=currentGameState.getGhostPositions() #list of the position of the ghosts
    GhostStates=currentGameState.getGhostStates() #list of future ghosts state
    ScaredTimes=[ghoststate.scaredTimer for ghoststate in GhostStates] #times when the ghosts are eatable
    capsules_number=len(currentGameState.getCapsules()) #big dots should give some points
    FoodList=currentGameState.getFood().asList() #list with food states
    number_of_food=currentGameState.getNumFood()  
    if currentGameState.isWin(): #win state should definitely visit
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")
    #food distance points
    distance_of_food= [manhattanDistance(food,pacman_position) for food in FoodList]
    if len(distance_of_food)==0: #empyt list.it should move(prevent loosing points from being stucked) 
            return float('inf')
    food_points=10.0/(sum(distance_of_food))*number_of_food #all available poiint of food
    #ghost dinstance 
    ghost_points=0
    for index in range(len(ScaredTimes)):
        if ScaredTimes[index]==0:#non eatable ghost 
            if manhattanDistance(pacman_position,GhostList[index])<2:#ghost is reaching
                ghost_points-=manhattanDistance(pacman_position,GhostList[index])#discourage pacman to go the that direction 

        else:#eatable ghost 
            ghost_points+=10.0/((manhattanDistance(pacman_position,GhostList[index]))+1) *200 
            #+1 IN ORDER NOT TO DIVIDE BY 
            #multiplication of 200 beacause 200 are the points of each ghost
    GameScore+= food_points+10.0/(capsules_number+1)+ghost_points
    return GameScore

    
# Abbreviation
better = betterEvaluationFunction
