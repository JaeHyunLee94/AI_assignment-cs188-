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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):

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

        from math import log2
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #successor state를 받아옴
        newPos = successorGameState.getPacmanPosition()  # next pacman postion
        newFood = successorGameState.getFood() # food location 에 대한 정보
        newGhostStates = successorGameState.getGhostStates()# Info about ghost state
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newFood_list = newFood.asList() # get food location in list
        ghost_dis = 20 # initial manhatan dis bet ghost and pacman
        mindis = float('inf') # initialize

        if len(newFood_list) == 0: # next state 가 food가 없는 상태면 큰 보상을 줌
            mindis = -10000

        for food_pos in newFood_list: # save the most minimum distance between pacman and food at mindis
            if manhattanDistance(food_pos, newPos) < mindis:
                mindis = manhattanDistance(food_pos, newPos)

        for ghostState in newGhostStates:# pacman 과 ghost 사이의 distance
            ghost_dis += manhattanDistance(ghostState.getPosition(), newPos)

            if manhattanDistance(ghostState.getPosition(), newPos) < 3: # pacman과 ghost 가 너무 가까이 있으면 ghost_dis 를 작게 만들어 sensitive 하게 반응하게함
                ghost_dis = ghost_dis * 0.0001

        "*** YOUR CODE HERE ***"

        '''
        mindis: 작을수록 보상이 큼
        log2(ghost_dis): 귀신과의 거리가 클수록 보상이큼, 거리가 멀수록 insensitive, 가까울수록 sensitive 하게 반응 
        successorGameState.getNumFood(): food 개수가 줄어들수록 보상이 커짐 
        '''
        return -mindis + 10 * log2(ghost_dis) - 50 * successorGameState.getNumFood()


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, now_depth, agentIndex):
        '''
        gameState: Current gameState
        now_depth: Current depth
        agentIndex: agentIndex (0: pacman else: ghost)
        '''
        if now_depth == self.depth or gameState.isWin() or gameState.isLose():
            #if searching reached leaf or limit depth, terminate.
            return self.evaluationFunction(gameState), []

        numAgent = gameState.getNumAgents()
        legal_action = gameState.getLegalActions(agentIndex)

        pathList = [] #Store (score,pathlist) ex) [(30,[s,w,e,w]),(15,[w,e,s,e])]
        for action in legal_action:
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            '''
            if pacman turn again, current_depth+1
            (agentIndex + 1) % numAgent : next agent
            '''
            score, path = self.minimax(nextGameState, now_depth + 1 if agentIndex == numAgent - 1 else now_depth,
                                       (agentIndex + 1) % numAgent) #recursively find score,path of child node

            pathList.append((score, [action] + path)) # add list as (score,path) tuple

        pathList = sorted(pathList, key=lambda x: x[0]) # sort the list with score

        return pathList[-1] if agentIndex == 0 else pathList[0] # if MaxPlayer return Maximum score, MinPlayer return minimum score

    def getAction(self, gameState):
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

        return self.minimax(gameState, 0, 0)[1][0] # return MaxPlayer's next action


class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self, gameState, now_depth, agentIndex, alpha, beta):
        '''
        alpha:
            if method is in minPlayer state, alpha means the Maximum value
            that had already searched so far.

        beta:
            if method is in maxPlayer state, beta means the Minumum value
            that had already searched so far.
        '''

        if now_depth == self.depth or gameState.isWin() or gameState.isLose():
            '''
            same with minimax
            '''
            return self.evaluationFunction(gameState), []

        numAgent = gameState.getNumAgents()
        legal_action = gameState.getLegalActions(agentIndex)

        pathList = []
        for action in legal_action:
            nextGameState = gameState.generateSuccessor(agentIndex, action)

            score, path = self.alphabeta(nextGameState, now_depth + 1 if agentIndex == numAgent - 1 else now_depth,
                                         (agentIndex + 1) % numAgent, alpha, beta) #same with minimax

            '''
            alpha beta pruning condition
            '''
            if agentIndex == 0: #In MaxPlayer:  beta cutoff
                if score > beta:
                    return score, [action]+path
                alpha = max(score, alpha)
            else: # In MinPlauer: alpha cutoff
                if score < alpha:
                    return score, [action]+path
                beta = min(score, beta)

            pathList.append((score, [action] + path))

        '''
        same with minumax
        '''
        pathList = sorted(pathList, key=lambda x: x[0])

        return pathList[-1] if agentIndex == 0 else pathList[0]

    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, 0, 0, -float('inf'), float('inf'))[1][0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
