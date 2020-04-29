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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFood_list = newFood.asList()
        ghost_dis = 20
        food_dis = 0
        mindis = float('inf')

        if len(newFood_list) == 0:
            mindis = -10000

        for food_pos in newFood_list:
            if manhattanDistance(food_pos, newPos) < mindis:
                mindis = manhattanDistance(food_pos, newPos)

        for ghostState in newGhostStates:
            # if manhattanDistance(ghostState.getPosition(),newPos) <3:
            ghost_dis += manhattanDistance(ghostState.getPosition(), newPos)
            if manhattanDistance(ghostState.getPosition(), newPos) < 3:
                ghost_dis = ghost_dis * 0.0001

        "*** YOUR CODE HERE ***"

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

        if now_depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), []

        numAgent = gameState.getNumAgents()
        legal_action = gameState.getLegalActions(agentIndex)

        pathList = []
        for action in legal_action:
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            score, path = self.minimax(nextGameState, now_depth + 1 if agentIndex == numAgent - 1 else now_depth,
                                       (agentIndex + 1) % numAgent)
            pathList.append((score, [action] + path))

        pathList = sorted(pathList, key=lambda x: x[0])

        return pathList[-1] if agentIndex == 0 else pathList[0]

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

        return self.minimax(gameState, 0, 0)[1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self, gameState, now_depth, agentIndex, alpha, beta):

        if now_depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), []

        numAgent = gameState.getNumAgents()
        legal_action = gameState.getLegalActions(agentIndex)

        pathList = []
        for action in legal_action:
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            score, path = self.alphabeta(nextGameState, now_depth + 1 if agentIndex == numAgent - 1 else now_depth,
                                         (agentIndex + 1) % numAgent, alpha, beta)
            if agentIndex == 0:
                if score > beta:
                    return score, [action]+path
                alpha = max(score, alpha)
            else:
                if score < alpha:
                    return score, [action]+path
                beta = min(score, beta)

            pathList.append((score, [action] + path))

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
