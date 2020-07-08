# myTeam.py
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
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveAgent', second = 'OffensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  # The following line is an example only; feel free to change it.
  agent1 = eval(first)(firstIndex)
  agent2 = eval(second)(secondIndex)
  agent1.registerTeam([firstIndex, secondIndex])
  agent2.registerTeam([firstIndex, secondIndex])
  return [agent1, agent2]
##########
# Agents #
##########
class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    '''
    You should change this in your own agent.
    '''
    return random.choice(actions)
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    if len(bestActions) == 0:
      return random.choice(actions)
    return random.choice(bestActions)
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features
  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
class MinimaxAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    # our initializations
    self.depth = 2
    self.currDepth = 0
  def chooseAction(self, gameState):
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
    start = time.clock()
    if gameState.getAgentState(self.index).isPacman:
        lst = [a for a in self.getOpponents(gameState) if not gameState.getAgentState(a).isPacman]
    else:
        lst = [a for a in self.getOpponents(gameState) if gameState.getAgentState(a).isPacman]
    lst.insert(0, self.index)
    numAgents = len(lst)
    maxDepth = self.depth * numAgents
    def value(state, agent):
      if self.isWin(state, agent == self.index) or self.isLose(state,agent == self.index) or self.currDepth == maxDepth:
        v = self.evaluate(state, 'Stop')
        return v
      self.currDepth += 1
      if lst[agent] == self.index:
        return maxValue(state, agent % numAgents)
      return minValue(state, agent % numAgents)
    def maxValue(state, agent):
      pacmanActions = state.getLegalActions(lst[agent])
      pacmanSuccessors = [(state.generateSuccessor(lst[agent], action), action) for action in pacmanActions]
      successorVals = [(value(successor[0], (agent + 1) % numAgents), successor[1]) for successor in pacmanSuccessors]
      self.currDepth -= 1
      return max(successorVals, key=lambda t: t[0])
    def minValue(state, agent):
      agentActions = state.getLegalActions(lst[agent])
      agentSuccessors = [(state.generateSuccessor(lst[agent], action), action) for action in agentActions]
      successorVals = [(value(successor[0], (agent + 1) % numAgents), successor[1]) for successor in agentSuccessors]
      self.currDepth -= 1
      return mmin(successorVals, key=lambda t: t[0])
    rv = value(gameState, 0)
    rootVal = rv[1]
    self.currDepth = 0
    return rootVal
  def isWin(self, gameState, team):
    if team:
      return len(self.getFood(gameState).asList()) <= 2
    else:
      return len(self.getFoodYouAreDefending(gameState).asList()) <= 2
  def isLose(self, gameState, team):
     if team:
       return len(self.getFoodYouAreDefending(gameState).asList()) <= 2
     else:
       return len(self.getFood(gameState).asList()) <= 2
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features
  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
def mmin(l):
  if len(l) > 0:
    return min(l)
  else:
    return 0
class OffensiveAgent(ReflexCaptureAgent):
  def getFeatures(self, gameState, action):
    def checkRadius(pos, ghostPos, r):
      x1, y1 = pos
      x2, y2 = ghostPos
      if x1 == x2:
        return abs(y2 - y1) < (r + 1)
      elif y1 == y2:
        return abs(x2 - x1) < (r + 1)
      else:
        return False
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    enemies = [(i, successor.getAgentState(i)) for i in self.getOpponents(successor)]
    ghost_states = [a for a in enemies if not a[1].isPacman]
    noisy_dists = successor.getAgentDistances()
    pac_state = successor.getAgentState(self.index)
    pac_pos = pac_state.getPosition()
    food_list = self.getFood(successor).asList()
    result = 0
    if gameState.getAgentState(self.index).numCarrying:
        teammate = [t for t in self.getTeam(gameState) if t != self.index]
        mid = self.distancer.getDistance(gameState.getAgentState(teammate[0]).getPosition(), pac_pos)
        result = 100000000.0/(mid + 0.001)
        features['successorScore'] = result
        return features
    bool = 1 if pac_state.isPacman else 0
    min_ghost_dist = float("inf")
    ghost_scared = 0
    for ghost in ghost_states:
      ghost_scared += ghost[1].scaredTimer
      ghost_pos = ghost[1].getPosition()
      if not ghost_pos:
        ghost_pos = noisy_dists[ghost[0]]
      d = self.distancer.getDistance(pac_pos, ghost_pos)
      if ghost[1].scaredTimer == 0:
        d = 1.0/(d * 10000)
        if checkRadius(pac_pos, ghost_pos, 2):
          result += -9999999
      else:
        if checkRadius(pac_pos, ghost_pos, 2):
          result += 90
      min_ghost_dist = min(min_ghost_dist, d)
    food_dist = [self.getMazeDistance(pac_pos, food) for food in food_list]
    minFoodDist = mmin(food_dist) if len(food_list) else 1
    result += pac_state.numCarrying * 100.0
    result += 1000.0 * (successor.getScore() - gameState.getScore()) + 100.0 / minFoodDist + bool * 10.0 * min_ghost_dist + 100.0 * ghost_scared
    features['successorScore'] = result
    return features
  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  def asList(self, key=True):
    list = []
    for x in range(self.width):
      for y in range(self.height):
        if self[x][y] == key: list.append((x, y))
    return list
class DefensiveAgent(ReflexCaptureAgent):
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    result = 0
    if successor.getAgentState(self.index).isPacman:
      result += -999999999
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemies_pacman = [a for a in enemies if a.isPacman and a.getPosition()]
    agent_state = successor.getAgentState(self.index)
    agent_pos = agent_state.getPosition()
    noisy_dists = successor.getAgentDistances()
    food_list = self.getFoodYouAreDefending(successor).asList()
    if food_list:
      max_food_dist = sum([self.distancer.getDistance(agent_pos, f) for f in food_list])
    else:
      max_food_dist = 0.00000001
    if enemies_pacman:
        enemies_dist = mmin([self.distancer.getDistance(e.getPosition(), agent_pos) for e in enemies_pacman])
    else:
        enemies_dist = mmin([self.distancer.getDistance(e.getPosition(), agent_pos) for e in enemies])
    if enemies_dist == None:
      enemies_dist = mmin([noisy_dists[e] for e in enemies])
    if enemies_dist == None:
      enemies_dist = float("inf")
    enemies_val = enemies_dist if agent_state.scaredTimer > 0 else 1.0/enemies_dist
    if agent_state.scaredTimer > 0:
        if enemies_dist > agent_state.scaredTimer:
            result -= 10000
    result += 1.0/max_food_dist + 50.0*enemies_val -100000*len(enemies_pacman)
    if agent_pos in [e.getPosition() for e in enemies_pacman]:
      result += 99999
    features['successorScore'] = result
    return features
  # TODO: not chasing down the pacman until it gets really close
  def getWeights(self, gameState, action):
    return {'successorScore': 1.0, 'stop': -100, 'reverse': -2}
  def asList(self, key=True):
    list = []
    for x in range(self.width):
      for y in range(self.height):
        if self[x][y] == key: list.append((x, y))
    return list