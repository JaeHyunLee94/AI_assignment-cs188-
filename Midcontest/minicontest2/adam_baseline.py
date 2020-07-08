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
"""
Students' Names: Ali Ahmed
Contest Number: 2
Description of Bot: I have two agents (offensive and defensive). The Defensive one is relatively straightforward, it copys lots of the baseline team code but uses inference to detect where the attacker/invader is if we have a noisy reading by seeing which dots disappeared on the map in the previous gameState. The bot can also go on offense when there is no threat, and when it does it uses all of the offensive tactics that the offensive bot has (now updated to include being able to recognize and eat capsules. The defensive bot transforms into the offensive bot when it is in pacman mode. This is pretty key as it confuses the opponent bots.
The offensive bot is also a slightly improved version of the baselineTeam.py model. I add other basic features for the bot such as when to eat the capsule and when to avoid eating the pacman (if you're scared). Added a preprocessing step so the bot knows that it should prioritize moving towards the exitCol when it has eaten more than 3 pellets of food (that's the main key insight, infering to go back once you've eaten more than 3 pellets) and knowing which general direction to move in to get back.
"""
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions, Grid
import game
from util import nearestPoint, manhattanDistance
import distanceCalculator
#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]
##########
# Agents #
##########
class OffensiveAgent(CaptureAgent):
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
    self.foodNum = 0
    # self.pathToExit = []
    self.myTeam = ''
    self.exitCol = []
    self.walls = gameState.getWalls()
    self.prevActions = [None, None, None, None]
    # get what team the bot is on
    if self.getTeam(gameState)[0] % 2 == 0:
      # exit direction left
      self.myTeam = 'red'
    else:
      # exit direction right
      self.myTeam = 'blue'
    # find available exit column spaces
    if self.myTeam == 'blue':
      exitCol = (gameState.data.layout.width) // 2
    else:
      exitCol = (gameState.data.layout.width - 1) // 2
    for i in range(1, gameState.data.layout.height - 1):
      # self.debugDraw([((gameState.data.layout.width - 1) // 2, i)], [0, 1, 0])
      if not self.walls[exitCol][i]:
        self.exitCol.append((exitCol, i))
    # for entry in self.exitCol:
    #   self.debugDraw([entry], [0, 1, 0])
  # Follows from getSuccessor function of ReflexCaptureAgent
  def getSuccessor(self,gameState,action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  #Follows from chooseAction function of ReflexCaptureAgent
  def chooseAction(self,gameState):
    nextAction = None
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions,values) if v == maxValue]
    nextAction = random.choice(bestActions)
    # if "West" in actions:
    newGameState = self.getSuccessor(gameState, nextAction)
    if not newGameState.getAgentState(self.index).isPacman:
      self.foodNum = 0
    self.foodNum += len(self.getFood(gameState).asList()) - len(self.getFood(newGameState).asList())
    # Update previous actions, make sure the list doesn't get too big and cause error
    self.prevActions.append(nextAction)
    if len(self.prevActions) > 20:
      self.prevActions = self.prevActions[14:21]
    return nextAction
  #Follows from evaluate function of ReflexCaptureAgent
  def evaluate(self,gameState,action):
    features = self.getFeatures(gameState,action)
    weights = self.getWeights(gameState,action)
    return features * weights
  def getFeatures(self, gameState, action):
    # Start like getFeatures of OffensiveReflexAgent
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    # Get other variables for later use
    food = self.getFood(gameState)
    capsules = gameState.getCapsules()
    foodList = food.asList()
    walls = gameState.getWalls()
    x, y = gameState.getAgentState(self.index).getPosition()
    vx, vy = Actions.directionToVector(action)
    newx = int(x + vx)
    newy = int(y + vy)
    # Get set of invaders and defenders
    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    # ghosts
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    # attacking pacmen
    defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    # Check if pacman has stopped
    if action == Directions.STOP:
        features["stuck"] = 1.0
    if self.prevActions[-4] != None and (self.prevActions[-3] == Directions.REVERSE[self.prevActions[-4]]) and (
            self.prevActions[-4] == self.prevActions[-2]) and (
            self.prevActions[-3] == self.prevActions[-1]) and action == self.prevActions[-4]:
        features['repeatMovement'] = 1
    # Get ghosts close by
    for ghost in invaders:
        ghostpos = ghost.getPosition()
        ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
        if (newx, newy) == ghostpos:
            # Encounter a Normal Ghost
            if ghost.scaredTimer == 0:
                features["scaredGhosts"] = 0
                features["normalGhosts"] = 1
            else:
                # Encounter a Scared Ghost (still prioritize food)
                features["eatFood"] += 2
                features["eatGhost"] += 1
        elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer > 0):
            features["scaredGhosts"] += 1
        elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer == 0):
            features["normalGhosts"] += 1
    # How to act if scared or not scared
    if gameState.getAgentState(self.index).scaredTimer == 0:
        for ghost in defenders:
            ghostpos = ghost.getPosition()
            ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
            if (newx, newy) == ghostpos:
                features["eatInvader"] = 1
    else:
        for ghost in enemies:
            if ghost.getPosition() != None:
                ghostpos = ghost.getPosition()
                ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
                if (newx, newy) in ghostNeighbors or (newx, newy) == ghostpos:
                    features["eatInvader"] = -10
    # Get capsules when nearby
    for cx, cy in capsules:
        if newx == cx and newy == cy and successor.getAgentState(self.index).isPacman:
            features["eatCapsule"] = 1
    # When to eat
    if not features["normalGhosts"]:
        if food[newx][newy]:
            features["eatFood"] = 1.0
        if len(foodList) > 0:
            tempFood = []
            for food in foodList:
                food_x, food_y = food
                adjustedindex = self.index - self.index % 2
                check1 = food_y > (adjustedindex / 2) * walls.height / 3
                check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3
                if (check1 and check2):
                    tempFood.append(food)
            if len(tempFood) == 0:
                tempFood = foodList
        if len(foodList) > 0:
            mazedist = [self.getMazeDistance((newx, newy), food) for food in tempFood]
        else:
            mazedist = [None]
        if min(mazedist) is not None:
            walldimensions = walls.width * walls.height
            features["nearbyFood"] = float(min(mazedist)) / walldimensions
            # If we've eaten enough food, try and go to an exit route
    if self.foodNum >= 4:  # and (newx, newy) in self.pathToExit:
        # closestExit = self.pathToExit[-1]
        closestExit = self.exitCol[0]
        dist = self.getMazeDistance((newx, newy), closestExit)
        for entry in self.exitCol:
            if self.getMazeDistance((newx, newy), entry) < dist:
                closestExit = entry
                dist = self.getMazeDistance((newx, newy), entry)
        # features["pathOnExitRoute"] = 1
        normalized = manhattanDistance((0, 0), closestExit)
        features["closeToExitPos"] = manhattanDistance(closestExit, (newx, newy)) / float(normalized)
    return features

  def getWeights(self, gameState, action):
      return {'eatInvader': 5, 'teammateDist': 1.5, 'nearbyFood': -5, 'eatCapsule': 10,
              'normalGhosts': -300, 'eatGhost': 1.0, 'scaredGhosts': 0.1, 'stuck': -10, 'eatFood': 1,
              'pathOnExitRoute': 10, 'closeToExitPos': -15, 'repeatMovement': -1}

class DefensiveAgent(CaptureAgent):
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
          self.foodNum = 0
          # self.pathToExit = []
          self.myTeam = ''
          self.exitCol = []
          self.walls = gameState.getWalls()
          # get what team the bot is on
          if self.getTeam(gameState)[0] % 2 == 0:
              # exit direction left
              self.myTeam = 'red'
          else:
              # exit direction right
              self.myTeam = 'blue'
          # find available exit column spaces
          if self.myTeam == 'blue':
              exitCol = (gameState.data.layout.width) // 2
          else:
              exitCol = (gameState.data.layout.width - 1) // 2
          for i in range(1, gameState.data.layout.height - 1):
              # self.debugDraw([((gameState.data.layout.width - 1) // 2, i)], [0, 1, 0])
              if not self.walls[exitCol][i]:
                  self.exitCol.append((exitCol, i))
          # for entry in self.exitCol:
          #   self.debugDraw([entry], [0, 1, 0])

      def getSuccessor(self, gameState, action):
          successor = gameState.generateSuccessor(self.index, action)
          pos = successor.getAgentState(self.index).getPosition()
          if pos != util.nearestPoint(pos):
              return successor.generateSuccessor(self.index, action)
          else:
              return successor

      def evaluate(self, gameState, action):
          features = self.getFeatures(gameState, action)
          weights = self.getWeights(gameState, action)
          return features * weights

      def chooseAction(self, gameState):
          nextAction = None
          actions = gameState.getLegalActions(self.index)
          values = [self.evaluate(gameState, a) for a in actions]
          maxValue = max(values)
          bestActions = [a for a, v in zip(actions, values) if v == maxValue]
          nextAction = random.choice(bestActions)
          newGameState = self.getSuccessor(gameState, nextAction)
          if not newGameState.getAgentState(self.index).isPacman:
              self.foodNum = 0
          self.foodNum += len(self.getFood(gameState).asList()) - len(self.getFood(newGameState).asList())
          return nextAction

      def getFeatures(self, gameState, action):
          features = util.Counter()
          successor = self.getSuccessor(gameState, action)
          myState = successor.getAgentState(self.index)
          myPos = myState.getPosition()
          # computes whether we're on offense (-1) or defense (1)
          features['onDefense'] = 1
          if myState.isPacman:
              features['onDefense'] = -1
          # standard from baseline - more numInvaders, larger minimum distance, worse successor (-30000, -1500).
          # Note if the opponent is greater than or equal to 5 blocks away (manhattan distance) then we only get a noisy reading.
          enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
          invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
          features['numInvaders'] = len(invaders)
          if len(invaders) > 0:
              distsManhattan = [manhattanDistance(myPos, a.getPosition()) for a in invaders]
              dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
              # get a more exact reading if food disappears between turns
              if min(distsManhattan) >= 5:
                  prevGamestate = self.getPreviousObservation()
                  currGamestate = self.getCurrentObservation()
                  prevFood = self.getFood(prevGamestate).asList()
                  currFood = self.getFood(currGamestate).asList()
                  missingFood = list(set(currFood) - set(prevFood))
                  dists.extend([self.getMazeDistance(myPos, a) for a in missingFood])
                  features['invaderDistance'] = min(dists)
              else:
                  features['invaderDistance'] = min(dists)
          # standard from baseline - is the action was to stop (-400) or to go back / reverse (-250)
          if action == Directions.STOP: features['stop'] = 1
          rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
          if action == rev: features['reverse'] = 1
          # transform defense agent into offensive when scared OR when there are no invaders
          # ADJUST WHEN DONE WITH OFFENSIVE AGENT - ADD ALL CHARACTERISTICS HERE ALSO CONSIDER THE OFFENSE OR DEFENSE PLAY (ABOVE).
          if (successor.getAgentState(self.index).scaredTimer > 0):
              features['numInvaders'] = 0  # change all of the defense ones to 0
              if (features['invaderDistance'] <= 2): features['invaderDistance'] = 2
          # use the minimum noisy distance between our agent and their agent
          teamNums = self.getTeam(gameState)
          features['stayApart'] = self.getMazeDistance(gameState.getAgentPosition(teamNums[0]),
                                                       gameState.getAgentPosition(teamNums[1]))
          features['offenseFood'] = 0
          # IF THERE ARE NO INVADERS THEN GO FOR FOOD / REFLEX AGENT. I LIKE THIS. COPY THE OFFENSE CODE HERE AS WELL.
          if (len(invaders) == 0 and successor.getScore() != 0):
              features['onDefense'] = -1
              if len(self.getFood(successor).asList()) != 0:
                  features['offenseFood'] = min(
                      [self.getMazeDistance(myPos, food) for food in self.getFood(successor).asList()])
              features['foodCount'] = len(self.getFood(successor).asList())
              features['stayAprts'] += 2
              features['stayApart'] *= features['stayApart']
              # If we've eaten enough food, try and go to an exit route
          if self.foodNum >= 4 and myState.isPacman:  # and (newx, newy) in self.pathToExit:
              # closestExit = self.pathToExit[-1]
              x, y = gameState.getAgentState(self.index).getPosition()
              vx, vy = Actions.directionToVector(action)
              newx = int(x + vx)
              newy = int(y + vy)
              closestExit = self.exitCol[0]
              dist = self.getMazeDistance((newx, newy), closestExit)
              for entry in self.exitCol:
                  if self.getMazeDistance((newx, newy), entry) < dist:
                      closestExit = entry
                      dist = self.getMazeDistance((newx, newy), entry)
              # features["pathOnExitRoute"] = 1
              normalized = manhattanDistance((0, 0), closestExit)
              features["closeToExitPos"] = manhattanDistance(closestExit, (newx, newy)) / float(normalized)
          if myState.isPacman:
              walls = gameState.getWalls()
              x, y = gameState.getAgentState(self.index).getPosition()
              vx, vy = Actions.directionToVector(action)
              newx = int(x + vx)
              newy = int(y + vy)
              # Get ghosts close by
              for ghost in invaders:
                  ghostpos = ghost.getPosition()
                  ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
                  if (newx, newy) == ghostpos:
                      # Encounter a Normal Ghost
                      if ghost.scaredTimer == 0:
                          features["scaredGhosts"] = 0
                          features["normalGhosts"] = 1
                      else:
                          # Encounter a Scared Ghost (still prioritize food)
                          features["eatFood"] += 2
                          features["eatGhost"] += 1
                  elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer > 0):
                      features["scaredGhosts"] += 1
                  elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer == 0):
                      features["normalGhosts"] += 1
              capsules = gameState.getCapsules()
              for cx, cy in capsules:
                  if newx == cx and newy == cy and successor.getAgentState(self.index).isPacman:
                      features["eatCapsule"] = 1
          return features

      def getWeights(self, gameState, action):
          return {'foodCount': -20, 'offenseFood': -1, 'numInvaders': -30000, 'onDefense': 10, 'stayApart': 50,
                  'invaderDistance': -1500, 'stop': -400, 'reverse': -250, "closeToExitPos": -50, 'normalGhosts': -3000,
                  'eatInvader': 5, 'teammateDist': 1.5, 'nearbyFood': -5, 'eatCapsule': 10, 'eatGhost': 1.0,
                  'scaredGhosts': 0.1, 'stuck': -10, 'eatFood': 1, 'pathOnExitRoute': 10, 'repeatMovement': -1}
