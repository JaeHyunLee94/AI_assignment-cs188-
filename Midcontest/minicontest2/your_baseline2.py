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


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffenseAgent', second='DefenseAgent'):
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

class ContestAgent(CaptureAgent):
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
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()

        self.ischasing = False
        self.onDefense = True
        self.EatNum = 0
        self.myTeam = 'Red' if self.red else 'Blue'

        self.x = gameState.data.layout.width // 2 if self.myTeam == 'Blue' else (gameState.data.layout.width - 1) // 2

        self.bridgeList = [(self.x, y) for y in range(1, gameState.data.layout.height - 1) if not self.walls[self.x][y]]

        # self.debugDraw(self.bridgeList, [0, 1, 0])

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = gameState.generateSuccessor(self.index, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

            return bestAction

        next_action = random.choice(bestActions)
        next_state = gameState.generateSuccessor(self.index, next_action)
        next_pos = next_state.getAgentPosition(self.index)

        # ghost 상태이면 ...
        if not next_state.getAgentState(self.index).isPacman:
            self.EatNum = 0
            self.onDefense = True
            self.ischasing = False

        self.EatNum += len(self.getFood(gameState).asList()) - len(self.getFood(next_state).asList())

        # self.debug_parameter()
        return next_action

    def evaluate(self, gameState, a):
        '''
    evaluate gameState after action a
    '''

        feature = self.getFeatures(gameState, a)
        weight = self.getWeights(gameState, a)

        return feature * weight

    def getFeatures(self, gameState, action):
        util.raiseNotDefined()

    def getWeights(self, gameState, action):
        util.raiseNotDefined()

    def debug_parameter(self):

        print(f'EatNum: {self.EatNum}')
        print(f'isChasing: {self.ischasing}')
        print(f'bridgeLsit: {self.bridgeList}')
        print(f'onDefense: {self.onDefense}')
        print(f'myTeam: {self.myTeam}')


class OffenseAgent(ContestAgent):

    def getFeatures(self, gameState, action):

        from math import log2
        nextState = gameState.generateSuccessor(self.index, action)
        nowPos = gameState.getAgentPosition(self.index)

        nextPos = nextState.getAgentPosition(self.index)
        now_foodList = self.getFood(gameState).asList()
        next_foodList = self.getFood(nextState).asList()

        agent_food_dis_list = [self.getMazeDistance(nextPos, food) for food in next_foodList]

        food_you_eat = len(next_foodList) * 10 - self.getScore(gameState)

        opponent = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invader = [pac for pac in opponent if pac.isPacman and pac.getPosition() != None]
        ghost = [ghost for ghost in opponent if not ghost.isPacman and ghost.getPosition() != None]

        feature = util.Counter()

        # remainFood  : 줄어들었으면 높은점수!!
        feature['successorScore'] = len(now_foodList) - len(next_foodList)

        # 가장 가까운 food 와의 거리를 줄이기!!

        feature['minDisFood'] = min(agent_food_dis_list)

        # threat score : ghost 와의 거리가 멀면 높은 점수, 거리가 가까우면  높은 페널티

        threat_dis = [self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in ghost if
                      self.getMazeDistance(nextPos, ghost.getPosition()) < 7]
        threat_score = 0

        if len(threat_dis) > 0:
            self.ischasing = True
            threat_score = min([log2(val * 0.001 + 0.0000001) for val in threat_dis])
        else:
            self.ischasing = False

        feature['ghostDiatance'] = threat_score

        # howmucheatfood 적당히 먹었으면 빠지기

        go_home = [self.getMazeDistance(homepoint, nextPos) for homepoint in self.bridgeList]

        if not self.ischasing:
            feature['goHome'] = 0
        elif self.ischasing and self.EatNum == 0:
            feature['goHome'] = 0
        else:
            feature['goHome'] = min(go_home)

        return feature

    def getWeights(self, gameState, aciton):
        weight = {
            'successorScore': 100,
            'minDisFood': -1,
            'ghostDiatance': 1,
            'goHome': -3
        }
        return weight


class DefenseAgent(ContestAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
