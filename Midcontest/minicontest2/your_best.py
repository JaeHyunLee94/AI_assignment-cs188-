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
import numpy as np


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

        self.ischasing = False  # 쫒기고 있는지에 대한 flag
        self.onDefense = True  # defense 상태인가에 대한 flag
        self.EatNum = 0  # 먹은 음식개수
        self.myTeam = 'Red' if self.red else 'Blue'
        self.counter_ondefense = 0
        self.threshold = 4

        self.x = gameState.data.layout.width // 2 if self.myTeam == 'Blue' else (gameState.data.layout.width - 1) // 2
        self.y = gameState.data.layout.height // 2

        self.bridgeList = [(self.x, y) for y in range(1, gameState.data.layout.height - 1) if
                           not self.walls[self.x][y]]  # 가장 가까운 탈출로

        # self.debugDraw(self.bridgeList, [0, 1, 0])

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]
        '''
        가장 상태 점수가 큰 action 고르기!! 
        '''
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

        if (self.myTeam == 'Red' and self.x < next_pos[0]) or (self.myTeam == 'Blue' and self.x - 1 > next_pos[0]):
            self.counter_ondefense = 0
        else:
            self.counter_ondefense += 1

        # ghost 상태이면 변수들 재 설정
        if not next_state.getAgentState(self.index).isPacman:
            self.EatNum = 0
            self.onDefense = True
            self.ischasing = False
        else:
            self.onDefense = False
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

        nextState = gameState.generateSuccessor(self.index, action)
        nowPos = gameState.getAgentPosition(self.index)

        nextPos = nextState.getAgentPosition(self.index)
        now_foodList = self.getFood(gameState).asList()
        next_foodList = self.getFood(nextState).asList()
        now_capsule = self.getCapsules(gameState)
        next_capsule = self.getCapsules(nextState)

        agent_food_dis_list = [self.getMazeDistance(nextPos, food) for food in next_foodList]


        next_opponent = [gameState.getAgentState(i) for i in self.getOpponents(nextState)]


        next_invader = [pac for pac in next_opponent if pac.isPacman and pac.getPosition() != None]

        next_ghost = [ghost for ghost in next_opponent if
                      not ghost.isPacman and ghost.getPosition() != None]

        next_scared_ghost = [ghost for ghost in next_opponent if
                             not ghost.isPacman and ghost.getPosition() != None and ghost.scaredTimer > 0]

        feature = util.Counter()

        # remainFood  : 줄어들었으면 높은점수!!
        if self.ischasing:
            feature['successorScore'] = 0
        else:
            feature['successorScore'] = len(now_foodList) - len(next_foodList)

        # 가장 가까운 food 와의 거리를 줄이기!!

        feature['minDisFood'] = min(agent_food_dis_list)

        # ghostDistance : 거리가 멀면 높은 점수, 거리가 좁으면 높은 페널티

        next_ghost_dis = [self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in next_ghost]

        threat_dis = [self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in next_ghost if
                      self.getMazeDistance(nextPos, ghost.getPosition()) < self.threshold]
        if self.onDefense:
            if len(threat_dis) > 0:
                feature['ghostDiatance'] = min(threat_dis)
            else:
                feature['ghostDiatance'] = 5

        else:
            if len(threat_dis) > 0:
                self.ischasing = True
                feature['ghostDiatance'] = min(threat_dis)
            else:
                feature['ghostDiatance'] = 5
                self.ischasing = False

        '''
        goHome: 언제 아군side 로 돌아갈건지 결정
        
        '''

        go_home = [self.getMazeDistance(homepoint, nextPos) for homepoint in self.bridgeList]

        if not self.ischasing:
            if self.EatNum == 0: # 도망칠 필요 없음
                feature['goHome'] = 0
            elif self.EatNum >= 3 and len(next_ghost_dis) > 0 and min(next_ghost_dis) < 6:  # 슬슬 튀어야함
                feature['goHome'] = 5 * min(go_home)
            elif self.EatNum >= 5:  # 충분히 많이 먹음
                feature['goHome'] = 5 * min(go_home)

        elif self.ischasing and self.EatNum == 0: # 도망치기보단 더 먹는게 나음
            feature['goHome'] = 0
        else:
            feature['goHome'] = min(go_home) # 집쪽으로 도망쳐야함



        # Ghost 상태 여도 상대 pacman 바로 죽일수 있으면 죽이기
        feature['supportDefense'] = 1
        if self.onDefense:
            feature['supportDefense'] = -len(next_invader)


        # kill scared_ghost

        feature['killScared'] = 2
        if len(next_scared_ghost) > 0:
            self.ischasing = False
            feature['killScared'] = - len(next_scared_ghost)


        next_capsule_dis = [self.getMazeDistance(nextPos, c) for c in next_capsule]
        if len(next_capsule_dis) > 0:
            feature['eatCapsule'] = 0.1 * min(next_capsule_dis)
        else:
            feature['eatCapsule'] = (len(next_capsule) - len(now_capsule)) * 100

        # dead check
        if self.getMazeDistance(nextPos, self.start) < 3:
            feature['dead'] = 9999999999
        else:
            feature['dead'] = 0

        if action == Directions.STOP: feature['stop'] = 1 # 매우 높은 확률로 stop 은 좋은 선택이 아님
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: feature['reverse'] = 1

        return feature

    def getWeights(self, gameState, aciton):
        weight = {
            'successorScore': 100,
            'minDisFood': -2,
            'ghostDiatance': 100, # 멀면 멀수록 높은 점수
            'goHome': -3,
            'isStuck': 1,
            'supportDefense': 1000,
            'killScared': -1.5, #줄어들수록 높은 점수
            'eatCapsule': -10,
            'dead': -1,
            'stop': -1000,
            'reverse': -5,

        }
        return weight

    # eat capsule


class DefenseAgent(ContestAgent):
    food_target = 0

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)

        nextState = successor.getAgentState(self.index)
        nextPos = nextState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if nextState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        enemies_pos = [i.getPosition() for i in enemies]
        enemies_dis = [self.getMazeDistance(nextPos, e) for e in enemies_pos]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        go_home = [self.getMazeDistance(homepoint, nextPos) for homepoint in self.bridgeList] # 적당히 혼란을 줬으면 아군쪽으로 돌아가야함

        nowFood = self.getFood(gameState).asList()
        nextFood = self.getFood(successor).asList()

        food_list = sorted(nextFood, key=lambda x: self.getMazeDistance(self.bridgeList[0], x)) # 아래쪽을 return point 로 잡음

        if len(food_list) > 0:
            features['minDisfood'] = -self.getMazeDistance(nextPos, food_list[0])

        features['successorScore'] = -len(nextFood)
        features['doOffense'] = 0
        if len(invaders) == 0 and min(enemies_dis) > 8: #침입자가 없고 거리가 멀면 공격해도 됌
            features['doOffense'] = 2
            features['invaderDistance'] = 0

        if self.EatNum > 1: # 한개 먹으면 돌아오기
            features['goHome'] = min(go_home)

        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(nextPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            features['goHome'] = min(go_home)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -10000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'goHome': -3, 'successorScore': 1000, 'doOffense': 1000, 'minDisFood': 5
                }
