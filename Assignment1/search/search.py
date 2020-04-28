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
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
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

    from util import Stack


    dfs_stack=Stack()
    now_state=problem.getStartState() # 현재 상태 저장
    dfs_stack.push((now_state,[])) # (현재 위치,현재 위치에 오기까지 경로) 튜플을 푸시
    is_visited=[now_state]# 방문체크 list




    while not dfs_stack.isEmpty():
        now_state=dfs_stack.pop()
        is_visited.append(now_state[0])

        if problem.isGoalState(now_state[0]):
            return now_state[1]

        for (location,direction,cost) in problem.getSuccessors(now_state[0]): # 다음 노드 탐색
            if location not in is_visited:
                dfs_stack.push((location, now_state[1]+[direction])) # 방문한적이 없는 node 면 stack 에 푸시


    return [] # 답안 존재 x



def breadthFirstSearch(problem):
    '''
    dfs 랑 자료구조만 다르고 로직은 동일
    '''
    from util import Queue

    bfs_q = Queue() # 큐 이용
    now_state =  problem.getStartState()
    bfs_q.push((now_state, []))
    is_visited = []


    while not bfs_q.isEmpty():
        now_state = bfs_q.pop()

        if now_state[0] in is_visited:
            continue
        else:
            is_visited.append(now_state[0])

        if problem.isGoalState(now_state[0]):
            return now_state[1]

        for (location, direction, cost) in problem.getSuccessors(now_state[0]):
            if location not in is_visited:
                bfs_q.push((location, now_state[1] + [direction]))

    return []

def uniformCostSearch(problem):

    from util import PriorityQueue

    pq=PriorityQueue() # 우선순위 큐 이용
    start_state = problem.getStartState()
    pq.push((start_state, [],0),0) # (현재 위치,[경로],누적 cost) 을 우선순위 큐에 저장
    is_visited = []


    while not pq.isEmpty():
        now_state = pq.pop()

        if now_state[0] in is_visited: #방문 체크
            continue
        else:
            is_visited.append(now_state[0])

        if problem.isGoalState(now_state[0]):
            return now_state[1]

        for location, direction, cost in problem.getSuccessors(now_state[0]): #다음 노드 탐색
            if location not in is_visited:
                pq.push((location, now_state[1] + [direction],now_state[2]+cost),now_state[2]+cost) # 방문 한적이 없는 노드면 경로와 누적 cost 를 저장

    return [] # 경로 못 찾음

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):

    '''
    h(n)!=0
    '''


    from util import PriorityQueue

    pq = PriorityQueue()
    start_state = problem.getStartState()
    pq.update((start_state, [],0),0) # (현재위치,[경로],누적 cost g(n))
    is_visited = [] # 방문한 노드 저장

    while not pq.isEmpty():
        now_state = pq.pop()

        if now_state[0] in is_visited: # 방문체크
            continue
        else:
            is_visited.append(now_state[0]) # 방문 안했으면 방문리스트에 추가

        if problem.isGoalState(now_state[0]): # 경로 찾음
            return now_state[1]

        for location, direction, cost in problem.getSuccessors(now_state[0]):
            if location not in is_visited: #방문한적 없는 노드 우선순위 큐에 넣음
                pq.update((location, now_state[1] + [direction], now_state[2] + cost), now_state[2] + cost+heuristic(location,problem)) #새로운 노드 저장, 새로운 노드의 우선순위 기준은 g(n)+h(n)

    return [] # 경로 못찾음


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
