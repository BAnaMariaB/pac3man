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
from util import Stack
from util import Queue
from util import PriorityQueue

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
"""
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    stack_xy = Stack()
    visited = []
    path = []
    
    if problem.isGoalState(problem.getStartState()):
        return []
    
    stack_xy.push((problem.getStartState(),[]))
    
    while(True):
        if stack_xy.isEmpty():
            return []
        
        xy, path = stack_xy.pop()
        visited.append(xy)
        
        if problem.isGoalState(xy):
            return path
        
        successor = problem.getSuccessors(xy)
        if successor:
            for i in successor:
                if i[0] not in visited:
                    newPath = path + [i[1]]
                    stack_xy.push((i[0], newPath))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    que_xy = Queue()
    visited = []
    path = []
    
    if problem.isGoalState(problem.getStartState()):
        return []
    
    que_xy.push((problem.getStartState(),[]))
    
    while(True):
        if que_xy.isEmpty():
            return []
        
        xy, path = que_xy.pop()
        visited.append(xy)
        
        if problem.isGoalState(xy):
            return path
        
        successor = problem.getSuccessors(xy)
        
        if successor : 
            for i in successor:
                if i[0] not in visited and i[0] not in(state[0] for state in que_xy.list):
                    newPath = path +[i[1]]
                    que_xy.push((i[0],newPath))
        
        

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    que_xy = PriorityQueue()
    visited = [] 
    path = [] 

    if problem.isGoalState(problem.getStartState()):
        return []

    que_xy.push((problem.getStartState(),[]),0)

    while(True):
        if que_xy.isEmpty():
            return []

        xy,path = que_xy.pop() 
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        successor = problem.getSuccessors(xy)

        if successor:
            for i in successor:
                if i[0] not in visited and (i[0] not in (state[2][0] for state in que_xy.heap)):

                    new_Path = path + [i[1]]
                    priority = problem.getCostOfActions(new_Path)

                    que_xy.push((i[0],new_Path),priority)

                elif i[0] not in visited and (i[0] in (state[2][0] for state in que_xy.heap)):
                    
                    for state in que_xy.heap:
                        if state[2][0] == i[0]:
                            old_Priorirty = problem.getCostOfActions(state[2][1])

                    new_Priority = problem.getCostOfActions(path + [i[1]])
                    
                    if old_Priorirty > new_Priority:
                        new_Path = path + [i[1]]
                        que_xy.update((i[0],new_Path),new_Priority)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    que_xy = Priority_Queue_Function(problem,f)
    path = [] 
    visited = [] 

    if problem.isGoalState(problem.getStartState()):
        return []

    ele = (problem.getStartState(),[])
    que_xy.push(ele,heuristic)

    while(True):
        if que_xy.isEmpty():
            return []

        xy,path = que_xy.pop() 

        if xy in visited:
            continue

        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        successor = problem.getSuccessors(xy)

        if successor:
            for item in successor:
                if item[0] not in visited:

                    newPath = path + [item[1]] 
                    ele = (item[0],newPath)
                    que_xy.push(ele,heuristic)


class Priority_Queue_Function(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, problem, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction    
        PriorityQueue.__init__(self)        
        self.problem = problem
    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(self.problem,item,heuristic))

# Calculate f(n) = g(n) + h(n) #
def f(problem,state,heuristic):

    return problem.getCostOfActions(state[1]) + heuristic(state[0],problem)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
