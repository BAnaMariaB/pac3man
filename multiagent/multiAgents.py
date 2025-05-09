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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food      = newFood.asList()
        capsules  = successorGameState.getCapsules()
        count     = 0.0

        for i in food:
            d = manhattanDistance(newPos, i)
            if d == 0:
                count += 10.0
            else:
                count += 1.0 / (d + 1)

        for cap in capsules:
            d = manhattanDistance(newPos, cap)
            count += 5.0 / (d + 1)

        for ghostState in newGhostStates:
            gd = manhattanDistance(newPos, ghostState.getPosition())
            
            if ghostState.scaredTimer > 0:
                count += 4.0 / (gd + 1)
                
            else:
                if gd <= 1:
                    return -float('inf')
                  
                count -= 5.0 / (gd + 1)

        return successorGameState.getScore() + count


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
              
                if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                  return self.evaluationFunction(gameState)
                
                if agent == 0:  
                  return max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                
                else: 
                  nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                  
                  if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                  
                  if nextAgent == 0:
                    depth += 1
                
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
 
        maximum = float("-inf")
        action = Directions.WEST
        
        for agentState in gameState.getLegalActions(0):
            utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
         # max function
        def maximizer(agent, depth, game_state, a, b): 
            v = float("-inf")
            
            for newState in game_state.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                
                if v > b:
                    return v
                
                a = max(a, v)
            return v
          
        # min function
        def minimizer(agent, depth, game_state, a, b):  
            v = float("inf")
            next_agent = agent + 1
            
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
                
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))
                
                if v < a:
                    return v
                
                b = min(b, v)
            
            return v

        def alphabetaprune(agent, depth, game_state, a, b):
            
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  
                return self.evaluationFunction(game_state)

            if agent == 0:  
                return maximizer(agent, depth, game_state, a, b)
            
            else:
                return minimizer(agent, depth, game_state, a, b)

        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        
        for agentState in gameState.getLegalActions(0):
            ghostValue = alphabetaprune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            
            if ghostValue > utility:
                utility = ghostValue
                action = agentState
            
            if utility > beta:
                return utility
            
            alpha = max(alpha, utility)

        return action


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
        
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):
      
      if depth is 0 or gameState.isLose() or gameState.isWin():
        return (action, self.evaluationFunction(gameState))

      if agentIndex is 0:
        return self.max_Value(gameState, action, depth, agentIndex)

      else:
        return self.exp_Value(gameState, action, depth, agentIndex)


    def max_Value(self, gameState, action, depth, agentIndex):
          
      best_action = ("max", -(float('inf')))
        
      for legal_action in gameState.getLegalActions(agentIndex):
        succ_action = None
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            
        if depth != self.depth * gameState.getNumAgents():
          succ_action = action
            
        else:
          succ_action = legal_action
            
        succ_value = self.expectimax(gameState.generateSuccessor(agentIndex, legal_action), succ_action,depth - 1, nextAgent)
        best_action = max(best_action, succ_value, key = lambda x:x[1])
          
          
      return best_action


    def exp_Value(self, gameState, action, depth, agentIndex):
      avg_score = 0
      legal_actions = gameState.getLegalActions(agentIndex)
      prob = 1.0/len(legal_actions)
        
      for legalAction in legal_actions:
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        bestAction = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),action, depth - 1, nextAgent)
        avg_score += bestAction[1] * prob
            
      return (action, avg_score)
        


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    min_Food_List = float('inf')
    
    for food in newFood:
      min_Food_List = min(min_Food_List, manhattanDistance(newPos, food))


    ghost_distance = 0
    
    for ghost in currentGameState.getGhostPositions():
      ghost_distance = manhattanDistance(newPos, ghost)
        
      if (ghost_distance < 2):
        return -float('inf')

    food_l = currentGameState.getNumFood()
    caps_l = len(currentGameState.getCapsules())

    food_l_mult = 950050
    caps_l_mult = 10000
    food_distance_mult = 950
    additional_factors = 0
    
    if currentGameState.isLose():
      additional_factors -= 50000
    
    elif currentGameState.isWin():
      additional_factors += 50000

    return 1.0/(food_l + 1) * food_l_mult + ghost_distance + \
           1.0/(min_Food_List + 1) * food_distance_mult + \
           1.0/(caps_l + 1) * caps_l_mult + additional_factors


# Abbreviation
better = betterEvaluationFunction

