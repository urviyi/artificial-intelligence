
from sample_players import DataPlayer
from isolation import Isolation, Agent, play
from random import choice as rndchoice
from copy import deepcopy
from math import log, sqrt
import random
import time


class TreeNode(object):
    
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.children = {}
        self.untried_actions = state.actions()
        self.N = 0
        self.uct = float('inf')
        self.Q = 1
        
    pass


class GameTree:
    def __init__(self, s, par_node=None, pre_action=None):
        self.parent = par_node
        self.pre_action = pre_action
        self.child = []
        self.untried_actions = s.actions()
        self.r = 0
        self.n = 0
        self.state = s
        self.player = s.player()
        self.uct = float('inf')
        self.result = s.utility(0)

    def __repr__(self):
        ratio = self.r / (self.n + 1)
        l = [str(e) for e in (self.pre_action, ''.join(self.state), self.r, self.n, str(ratio)[:5], str(self.uct)[:5])]
        return ' '.join(l)

    def update(self, v):
        self.n += 1
        if self.state.utility(0) < 0:
            self.r += 0
        elif self.state.utility(0) > 0:
            self.r += 1


class MCTS:
    def __init__(self, s):
        self.root = GameTree(s)
        self.game = Isolation() ## Initialize the game
        self.expand_node(self.root) ## Start the initial node expansion

    def run_mcts(self, board):
        self.__init__(board)
        start_time = time.time()
        iii = 0
        while time.time() - start_time < 0.010: # run for 15 milliseconds
            self.mcts_loop()
            iii += 1

    def ai_move(self):
        best_node, best_visits = None, 0
        for n in self.root.child:
            if n.n > best_visits: best_visits, best_node = n.n, n
        return best_node.pre_action

    def mcts_loop(self):
        node = self.node_selection(self.root)
        self.expand_node(node)
        if len(node.child) > 0:
            selected_node = rndchoice(node.child)
        else:
            selected_node = node
        v = self.simulation(deepcopy(selected_node.state))
        self.backpropagation(selected_node, v)

    def node_selection(self, node):
        if node.child:
            imax, vmax = 0, 0
            for i, n in enumerate(node.child):
                n.uct = MCTS.uct(n)
                v = n.uct
                if v > vmax:
                    imax, vmax = i, v
            selected = node.child[imax]
            return self.node_selection(selected)
        else:
            selected = node
            return selected

    def expand_node(self, node):
        if node.state.terminal_test() == False:
            for a in node.state.actions():
                state_after_action = node.state.result(a)
                node.child.append(GameTree(state_after_action, node, a))

    def simulation(self, s):
        if not s.terminal_test():
            actions = s.actions()
            a = rndchoice(actions)
            new_s = s.result(a)
            return new_s
        else:
            return s.utility(0)

    def backpropagation(self, node, v):
        node.update(v)
        if node.parent != None:
            self.backpropagation(node.parent, v)

    @staticmethod
    def uct(node):
        v = (node.r / (node.n + 1e-12)) + sqrt(2 * log(node.parent.n + 1) / (node.n + 1e-12))
        return v    



class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
        
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #import random
        #self.queue.put(random.choice(state.actions()))
        
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        
        #self.ai = MCTS(state)
        #self.ai.run_mcts(state)
        
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            #self.queue.put(self.minimax(state, depth=3))
            self.queue.put(self.monte_carlo_tree_search(state))
            #self.queue.put(self.principal_variation_search(state, depth=3))
            #self.queue.put(self.ai.ai_move())
           
            
        
            
    def principal_variation_search(self, state, depth):
        """ Implement principal variation search """
        
        """
        def pvs_min_max(self,state,depth,alpha,beta,color):
            if depth <=0 or state.terminal_test():
                return color*self.score(state)
            explored = []
            
            for a in state.actions():
                if a not in explored:
                    explored.append(a)
                    score = pvs_min_max(self, state.result(a),depth-1,alpha,beta,-color)
                    #score = -score
                else:
                    score = pvs_min_max(self, state.result(a),depth-1,alpha,alpha+1,-color)
                   # score = -score
                    if alpha<score and beta>score:
                        score = self.pvs_min_max(self, state.result(a),depth-1,alpha,beta,-color)
                        #score = -score
                if alpha<score:
                    alpha = score
                if alpha>=beta:
                    break
            return alpha
        
        alpha = float("-inf") 
        beta = float("inf")
        actions = state.actions()
        
        if actions:
            best_move = actions[0]
        else:
            best_move = None
            
        color = True
        
        v = -float("inf")
        
        for i, action in enumerate(actions):
            new_state = state.result(action)
            
            if i == 0:
                v = max(v, pvs_min_max(self, new_state, depth-1, alpha, beta, color))
            else:
                v = max(v, pvs_min_max(self, new_state, depth-1, alpha, alpha+1, color))
                if v>alpha:
                    v = max(v, pvs_min_max(self, new_state, depth-1, alpha, beta, color))
            if v > alpha:
                alpha = v
                best_move = action
        return best_move
        """

                    
        
    def monte_carlo_tree_search(self, s):
            
        def tree_policy(v):
            if v.state.terminal_test(): 
                return v
            while not v.state.terminal_test():
                print(v.state.actions())
                if len(v.untried_actions) > 0: 
                    return expand(v) 
                else:
                    v = best_child(v, 0)
            return v
        
        def expand(v):
            a = v.untried_actions.pop()
            next_state = v.state.result(a)
            next_v = v.child.append(GameTree(next_state, v, a))
            return next_v
        
        def uct(v):
            uct = (v.r / (v.n + 1e-12)) + sqrt(2 * log(v.parent.n + 1) / (v.n + 1e-12))
            return uct
            
        def best_child(v, c):
            imax, vmax = 0,0
            for i, next_v in enumerate(v.child):
                next_v.uct = uct(v)
                value = next_v.uct 
                if value > vmax:
                    imax, vmax = i, value 
            best_child = v.child[i]
            return best_child 
                
        def default_policy(s):
            while not s.terminal_test():
                s = s.result(random.choice(s.actions()))
            return s.utility(self.player_id)
        
        def backup(v, delta):
            while v.parent is not None:
                v.n = v.n + 1
                v.r = v.r + delta
                delta = -delta 
                v = v.parent
         
        v0 = GameTree(s)        
        start_time = time.time()        
        
        while time.time() - start_time < 0.010:
            vl = tree_policy(v0)
            delta = default_policy(vl.state)
            backup(vl, delta)
            best_child = best_child(v0,0) #return a node
        return best_child.pre_action
    
    
        
        
    def minimax(self, state, depth):
        """ Implement minimax with alpha-beta pruning """

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha: return value
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta: return value
                alpha = max(alpha, value)
            return value
        
        alpha = float("-inf") 
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            value = min_value(state.result(a), depth-1, alpha, beta)
            alpha = max(alpha, value)
            if value > best_score:
                best_score = value
                best_move = a
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
