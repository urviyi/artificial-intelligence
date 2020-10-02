
from sample_players import DataPlayer
from isolation import Isolation, Agent, play
from random import choice as rndchoice
from copy import deepcopy
from math import log, sqrt
import random
import time
from isolation import DebugState



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
        
        print('In get_action(), state received:')
        debug_board = DebugState.from_state(state)
        print(debug_board)
              
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            #self.queue.put(self.minimax(state, depth=3))
            self.queue.put(self.monte_carlo_tree_search(state))
                        
        
    def monte_carlo_tree_search(self, s):
        
        def score(state):
            if state.terminal_test():
                if state.utility(self.player_id) > 0:
                    return 1
                else:
                    return -1
            else:
                return 0
            
        def tree_policy(v, c):
            if v.state.terminal_test(): return v           
            if len(v.state.actions()) != len(v.child): 
                expand(v) 
            v = best_child(v, c)
            return v
        
        def expand(v):
            a = random.choice([a for a in v.state.actions() if a not in [child.state.actions() for child in v.child]])
            next_state = v.state.result(a)
            next_v = v.child.append(self.GameTree(next_state, v, a))
            return next_v
        
        def uct(v, c):
            uct = (v.r / (v.n + 1e-12)) + c*sqrt(2 * log(v.parent.n + 1) / (v.n + 1e-12))
            return uct
            
        def best_child(v, c):
            if (v.state.terminal_test()): return v
            
            best_child = v.child[-1]
            uctmax = uct(best_child, c)
            imax = len(v.child) - 1
            
            for i, child in enumerate(v.child):
                i_uct = uct(child, c)
                if i_uct > uctmax:
                    imax, uctmax = i,i_uct
                    best_child = v.child[i]
                return best_child
                
        def default_policy(s):
            if not (s.terminal_test()):
                a = random.choice(s.actions())
                s = s.result(a)
            return score(s)
            
        def backup(v, delta):
            while v.parent is not None:
                v.n += 1
                v.r += delta
                delta *= -1 
                v = v.parent
                              
        v0 = self.GameTree(s)        
        start_time = time.time()        
        c = 0.90
        
        while time.time() - start_time < 0.010:
            
            vl = tree_policy(v0, c)
            vl2 = tree_policy(vl, c)
            vl3 = tree_policy(vl2, c)
            
            delta = default_policy(vl3.state)
            backup(vl3, delta)
            
        return best_child(v0,c).pre_action
    
    
    
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
