
from sample_players import DataPlayer
import random
import time


class TreeNode(object):
    
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.children = {}
        self.untried_actions = state.actions()
        self.N = 1
        self.Q = 1
        
    pass



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
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            #self.queue.put(self.minimax(state, depth=3))
            #self.queue.put(self.monte_carlo_tree_search(state))
            self.queue.put(self.principal_variation_search(state, depth=3))
            
                        
            
    def principal_variation_search(self, state, depth):
        """ Implement principal variation search """
        
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
                    
                    
            
            
        

        
    def monte_carlo_tree_search(self, s):
            
        def tree_policy(v):
            if v.state.terminal_test: return v
            
            if len(v.untried_actions) >= 0: 
                return expand(v)
            else:
                v = best_child(v, 0)
            return v
        
        def expand(v):
            a = v.untried_actions.pop()
            next_state = v.state.result(a)
            next_v = TreeNode(next_state)
            next_v.parent = v
            v.children[a] = next_v
            return next_v
        
        def best_child(v, c):
            result_dict = {}
            for a, next_v in v.children.items():
                result_dict[next_v] = next_v.Q/next_v.N
            return max(result_dict, key=result_dict.get)
                
        def default_policy(s):
            while not s.terminal_test():
                s = s.result(random.choice(s.actions()))
            return s.utility(self.player_id)
        
        def backup(v, delta):
            while v is not None:
                v.N = v.N+1
                v.Q = v.Q+delta
                delta = -delta 
                v = v.parent
                
        v0 = TreeNode(s)
        start_time = time.time()
        end_time = start_time + 100 # ms
        while time.time() < end_time:
            vi = tree_policy(v0)
            delta = default_policy(s)
            backup(vi, delta)
        return best_child(v0, 0)
        
        
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

        #return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))


    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)    
    