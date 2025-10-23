import numpy as np
import pickle
import random
from collections import defaultdict

class ConnectFourEnv:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        
    def reset(self):
        # start fresh game with empty board
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        # converting board to tuple so we can use it as dict key
        # flatten turns 2d array into 1d which is easier to hash
        return tuple(self.board.flatten())
    
    def get_valid_moves(self):
        # check which columns arent full yet
        # top row being empty means column has space
        valid = []
        for col in range(self.cols):
            if self.board[0][col] == 0:
                valid.append(col)
        return valid
    
    def make_move(self, col):
        # handle invalid moves with penalty
        if col not in self.get_valid_moves():
            return self.get_state(), -10, True
        
        # drop piece in lowest available row (gravity effect)
        # start from bottom and work our way up
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        
        # check if game ended and calculate reward
        winner = self.check_winner()
        done = winner != 0 or len(self.get_valid_moves()) == 0
        
        # reward structure: win = 1, loss = -1, draw = -0.2 (penalize draws), continue = 0
        # penalizing draws encourages agents to play more aggressively
        if winner == self.current_player:
            reward = 1
        elif winner == -self.current_player:
            reward = -1
        elif done:
            reward = -0.2  # small penalty for draws to encourage winning
        else:
            reward = 0
        
        # switch to other player
        self.current_player *= -1
        return self.get_state(), reward, done
    
    def check_winner(self):
        # check all possible ways to win in connect four
        # need 4 CONSECUTIVE IDENTICAL pieces in a row: horizontal, vertical, or diagonal
        
        # check horizontal wins
        # sliding window of size 4 across each row
        for row in range(self.rows):
            for col in range(self.cols - 3):
                window = [self.board[row][col+i] for i in range(4)]
                # ALL 4 pieces must be the same AND not empty
                if window[0] != 0 and all(piece == window[0] for piece in window):
                    return window[0]
        
        # check vertical wins
        # sliding window of size 4 down each column
        for row in range(self.rows - 3):
            for col in range(self.cols):
                window = [self.board[row+i][col] for i in range(4)]
                if window[0] != 0 and all(piece == window[0] for piece in window):
                    return window[0]
        
        # check diagonal wins going down-right (\)
        # start from top-left and check diagonal patterns
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                window = [self.board[row+i][col+i] for i in range(4)]
                if window[0] != 0 and all(piece == window[0] for piece in window):
                    return window[0]
        
        # check diagonal wins going down-left (/)
        # start from top-right and check diagonal patterns
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                window = [self.board[row+i][col-i] for i in range(4)]
                if window[0] != 0 and all(piece == window[0] for piece in window):
                    return window[0]
        
        # no winner found
        return 0

class QLearningAgent:
    def __init__(self, learning_rate=0.15, discount_factor=0.95, 
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.99985):
        # hyperparameters for q learning
        # increased lr to 0.15 for faster learning
        # increased epsilon_min to 0.05 to maintain more exploration
        # adjusted decay rate for smoother epsilon reduction
        self.lr = learning_rate  # how fast we update q values
        self.gamma = discount_factor  # how much we care about future rewards
        self.epsilon = epsilon_start  # exploration rate starts high
        self.epsilon_min = epsilon_min  # minimum exploration to maintain
        self.epsilon_decay = epsilon_decay  # how fast we reduce exploration
        
        # q table stores state-action values
        # defaultdict automatically creates entries for new states
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def get_action(self, state, valid_moves, training=True):
        # epsilon greedy strategy: explore vs exploit
        if training and random.random() < self.epsilon:
            # exploration: try random moves to discover new strategies
            return random.choice(valid_moves)
        else:
            # exploitation: use what we learned to pick best move
            q_values = {move: self.q_table[state][move] for move in valid_moves}
            max_q = max(q_values.values())
            # if multiple moves have same q value pick randomly among them
            # this adds variety and prevents getting stuck
            best_moves = [move for move, q in q_values.items() if q == max_q]
            return random.choice(best_moves)
    
    def update_q_value(self, state, action, reward, next_state, valid_next_moves):
        # q learning update formula
        # Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]
        # this is temporal difference learning
        current_q = self.q_table[state][action]
        
        # find best possible future reward from next state
        if len(valid_next_moves) > 0:
            max_next_q = max([self.q_table[next_state][a] for a in valid_next_moves])
        else:
            # terminal state has no future rewards
            max_next_q = 0
        
        # calculate new q value using bellman equation
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        # gradually reduce exploration as agent learns
        # we want to explore more early on and exploit more later
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        # save q table to file so we dont lose training progress
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load(self, filename):
        # load previously trained q table
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), loaded)

def train_agent(num_episodes=15000, save_interval=1000):
    # main training loop where agents learn through self play
    # reduced to 15k episodes to prevent over-convergence to draws
    env = ConnectFourEnv()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    
    # track performance over time
    stats = {
        'player1_wins': [],
        'player2_wins': [],
        'draws': [],
        'episodes': [],
        'avg_game_length': [],
        'epsilon': []
    }
    
    # counters for current batch
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0
    games_in_batch = 0
    
    # track running metrics for early stopping
    recent_draw_rates = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # store transitions for both players
        # need to track (state, action, valid_moves) for proper q updates
        player1_history = []
        player2_history = []
        move_count = 0
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            # alternate between players
            if env.current_player == 1:
                action = agent1.get_action(state, valid_moves)
                # store current valid moves for accurate q update later
                player1_history.append((state, action, valid_moves.copy()))
            else:
                action = agent2.get_action(state, valid_moves)
                player2_history.append((state, action, valid_moves.copy()))
            
            next_state, reward, done = env.make_move(action)
            move_count += 1
            
            if not done:
                state = next_state
            else:
                # game ended, update q values for both players
                winner = env.check_winner()
                
                # determine final rewards based on outcome
                if winner == 1:
                    p1_wins += 1
                    final_reward_p1 = 1
                    final_reward_p2 = -1
                elif winner == -1:
                    p2_wins += 1
                    final_reward_p1 = -1
                    final_reward_p2 = 1
                else:
                    draws += 1
                    final_reward_p1 = -0.2  # penalty for draw
                    final_reward_p2 = -0.2  # penalty for draw
                
                # update q values for player 1
                # work backwards through game to propagate rewards
                for i, (s, a, vm) in enumerate(player1_history):
                    if i < len(player1_history) - 1:
                        # intermediate moves get 0 reward
                        next_s = player1_history[i + 1][0]
                        next_vm = player1_history[i + 1][2]
                        agent1.update_q_value(s, a, 0, next_s, next_vm)
                    else:
                        # final move gets actual game outcome
                        agent1.update_q_value(s, a, final_reward_p1, next_state, [])
                
                # update q values for player 2
                for i, (s, a, vm) in enumerate(player2_history):
                    if i < len(player2_history) - 1:
                        next_s = player2_history[i + 1][0]
                        next_vm = player2_history[i + 1][2]
                        agent2.update_q_value(s, a, 0, next_s, next_vm)
                    else:
                        agent2.update_q_value(s, a, final_reward_p2, next_state, [])
        
        # reduce exploration rate after each game
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        total_moves += move_count
        games_in_batch += 1
        
        # track stats every 100 episodes for visualization
        if (episode + 1) % 100 == 0:
            draw_rate = draws / games_in_batch
            recent_draw_rates.append(draw_rate)
            
            # keep only last 20 measurements for rolling average
            if len(recent_draw_rates) > 20:
                recent_draw_rates.pop(0)
            
            stats['episodes'].append(episode + 1)
            stats['player1_wins'].append(p1_wins)
            stats['player2_wins'].append(p2_wins)
            stats['draws'].append(draws)
            stats['avg_game_length'].append(total_moves / games_in_batch)
            stats['epsilon'].append(agent1.epsilon)
            
            print(f"episode {episode + 1}/{num_episodes} | p1: {p1_wins} | p2: {p2_wins} | draws: {draws} | avg moves: {total_moves/games_in_batch:.1f} | epsilon: {agent1.epsilon:.4f}")
            
            # early stopping if draws get too high
            if len(recent_draw_rates) >= 10:
                avg_recent_draws = sum(recent_draw_rates[-10:]) / 10
                if avg_recent_draws > 0.5 and episode > 5000:
                    print(f"\nwarning: high draw rate detected ({avg_recent_draws:.1%})")
                    print("stopping training to prevent defensive convergence")
                    break
            
            # reset counters for next batch
            p1_wins = 0
            p2_wins = 0
            draws = 0
            total_moves = 0
            games_in_batch = 0
        
        # save checkpoints periodically so we dont lose progress
        if (episode + 1) % save_interval == 0:
            agent1.save(f'agent1_checkpoint_{episode + 1}.pkl')
            agent2.save(f'agent2_checkpoint_{episode + 1}.pkl')
            print(f"checkpoint saved at episode {episode + 1}")
    
    # save final trained agents
    agent1.save('agent1_final.pkl')
    agent2.save('agent2_final.pkl')
    
    # save training stats for visualization
    with open('training_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\ntraining complete! q-table size: agent1={len(agent1.q_table)}, agent2={len(agent2.q_table)}")
    
    return agent1, agent2, stats

def play_game(agent1, agent2, render=False):
    # play single game between two agents
    # used for testing trained agents
    env = ConnectFourEnv()
    state = env.reset()
    done = False
    
    while not done:
        if render:
            print("\n" + str(env.board))
        
        valid_moves = env.get_valid_moves()
        
        # agents take turns based on current player
        if env.current_player == 1:
            action = agent1.get_action(state, valid_moves, training=False)
        else:
            action = agent2.get_action(state, valid_moves, training=False)
        
        state, reward, done = env.make_move(action)
    
    if render:
        print("\n" + str(env.board))
        winner = env.check_winner()
        if winner == 1:
            print("player 1 wins!")
        elif winner == -1:
            print("player 2 wins!")
        else:
            print("its a draw!")
    
    return env.check_winner()

if __name__ == "__main__":
    print("starting training...")
    print("agents will learn connect four through self-play using q-learning")
    print("training with improved hyperparameters to prevent draw convergence")
    print("this might take a few minutes...\n")
    
    # train agents with improved parameters
    agent1, agent2, stats = train_agent(num_episodes=15000)
    print("\ntraining complete!")
    
    # test final agents performance
    print("\ntesting final agents over 100 games...")
    results = {'p1': 0, 'p2': 0, 'draw': 0}
    for _ in range(100):
        winner = play_game(agent1, agent2)
        if winner == 1:
            results['p1'] += 1
        elif winner == -1:
            results['p2'] += 1
        else:
            results['draw'] += 1
    
    print(f"final test results: {results}")
    print(f"win rates: p1={results['p1']}%, p2={results['p2']}%, draws={results['draw']}%")
    
    # if draw rate still too high, suggest using checkpoint
    if results['draw'] > 40:
        print("\nwarning: high draw rate detected in final agents")
        print("consider using an earlier checkpoint (around episode 8000-12000)")
        print("you can modify streamlit_app.py to load a checkpoint instead:")