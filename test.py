import pickle
import random
from collections import defaultdict
import numpy as np
import pygame
import sys

# Import from your training script
from Qlearning import ConnectFourEnv, QLearningAgent

# Pygame constants
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 650
ROWS = 6
COLS = 7
SQUARE_SIZE = WINDOW_WIDTH // COLS
CIRCLE_RADIUS = SQUARE_SIZE // 2 - 5

# Colors
BLUE = (0, 102, 204)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

def analyze_agent_policy(agent, num_samples=1000):
    """
    Analyze how often agents exploit vs explore at different epsilon levels
    and check for policy stability
    """
    state_action_counts = defaultdict(list)
    
    for state_idx, (state, actions) in enumerate(agent.q_table.items()):
        if state_idx >= num_samples:
            break
        
        if not actions:
            continue
        
        q_values = list(actions.values())
        max_q = max(q_values)
        
        greedy_actions = sum(1 for q in q_values if q == max_q)
        all_actions = len(actions)
        
        confidence = max_q - (sum(q_values) / len(q_values))
        
        state_action_counts['avg_actions_per_state'].append(all_actions)
        state_action_counts['greedy_concentration'].append(greedy_actions / all_actions)
        state_action_counts['avg_confidence'].append(abs(confidence))
    
    if state_action_counts['avg_actions_per_state']:
        avg_actions = sum(state_action_counts['avg_actions_per_state']) / len(state_action_counts['avg_actions_per_state'])
        avg_concentration = sum(state_action_counts['greedy_concentration']) / len(state_action_counts['greedy_concentration'])
        avg_conf = sum(state_action_counts['avg_confidence']) / len(state_action_counts['avg_confidence'])
        
        print(f"\n=== Agent Policy Analysis ===")
        print(f"Average actions per state: {avg_actions:.2f}")
        print(f"Average greedy action concentration: {avg_concentration:.2%}")
        print(f"Average action value confidence: {avg_conf:.4f}")
        print(f"Total unique states learned: {len(agent.q_table)}")
        
        return {
            'avg_actions': avg_actions,
            'concentration': avg_concentration,
            'confidence': avg_conf
        }

def test_agent_opening(agent1, agent2, num_games=30):
    """
    Test how agents perform when starting from different columns
    to understand opening strategy
    """
    opening_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    
    for col in range(7):
        for test_game in range(num_games):
            env = ConnectFourEnv()
            state = env.reset()
            done = False
            
            valid_moves = env.get_valid_moves()
            if col not in valid_moves:
                continue
            
            state, _, done = env.make_move(col)
            
            while not done:
                valid_moves = env.get_valid_moves()
                if env.current_player == 1:
                    action = agent1.get_action(state, valid_moves, training=False)
                else:
                    action = agent2.get_action(state, valid_moves, training=False)
                
                state, _, done = env.make_move(action)
            
            winner = env.check_winner()
            if winner == 1:
                opening_stats[col]['wins'] += 1
            elif winner == -1:
                opening_stats[col]['losses'] += 1
            else:
                opening_stats[col]['draws'] += 1
    
    print(f"\n=== Opening Move Analysis ({num_games} games per column) ===")
    print("Col | Wins | Losses | Draws | Win%")
    print("-" * 35)
    for col in range(7):
        stats = opening_stats[col]
        total = stats['wins'] + stats['losses'] + stats['draws']
        if total > 0:
            win_pct = stats['wins'] / total * 100
            print(f" {col}  | {stats['wins']:3d} | {stats['losses']:3d}   | {stats['draws']:3d}  | {win_pct:5.1f}%")

def draw_board(screen, board):
    """Draw the Connect Four board with pygame"""
    # Draw blue background
    screen.fill(BLUE)
    
    # Draw board grid and pieces
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, BLUE, (col * SQUARE_SIZE, row * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE // 2), CIRCLE_RADIUS)
            
            # Draw pieces
            if board[row][col] == 1:
                pygame.draw.circle(screen, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE // 2), CIRCLE_RADIUS - 4)
            elif board[row][col] == -1:
                pygame.draw.circle(screen, YELLOW, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE // 2), CIRCLE_RADIUS - 4)

def draw_top_buttons(screen):
    """Draw column selection buttons at top"""
    for col in range(COLS):
        pygame.draw.rect(screen, GRAY, (col * SQUARE_SIZE, 0, SQUARE_SIZE, SQUARE_SIZE))
        pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, 0, SQUARE_SIZE, SQUARE_SIZE), 2)

def play_human_vs_agent_pygame(agent, human_is_player1=True):
    """
    Play Connect Four against trained agent with Pygame visualization
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Connect Four: Human vs AI Agent")
    font_small = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)
    
    env = ConnectFourEnv()
    state = env.reset()
    done = False
    message = ""
    message_timer = 0
    
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(60)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN and not done:
                x = event.pos[0]
                col = x // SQUARE_SIZE
                
                if col < COLS:
                    valid_moves = env.get_valid_moves()
                    
                    if (env.current_player == 1 and human_is_player1) or (env.current_player == -1 and not human_is_player1):
                        if col in valid_moves:
                            state, reward, done = env.make_move(col)
                            message = f"Player {'1 (Red)' if env.current_player == -1 else '2 (Yellow)'} played column {col}"
                            message_timer = 60
                        else:
                            message = "Invalid move! Column is full."
                            message_timer = 60
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    env = ConnectFourEnv()
                    state = env.reset()
                    done = False
                    message = "Game reset!"
                    message_timer = 60
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        # Agent's turn
        if not done and ((env.current_player == -1 and human_is_player1) or (env.current_player == 1 and not human_is_player1)):
            valid_moves = env.get_valid_moves()
            action = agent.get_action(state, valid_moves, training=False)
            state, reward, done = env.make_move(action)
            message = f"Agent {'(Red)' if env.current_player == -1 else '(Yellow)'} played column {action}"
            message_timer = 120
        
        # Draw everything
        draw_board(screen, env.board)
        draw_top_buttons(screen)
        
        # Draw instructions
        instruction_text = font_small.render("Click column to play | SPACE to reset | ESC to exit", True, WHITE)
        screen.blit(instruction_text, (10, WINDOW_HEIGHT - 30))
        
        # Draw player info
        if human_is_player1:
            player_text = font_small.render("You: RED (Player 1) | AI: YELLOW (Player 2)", True, WHITE)
        else:
            player_text = font_small.render("You: YELLOW (Player 2) | AI: RED (Player 1)", True, WHITE)
        screen.blit(player_text, (10, WINDOW_HEIGHT - 60))
        
        # Draw game status
        if done:
            winner = env.check_winner()
            if winner == 1:
                status_text = font_large.render("🎉 Player 1 (Red) Wins!", True, RED)
            elif winner == -1:
                status_text = font_large.render("🎉 Player 2 (Yellow) Wins!", True, YELLOW)
            else:
                status_text = font_large.render("It's a Draw!", True, WHITE)
            screen.blit(status_text, (WINDOW_WIDTH // 2 - 150, 10))
        else:
            current_player = "RED (Player 1)" if env.current_player == 1 else "YELLOW (Player 2)"
            current_text = font_small.render(f"Current: {current_player}", True, WHITE)
            screen.blit(current_text, (WINDOW_WIDTH - 250, 10))
        
        # Draw message
        if message_timer > 0:
            msg = font_small.render(message, True, WHITE)
            screen.blit(msg, (10, WINDOW_HEIGHT - 90))
            message_timer -= 1
        
        pygame.display.flip()

def agent_vs_agent_visual(agent1, agent2, speed=1):
    """
    Visualize two agents playing against each other
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Agent vs Agent - Connect Four")
    font_small = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)
    
    env = ConnectFourEnv()
    state = env.reset()
    done = False
    move_count = 0
    frame_counter = 0
    
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(10 * speed)  # Control game speed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        if not done:
            valid_moves = env.get_valid_moves()
            
            if env.current_player == 1:
                action = agent1.get_action(state, valid_moves, training=False)
            else:
                action = agent2.get_action(state, valid_moves, training=False)
            
            state, reward, done = env.make_move(action)
            move_count += 1
        
        # Draw board
        draw_board(screen, env.board)
        draw_top_buttons(screen)
        
        # Draw status
        status = f"Agent 1 (Red) vs Agent 2 (Yellow) - Move: {move_count}"
        status_text = font_small.render(status, True, WHITE)
        screen.blit(status_text, (10, WINDOW_HEIGHT - 30))
        
        if done:
            winner = env.check_winner()
            if winner == 1:
                result_text = font_large.render("Agent 1 (Red) Wins!", True, RED)
            elif winner == -1:
                result_text = font_large.render("Agent 2 (Yellow) Wins!", True, YELLOW)
            else:
                result_text = font_large.render("Draw!", True, WHITE)
            screen.blit(result_text, (WINDOW_WIDTH // 2 - 150, 10))
        
        pygame.display.flip()

if __name__ == "__main__":
    print("Loading trained agents...")
    
    # Load the trained agents
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    
    try:
        agent1.load('agent1_final.pkl')
        agent2.load('agent2_final.pkl')
        print("✓ Agents loaded successfully!\n")
    except FileNotFoundError:
        print("✗ Error: Could not find agent1_final.pkl or agent2_final.pkl")
        print("Make sure you've run the training script first.")
        exit()
    
    # Run console analyses
    print("=" * 50)
    print("AGENT ANALYSIS")
    print("=" * 50)
    
    print("\nAgent 1 Statistics:")
    analyze_agent_policy(agent1)
    
    print("\nAgent 2 Statistics:")
    analyze_agent_policy(agent2)
    
    test_agent_opening(agent1, agent2, num_games=30)
    
    # Menu
    print("\n" + "=" * 50)
    print("VISUALIZATION MENU")
    print("=" * 50)
    print("1. Play against Agent (Human vs AI)")
    print("2. Watch Agents Play (Agent vs Agent)")
    print("3. Exit")
    print("=" * 50)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        player_choice = input("Play as Player 1 (Red)? (y/n): ").lower()
        is_p1 = player_choice == 'y'
        print("Starting game... (ESC to exit, SPACE to reset)")
        play_human_vs_agent_pygame(agent1, human_is_player1=is_p1)
    
    elif choice == "2":
        speed_choice = input("Game speed (1=normal, 2=fast, 0.5=slow): ").strip()
        try:
            speed = float(speed_choice)
        except:
            speed = 1
        print("Starting visualization... (ESC to exit)")
        agent_vs_agent_visual(agent1, agent2, speed=speed)
    
    elif choice == "3":
        print("Goodbye!")
    
    else:
        print("Invalid choice!")