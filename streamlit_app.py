import streamlit as st
import numpy as np
import pickle
import pandas as pd
from Qlearning import ConnectFourEnv, QLearningAgent

st.set_page_config(page_title="connect four rl", layout="wide")

def draw_board(board):
    colors = {0: '⚪', 1: '🔴', -1: '🟡'}
    board_str = ""
    for row in board:
        board_str += " ".join([colors[int(cell)] for cell in row]) + "\n"
    board_str += " ".join([str(i) for i in range(7)])
    return board_str

@st.cache_data
def load_training_stats():
    try:
        with open('training_stats.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_agents(checkpoint=None):
    try:
        agent1 = QLearningAgent()
        agent2 = QLearningAgent()
        
        if checkpoint:
            agent1.load(f'agent1_checkpoint_{checkpoint}.pkl')
            agent2.load(f'agent2_checkpoint_{checkpoint}.pkl')
        else:
            agent1.load('agent1_final.pkl')
            agent2.load('agent2_final.pkl')
        
        return agent1, agent2
    except Exception as e:
        st.error(f"Error loading agents: {e}")
        return None, None

st.title("connect four reinforcement learning")
st.markdown("**MSML642 HW4 | Yaxita Amin**")

st.sidebar.header("agent selection")
use_checkpoint = st.sidebar.checkbox("use checkpoint instead of final agents")
checkpoint_episode = None

if use_checkpoint:
    checkpoint_episode = st.sidebar.selectbox(
        "select checkpoint episode",
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
        index=9
    )

tab1, tab2, tab3 = st.tabs(["training stats", "play against ai", "watch ai vs ai"])

# TAB 1: TRAINING STATS
with tab1:
    st.header("training performance over time")
    
    stats = load_training_stats()
    if stats:
        df = pd.DataFrame({
            'episode': stats['episodes'],
            'player 1 wins': stats['player1_wins'],
            'player 2 wins': stats['player2_wins'],
            'draws': stats['draws']
        })
        
        st.line_chart(df.set_index('episode'))
        
        total_games = len(stats['episodes']) * 100
        total_p1_wins = df['player 1 wins'].sum()
        total_p2_wins = df['player 2 wins'].sum()
        total_draws = df['draws'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("total games", f"{total_games:,}")
        col2.metric("player 1 wins", f"{total_p1_wins:,} ({100*total_p1_wins/total_games:.1f}%)")
        col3.metric("player 2 wins", f"{total_p2_wins:,} ({100*total_p2_wins/total_games:.1f}%)")
        col4.metric("draws", f"{total_draws:,} ({100*total_draws/total_games:.1f}%)")

# TAB 2: PLAY AGAINST AI
with tab2:
    st.header("play against the trained agent")
    st.markdown("you are 🔴 (player 1), ai is 🟡 (player 2)")
    
    agent1, agent2 = load_agents(checkpoint_episode)
    
    if agent1 and agent2:
        # Initialize session state for human play
        if 'play_moves' not in st.session_state:
            st.session_state.play_moves = []
        
        # NEW GAME BUTTON
        if st.button("🎮 NEW GAME", type="primary", use_container_width=True):
            st.session_state.play_moves = []
            st.rerun()
        
        # Reconstruct game state from move history
        game = ConnectFourEnv()
        for move in st.session_state.play_moves:
            game.make_move(move)
        
        # DEBUG INFO
        st.write(f"**DEBUG: Move list = {st.session_state.play_moves}**")
        st.write(f"**DEBUG: Current player = {game.current_player}**")
        
        # Display current board
        st.subheader("current board")
        st.text(draw_board(game.board))
        
        # Check game status
        winner = game.check_winner()
        valid_moves = game.get_valid_moves()
        game_over = (winner != 0) or (len(valid_moves) == 0)
        
        st.write(f"**Moves made: {len(st.session_state.play_moves)}**")
        
        # Show game result or continue playing
        if game_over:
            if winner == 1:
                st.success("✅ YOU WIN! 🎉")
            elif winner == -1:
                st.error("❌ AI WINS!")
            else:
                st.info("🤝 DRAW - BOARD FULL")
        else:
            # Game continues - show whose turn it is
            if game.current_player == 1:
                # HUMAN'S TURN
                st.info("⬇️ YOUR TURN - Click a column to drop your piece")
                cols = st.columns(7)
                for i in range(7):
                    with cols[i]:
                        if i in valid_moves:
                            # Button to make a move - ONLY appends move, nothing else
                            if st.button(f"↓ {i}", key=f"play_col_{i}", use_container_width=True):
                                st.session_state.play_moves.append(i)
                                st.rerun()
                        else:
                            st.button("FULL", disabled=True, use_container_width=True)
            else:
                # AI'S TURN
                st.warning("🤖 AI'S TURN - Click button to see AI move")
                if st.button("▶️ LET AI MOVE", type="primary", use_container_width=True):
                    # AI chooses move
                    ai_action = agent2.get_action(game.get_state(), valid_moves, training=False)
                    st.session_state.play_moves.append(ai_action)
                    st.rerun()

# TAB 3: WATCH AI VS AI
with tab3:
    st.header("watch ai agents play")
    
    agent1, agent2 = load_agents(checkpoint_episode)
    
    if agent1 and agent2:
        # Initialize session state for AI vs AI
        if 'ai_moves' not in st.session_state:
            st.session_state.ai_moves = []
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎮 NEW AI GAME", type="primary", use_container_width=True):
                st.session_state.ai_moves = []
                st.rerun()
        
        # Reconstruct game state
        game = ConnectFourEnv()
        for move in st.session_state.ai_moves:
            game.make_move(move)
        
        # Display board
        st.subheader("current board")
        st.text(draw_board(game.board))
        
        # Check game status
        winner = game.check_winner()
        valid_moves = game.get_valid_moves()
        game_over = (winner != 0) or (len(valid_moves) == 0)
        
        st.write(f"**Moves made: {len(st.session_state.ai_moves)}**")
        st.write(f"**Next player: {'🔴 Agent 1' if game.current_player == 1 else '🟡 Agent 2'}**")
        
        # Next move button
        with col2:
            if not game_over:
                if st.button("▶️ NEXT MOVE", use_container_width=True):
                    # Current player makes a move
                    if game.current_player == 1:
                        action = agent1.get_action(game.get_state(), valid_moves, training=False)
                    else:
                        action = agent2.get_action(game.get_state(), valid_moves, training=False)
                    st.session_state.ai_moves.append(action)
                    st.rerun()
        
        # Show result
        if game_over:
            if winner == 1:
                st.success("✅ Agent 1 (Red) Wins! 🔴")
            elif winner == -1:
                st.success("✅ Agent 2 (Yellow) Wins! 🟡")
            else:
                st.info("🤝 Draw - Board Full")
        
        # Move history in expandable section
        if st.session_state.ai_moves:
            with st.expander("📜 View Move History", expanded=False):
                for i, m in enumerate(st.session_state.ai_moves, 1):
                    player = "🔴 Agent 1" if i % 2 == 1 else "🟡 Agent 2"
                    st.write(f"Move {i}: {player} → Column {m}")

st.markdown("---")
st.markdown("*MSML642 HW4 | Yaxita Amin*")