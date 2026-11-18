import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout
st.set_page_config(page_title="SupeRR Selector Dashboard", layout="wide")

@st.cache_data
def load_and_process_data():
    # Load data
    try:
        bbb = pd.read_csv('ball_by_ball.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'ball_by_ball.csv' is in the same directory.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bbb = bbb.sort_values(by=['match_id', 'innings', 'innings_ball_num'])

    # Core Metrics
    bbb['total_runs'] = bbb['runs'].fillna(0) + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)
    bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'wides', 'Overstep', 'noballs', 'wide', 'noball'])
    bbb['is_legal_val'] = bbb['is_legal'].astype(int)
    bbb['is_wicket_val'] = bbb['is_wicket'].astype(int)

    # Match Context
    bbb['innings_runs'] = bbb.groupby(['match_id', 'innings'])['total_runs'].cumsum()
    bbb['innings_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket_val'].cumsum()
    bbb['innings_legal_balls'] = bbb.groupby(['match_id', 'innings'])['is_legal_val'].cumsum()

    bbb['runs_at_start'] = bbb.groupby(['match_id', 'innings'])['innings_runs'].shift(1).fillna(0)
    bbb['wickets_at_start'] = bbb.groupby(['match_id', 'innings'])['innings_wickets'].shift(1).fillna(0)
    bbb['legal_balls_at_start'] = bbb.groupby(['match_id', 'innings'])['innings_legal_balls'].shift(1).fillna(0)

    # Target & Run Rates
    inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['innings_runs'].max().reset_index()
    inn1_totals.rename(columns={'innings_runs': 'target_runs'}, inplace=True)
    inn1_totals['target_runs'] += 1
    bbb = bbb.merge(inn1_totals, on='match_id', how='left')

    bbb['balls_remaining'] = 120 - bbb['legal_balls_at_start']
    bbb['runs_required'] = bbb['target_runs'] - bbb['runs_at_start']
    bbb['rrr'] = np.where(bbb['balls_remaining'] > 0, bbb['runs_required'] / (bbb['balls_remaining'] / 6), 0)
    bbb['crr'] = np.where(bbb['legal_balls_at_start'] > 0, bbb['runs_at_start'] / (bbb['legal_balls_at_start'] / 6), 0)

    # Partnerships
    bbb['partnership_id'] = bbb.groupby(['match_id', 'innings'])['is_wicket_val'].cumsum().shift(1).fillna(0)
    bbb['partnership_runs'] = bbb.groupby(['match_id', 'innings', 'partnership_id'])['total_runs'].cumsum().shift(1).fillna(0)
    bbb['partnership_balls'] = bbb.groupby(['match_id', 'innings', 'partnership_id'])['is_legal_val'].cumsum().shift(1).fillna(0)

    # --- DEFINE BATTING PRESSURE ---
    bbb['is_batter_pressure'] = False
    cond_rrr = (bbb['innings'] == 2) & (bbb['rrr'] > 10)
    cond_wkts = (bbb['innings'] == 2) & (bbb['wickets_at_start'] >= 5)
    cond_clutch = (bbb['innings'] == 2) & (bbb['overid'] >= 16) & (bbb['runs_required'] < 45) & (bbb['runs_required'] > 0)
    cond_collapse = (bbb['innings'] == 1) & (bbb['overid'] <= 6) & (bbb['wickets_at_start'] >= 3)
    cond_death = (bbb['innings'] == 1) & (bbb['overid'] >= 16)
    bbb.loc[cond_rrr | cond_wkts | cond_clutch | cond_collapse | cond_death, 'is_batter_pressure'] = True

    # --- DEFINE BOWLING PRESSURE ---
    bbb['is_bowler_pressure'] = False
    cond_defend = (bbb['innings'] == 2) & (bbb['overid'] >= 16) & (bbb['rrr'] <= 12)
    cond_pp_contain = (bbb['overid'] <= 6) & (bbb['crr'] > 9)
    cond_break_ptnr = (bbb['partnership_runs'] >= 50) & (bbb['partnership_balls'] >= 24)
    bbb.loc[cond_defend | cond_pp_contain | cond_break_ptnr, 'is_bowler_pressure'] = True

    # --- CALCULATE BATTER STATS ---
    batter_stats = bbb[bbb['is_legal']].groupby(['striker_name', 'is_batter_pressure']).agg(
        runs=('runs', 'sum'), balls=('is_legal_val', 'sum')
    ).reset_index()
    batter_piv = batter_stats.pivot(index='striker_name', columns='is_batter_pressure', values=['runs', 'balls']).fillna(0)
    batter_piv.columns = [f"{col[0]}_{col[1]}" for col in batter_piv.columns]
    
    if 'runs_False' not in batter_piv: batter_piv['runs_False'] = 0
    if 'balls_False' not in batter_piv: batter_piv['balls_False'] = 0
    if 'runs_True' not in batter_piv: batter_piv['runs_True'] = 0
    if 'balls_True' not in batter_piv: batter_piv['balls_True'] = 0

    batter_piv['sr_base'] = (batter_piv['runs_False'] / batter_piv['balls_False']) * 100
    batter_piv['sr_press'] = (batter_piv['runs_True'] / batter_piv['balls_True']) * 100
    batter_piv['sr_diff'] = batter_piv['sr_press'] - batter_piv['sr_base']
    batter_piv = batter_piv.reset_index()

    # --- CALCULATE BOWLER STATS ---
    bbb['bowler_runs'] = np.where(bbb['extras_type'].isin(['legbyes', 'byes', 'Penalty']), 0, bbb['runs'] + bbb['extras1_runs'].fillna(0))
    bowler_stats = bbb.groupby(['bowler_name', 'is_bowler_pressure']).agg(
        runs_c=('bowler_runs', 'sum'), balls_b=('is_legal_val', 'sum')
    ).reset_index()
    bowler_piv = bowler_stats.pivot(index='bowler_name', columns='is_bowler_pressure', values=['runs_c', 'balls_b']).fillna(0)
    bowler_piv.columns = [f"{col[0]}_{col[1]}" for col in bowler_piv.columns]

    if 'runs_c_False' not in bowler_piv: bowler_piv['runs_c_False'] = 0
    if 'balls_b_False' not in bowler_piv: bowler_piv['balls_b_False'] = 0
    if 'runs_c_True' not in bowler_piv: bowler_piv['runs_c_True'] = 0
    if 'balls_b_True' not in bowler_piv: bowler_piv['balls_b_True'] = 0

    bowler_piv['econ_base'] = (bowler_piv['runs_c_False'] / bowler_piv['balls_b_False']) * 6
    bowler_piv['econ_press'] = (bowler_piv['runs_c_True'] / bowler_piv['balls_b_True']) * 6
    bowler_piv['econ_diff'] = bowler_piv['econ_base'] - bowler_piv['econ_press']
    bowler_piv = bowler_piv.reset_index()

    return bbb, batter_piv, bowler_piv

# Load data
st.sidebar.title("SupeRR Selector")
st.sidebar.markdown("Identify the Most Reliable Players Under Pressure")
bbb, batters_df, bowlers_df = load_and_process_data()

if bbb.empty:
    st.stop()

# Filter for qualified players for rankings
qual_batters = batters_df[(batters_df['balls_False'] >= 100) & (batters_df['balls_True'] >= 50)].copy()
top_10_batters = qual_batters.sort_values('sr_diff', ascending=False).head(10)

qual_bowlers = bowlers_df[(bowlers_df['balls_b_False'] >= 120) & (bowlers_df['balls_b_True'] >= 60)].copy()
top_10_bowlers = qual_bowlers.sort_values('econ_diff', ascending=False).head(10)

# Sidebar Navigation
nav = st.sidebar.radio("Navigation", ["Top 10 Rankings", "Player Search (Dropdown)"])

if nav == "Top 10 Rankings":
    st.title("🏆 Top 10 Reliable Players Under Pressure")
    
    # --- BATTERS SECTION (TOP) ---
    st.header("🏏 Top 10 Batters")
    st.markdown("**Metric:** Strike Rate Improvement in Pressure Situations")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='sr_diff', y='striker_name', data=top_10_batters, palette='viridis', ax=ax)
    ax.set_xlabel('Strike Rate Increase')
    ax.set_ylabel('')
    st.pyplot(fig)
    
    display_batters = top_10_batters[['striker_name', 'sr_base', 'sr_press', 'sr_diff']].round(2)
    display_batters.columns = ['Batter', 'Baseline SR', 'Pressure SR', 'SR Improvement']
    st.dataframe(display_batters.set_index('Batter'), use_container_width=True)

    st.divider() # Adds a nice horizontal line between the two sections

    # --- BOWLERS SECTION (BOTTOM) ---
    st.header("🎯 Top 10 Bowlers")
    st.markdown("**Metric:** Economy Rate Improvement (Fewer Runs Conceded)")
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='econ_diff', y='bowler_name', data=top_10_bowlers, palette='magma', ax=ax2)
    ax2.set_xlabel('Economy Rate Decrease (Runs Saved/Over)')
    ax2.set_ylabel('')
    st.pyplot(fig2)
    
    display_bowlers = top_10_bowlers[['bowler_name', 'econ_base', 'econ_press', 'econ_diff']].round(2)
    display_bowlers.columns = ['Bowler', 'Baseline Econ', 'Pressure Econ', 'Econ Improvement']
    st.dataframe(display_bowlers.set_index('Bowler'), use_container_width=True)

elif nav == "Player Search (Dropdown)":
    st.title("🔍 Player Search & Analytics")
    
    # Create unified player list
    all_batters = batters_df['striker_name'].unique().tolist()
    all_bowlers = bowlers_df['bowler_name'].unique().tolist()
    all_players = sorted(list(set(all_batters + all_bowlers)))
    
    selected_player = st.selectbox("Select a Player to view their Pressure Profile:", all_players)
    
    st.divider()
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("🏏 Batting Profile")
        player_bat = batters_df[batters_df['striker_name'] == selected_player]
        
        if not player_bat.empty and (player_bat['balls_False'].values[0] > 0 or player_bat['balls_True'].values[0] > 0):
            st.metric("Baseline Strike Rate", f"{player_bat['sr_base'].values[0]:.2f}", 
                      f"{int(player_bat['balls_False'].values[0])} balls faced", delta_color="off")
            
            diff = player_bat['sr_diff'].values[0]
            st.metric("Pressure Strike Rate", f"{player_bat['sr_press'].values[0]:.2f}", 
                      f"{diff:.2f} SR difference", delta_color="normal")
                      
            # Simple Bar Chart for Batter
            fig_bat, ax_bat = plt.subplots(figsize=(4,3))
            sns.barplot(x=['Baseline', 'Pressure'], 
                        y=[player_bat['sr_base'].values[0], player_bat['sr_press'].values[0]], 
                        palette=['#1f77b4', '#ff7f0e'], ax=ax_bat)
            ax_bat.set_ylabel("Strike Rate")
            st.pyplot(fig_bat)
        else:
            st.info(f"{selected_player} has no significant batting data.")

    with colB:
        st.subheader("🎯 Bowling Profile")
        player_bowl = bowlers_df[bowlers_df['bowler_name'] == selected_player]
        
        if not player_bowl.empty and (player_bowl['balls_b_False'].values[0] > 0 or player_bowl['balls_b_True'].values[0] > 0):
            st.metric("Baseline Economy Rate", f"{player_bowl['econ_base'].values[0]:.2f}", 
                      f"{int(player_bowl['balls_b_False'].values[0])} balls bowled", delta_color="off")
            
            diff = player_bowl['econ_diff'].values[0]
            # Inverse delta color because lower economy is better
            st.metric("Pressure Economy Rate", f"{player_bowl['econ_press'].values[0]:.2f}", 
                      f"{-diff:.2f} Econ difference", delta_color="inverse")
            
            # Simple Bar Chart for Bowler
            fig_bowl, ax_bowl = plt.subplots(figsize=(4,3))
            sns.barplot(x=['Baseline', 'Pressure'], 
                        y=[player_bowl['econ_base'].values[0], player_bowl['econ_press'].values[0]], 
                        palette=['#1f77b4', '#d62728'], ax=ax_bowl)
            ax_bowl.set_ylabel("Economy Rate")
            st.pyplot(fig_bowl)
        else:
            st.info(f"{selected_player} has no significant bowling data.")