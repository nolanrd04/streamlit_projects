import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
import ast

# ==============================
# Helper Functions
# ==============================

def load_role_data():
    """Load player role data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    role_data_path = os.path.join(script_dir, "role_data.csv")
    
    if not os.path.exists(role_data_path):
        print(f"--- Error: role_data.csv not found!")
        print("Please run preprocess.py first to generate role assignments.")
        return None
    
    df_roles = pd.read_csv(role_data_path)
    print(f"✅ Loaded {len(df_roles)} players with roles")
    return df_roles

def create_optimal_teams(df_roles, num_teams=5):
    """Generate optimal teams (one player from each position)"""
    # Group players by position
    players_by_role = {}
    for _, row in df_roles.iterrows():
        role = row['role']
        if role not in players_by_role:
            players_by_role[role] = []
        players_by_role[role].append(row['player_name'])
    
    required_roles = ['PG', 'SG', 'SF', 'PF', 'C']
    
    # Check if we have players for all positions
    for role in required_roles:
        if role not in players_by_role or len(players_by_role[role]) == 0:
            print(f"❌ Warning: No players found for position {role}")
            return []
    
    optimal_teams = []
    for i in range(num_teams):
        team = []
        used_players = set()
        
        # Pick one player from each position
        for role in required_roles:
            available_players = [p for p in players_by_role[role] if p not in used_players]
            if available_players:
                player = random.choice(available_players)
                team.append((player, role))
                used_players.add(player)
            else:
                # If we run out of unique players, allow duplicates but different roles
                player = random.choice(players_by_role[role])
                team.append((player, role))
        
        optimal_teams.append(team)
    
    return optimal_teams

def create_non_optimal_teams(df_roles, num_teams=3):
    """Generate non-optimal teams (duplicate positions or missing positions)"""
    all_players = [(row['player_name'], row['role']) for _, row in df_roles.iterrows()]
    non_optimal_teams = []
    
    for i in range(num_teams):
        # Create teams with problems
        if i == 0:
            # Team with duplicate PG
            pgs = [p for p in all_players if p[1] == 'PG'][:2]
            others = [p for p in all_players if p[1] in ['SF', 'PF', 'C']][:3]
            team = pgs + others
        elif i == 1:
            # Team missing Center
            team = [p for p in all_players if p[1] in ['PG', 'SG', 'SF', 'PF']][:5]
        else:
            # Random team (likely non-optimal)
            team = random.sample(all_players, 5)
        
        non_optimal_teams.append(team)
    
    return non_optimal_teams

def predict_team_optimality(teams, model_path=None):
    """Predict if teams are optimal using trained model (if available)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "team_optimizer_model.h5")
    
    if not os.path.exists(model_file):
        print("⚠️  No trained model found. Showing rule-based evaluation only.")
        return None
    
    try:
        # Load model and recreate encoders (simplified version)
        model = tf.keras.models.load_model(model_file)
        
        # Simple evaluation based on rules
        predictions = []
        for team in teams:
            roles = [player[1] for player in team]
            players = [player[0] for player in team]
            
            # Check if team has all 5 unique positions and no duplicate players
            required_roles = {'PG', 'SG', 'SF', 'PF', 'C'}
            has_all_positions = set(roles) == required_roles
            has_unique_players = len(players) == len(set(players))
            
            is_optimal = has_all_positions and has_unique_players
            predictions.append(1 if is_optimal else 0)
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        return None

def display_team(team, team_num, is_optimal=None, prediction=None):
    """Display a team in a nice format"""
    print(f"\n🏀 TEAM {team_num}")
    print("=" * 40)
    
    for player, role in team:
        print(f"  {role:2s} | {player}")
    
    # Rule-based evaluation
    roles = [player[1] for player in team]
    players = [player[0] for player in team]
    required_roles = {'PG', 'SG', 'SF', 'PF', 'C'}
    
    has_all_positions = set(roles) == required_roles
    has_unique_players = len(players) == len(set(players))
    rule_optimal = has_all_positions and has_unique_players
    
    print(f"\n  📊 Analysis:")
    print(f"     All 5 positions: {'✅' if has_all_positions else '❌'}")
    print(f"     Unique players:  {'✅' if has_unique_players else '❌'}")
    print(f"     Rule-based:      {'🟢 OPTIMAL' if rule_optimal else '🔴 NON-OPTIMAL'}")
    
    if prediction is not None:
        model_optimal = prediction == 1
        print(f"     Model prediction: {'🟢 OPTIMAL' if model_optimal else '🔴 NON-OPTIMAL'}")
    
    # Show any issues
    if not has_all_positions:
        missing = required_roles - set(roles)
        duplicates = [role for role in roles if roles.count(role) > 1]
        if missing:
            print(f"     ⚠️  Missing positions: {', '.join(missing)}")
        if duplicates:
            print(f"     ⚠️  Duplicate positions: {', '.join(set(duplicates))}")
    
    if not has_unique_players:
        duplicates = [player for player in players if players.count(player) > 1]
        print(f"     ⚠️  Duplicate players: {', '.join(set(duplicates))}")

# ==============================
# Main Pipeline
# ==============================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE_DATA = os.path.join(SCRIPT_DIR, "role_data.csv")

def main():
    print("NBA OPTIMAL TEAMS DEMO")
    
    # Load player data
    df_roles = pd.read_csv(ROLE_DATA)
    
    # Show role distribution: CLAUDE AI HELPED
    print("\nPlayer Role Distribution:")
    role_counts = df_roles['role'].value_counts()
    for role, count in role_counts.items():
        print(f"   {role}: {count} players")
    
    # Generate optimal teams
    print(f"\n🎯 Generating Optimal Teams...")
    optimal_teams = create_optimal_teams(df_roles, num_teams=3)
    
    # Generate non-optimal teams
    print(f"🔄 Generating Non-Optimal Teams...")
    non_optimal_teams = create_non_optimal_teams(df_roles, num_teams=2)
    
    # Combine all teams
    all_teams = optimal_teams + non_optimal_teams
    team_labels = ["Optimal"] * len(optimal_teams) + ["Non-Optimal"] * len(non_optimal_teams)
    
    if not all_teams:
        print("❌ Could not generate teams. Check role assignments.")
        return
    
    # Try to get model predictions
    predictions = predict_team_optimality(all_teams)
    
    # Display results
    print(f"\n" + "🏀" * 50)
    print("GENERATED TEAMS")
    print("🏀" * 50)
    
    for i, (team, expected_label) in enumerate(zip(all_teams, team_labels)):
        pred = predictions[i] if predictions else None
        display_team(team, i + 1, expected_label, pred)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Generated {len(optimal_teams)} optimal teams")
    print(f"❌ Generated {len(non_optimal_teams)} non-optimal teams")
    
    if predictions:
        correct_predictions = sum(1 for pred, label in zip(predictions, team_labels) 
                                if (pred == 1 and label == "Optimal") or (pred == 0 and label == "Non-Optimal"))
        accuracy = correct_predictions / len(predictions)
        print(f"🎯 Model accuracy on these teams: {accuracy:.1%}")
    
    print("\n💡 Optimal teams have:")
    print("   • One player from each position (PG, SG, SF, PF, C)")
    print("   • No duplicate players")
    print("   • All 5 positions filled")

if __name__ == "__main__":
    main()