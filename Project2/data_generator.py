import csv
import random
import ast
import pandas as pd
import os

# ==============================
# Team Evaluation Rules
# ==============================

def is_optimal_team(team):
    """
    Check if a team is optimal based on rules:
    1. No duplicate players
    2. All 5 basketball positions must be present (PG, SG, SF, PF, C)
    """
    # Rule 1: Team cannot have duplicate players
    players = [t[0] for t in team]
    if len(players) != len(set(players)):
        return 0

    # Rule 2: Must have all 5 unique positions
    roles = [t[1] for t in team]
    required_roles = {'PG', 'SG', 'SF', 'PF', 'C'} # Point guard, Shooting guard, Small Forward, Power Forward, Center
    team_roles = set(roles)
    
    # Check if we have exactly 5 unique roles and they match required positions
    if len(team_roles) != 5 or team_roles != required_roles:
        return 0

    return 1  # Passed all rules → optimal


####### TEAM GENERATION #######

def load_doubles(input_file):
    """Load player-role pairs from CSV file"""
    doubles = []
    try: # Dynamic paths handling for different environments
        if not os.path.isabs(input_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_file = os.path.join(script_dir, input_file)
        
        df = pd.read_csv(input_file)

        ###### Handle different CSV formats ###### - Claude AI helped
        for _, row in df.iterrows():
            if 'double' in df.columns:
                # If it's already in double format like "(player, role)"
                double_str = row['double'].strip("()")
                parts = [p.strip().strip("'\"") for p in double_str.split(",")]
                if len(parts) == 2:
                    doubles.append((parts[0], parts[1]))
            elif 'player_name' in df.columns and 'role' in df.columns:
                # If it's in separate columns
                doubles.append((row['player_name'], row['role']))
            else:
                print(f"Warning: Unrecognized CSV format in {input_file}")
                break
    except Exception as e:
        print(f"Error loading doubles from {input_file}: {e}")
    
    print(f"Loaded {len(doubles)} player-role pairs")
    
    # Print role distribution for debugging - Claude AI helped
    roles = [d[1] for d in doubles]
    role_counts = {}
    for role in roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    print("Role distribution:", role_counts)
    
    return doubles


def generate_teams(doubles, num_teams=1000):
    """Generate teams with a mix of optimal and non-optimal teams"""
    teams, labels = [], []
    
    # Separate players by role for easier optimal team generation
    players_by_role = {}
    for player, role in doubles:
        if role not in players_by_role:
            players_by_role[role] = []
        players_by_role[role].append((player, role))
    
    required_roles = ['PG', 'SG', 'SF', 'PF', 'C'] # Point guard, Shooting guard, Small Forward, Power Forward, Center
    
    # Check if we have players for all roles
    missing_roles = [role for role in required_roles if role not in players_by_role or len(players_by_role[role]) == 0]
    if missing_roles:
        print(f"Warning: Missing players for roles: {missing_roles}")
    
    optimal_count = 0
    
    # Synthetic Data Generation... generates 70% optimal teams and 30% non-optimal teams for better balance
    for i in range(num_teams):
        if i < num_teams * 0.7 and all(role in players_by_role and len(players_by_role[role]) > 0 for role in required_roles):
            # Generate optimal team (one player from each position)
            team = []
            for role in required_roles:
                if players_by_role[role]:
                    player = random.choice(players_by_role[role])
                    team.append(player)
            
            # Actually optimal
            if len(team) == 5 and is_optimal_team(team):
                teams.append(team)
                labels.append(1)
                optimal_count += 1
            else:
                # Fallback to random team
                team = random.sample(doubles, min(5, len(doubles)))
                teams.append(team)
                labels.append(is_optimal_team(team))
        else:
            # Generate non-optimal team (random selection)
            team = random.sample(doubles, min(5, len(doubles)))
            label = is_optimal_team(team)
            teams.append(team)
            labels.append(label)
            if label == 1:
                optimal_count += 1
    
    print(f"Generated {len(teams)} teams: {optimal_count} optimal, {len(teams) - optimal_count} non-optimal")
    return teams, labels


def save_dataset(teams, labels, output_file):
    """Convert teams + labels to DataFrame and save"""
    if not os.path.isabs(output_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if 'datasets' in output_file:
            base_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) != 'datasets' else script_dir
            output_file = os.path.join(base_dir, output_file)
        else:
            output_file = os.path.join(script_dir, output_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.DataFrame({
        "team": [str(t) for t in teams],
        "label": labels
    })
    df.to_csv(output_file, index=False)
    
    # Print label distribution
    label_counts = df['label'].value_counts()
    print(f"+++ Saved {len(teams)} teams to {output_file}")
    print(f"Label distribution: {dict(label_counts)}")


def build_dataset(input_file, output_file, num_teams=1000):
    """Build dataset from input file"""
    # Load player doubles
    doubles = load_doubles(input_file)
    
    if not doubles:
        print(f"Error: No valid player-role pairs loaded from {input_file}")
        return
    
    # Generate labeled teams
    teams, labels = generate_teams(doubles, num_teams)
    
    # Save dataset
    save_dataset(teams, labels, output_file)

if __name__ == "__main__":
    # Paths may be different if cloned from GitHub. There are several checks throughout the code to handle this.
    # Below is an example of this:

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
    
    # Create datasets directory if it doesn't exist
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Check which input file exists
    input_files = [
        os.path.join(DATASETS_DIR, "role_double.csv"),
        os.path.join(DATASETS_DIR, "role_data.csv")
    ]
    input_file = None
    
    for file in input_files:
        if os.path.exists(file):
            input_file = file
            print(f"Using input file: {os.path.basename(input_file)}")
            break
    
    if input_file is None:
        print("Error: No input file found. Please run preprocess.py first to generate role_double.csv")
        print("Or ensure one of these files exists: role_double.csv, top3_quad.csv, role_data.csv")
    else:
        # Generate training dataset
        print("Generating training dataset...")
        build_dataset(input_file, os.path.join(DATASETS_DIR, "train_dataset.csv"), num_teams=1000)
        
        print("\nGenerating testing dataset...")
        # Generate testing dataset
        build_dataset(input_file, os.path.join(DATASETS_DIR, "test_dataset.csv"), num_teams=200)