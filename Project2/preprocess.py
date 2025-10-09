import pandas as pd
import numpy as np
import os

# ==============================
# Role Assignment
# ==============================

def assign_roles(input_file="relevant_data.csv", output_file="role_data.csv"):
    """Assign basketball positions to players based on their stats and height"""
    # Dynamic path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(input_file):
        input_file = os.path.join(script_dir, input_file)
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Load dataset
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} players from {input_file}")
    
    # Handle missing values
    df = df.dropna(subset=['ast', 'pts', 'reb', 'player_height'])
    print(f"After removing players with missing key stats: {len(df)} players")

    # Compute percentiles for more robust role assignment
    ast_75 = df["ast"].quantile(0.75)
    ast_25 = df["ast"].quantile(0.25)
    pts_75 = df["pts"].quantile(0.75)
    pts_50 = df["pts"].quantile(0.50)
    reb_75 = df["reb"].quantile(0.75)
    reb_50 = df["reb"].quantile(0.50)
    height_25 = df["player_height"].quantile(0.25)
    height_50 = df["player_height"].quantile(0.50)
    height_75 = df["player_height"].quantile(0.75)

    print(f"Stats percentiles:")
    print(f"  Assists: 25th={ast_25:.1f}, 75th={ast_75:.1f}")
    print(f"  Points: 50th={pts_50:.1f}, 75th={pts_75:.1f}")
    print(f"  Rebounds: 50th={reb_50:.1f}, 75th={reb_75:.1f}")
    print(f"  Height: 25th={height_25:.1f}, 50th={height_50:.1f}, 75th={height_75:.1f}")

    roles = []
    role_counts = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    
    for idx, row in df.iterrows():
        # Default role
        role = "SF"
        
        height = row["player_height"]
        assists = row["ast"]
        points = row["pts"]
        rebounds = row["reb"]

        # Enhanced role assignment logic
        if assists >= ast_75 and height <= height_25:
            role = "PG"  # High assists, shorter players
        elif points >= pts_75 and height <= height_50:
            role = "SG"  # High scoring, guard-sized
        elif height >= height_75 and rebounds >= reb_75:
            role = "C"   # Tall and good rebounder
        elif height >= height_50 and rebounds >= reb_50:
            role = "PF"  # Medium-tall, decent rebounder
        else:
            role = "SF"  # Default for versatile players

        roles.append([row["player_name"], role])
        role_counts[role] += 1

    print(f"\nRole distribution:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} players")

    # Create final DataFrame
    df_roles = pd.DataFrame(roles, columns=["player_name", "role"])

    # Save to CSV
    df_roles.to_csv(output_file, index=False)
    print(f"✅ Roles assigned and saved to {output_file}")
    
    return df_roles


# ==============================
# Double Builder
# ==============================

def create_role_double(input_file="role_data.csv", output_file="role_double.csv"):
    """Create player-role tuples for team generation"""
    # Dynamic path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(input_file):
        input_file = os.path.join(script_dir, input_file)
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Load the role dataset
    df = pd.read_csv(input_file)
    
    print(f"Creating doubles from {len(df)} players")

    # Create 2-tuple strings for each player
    # Format: (player_name, role)
    doubles = df.apply(
        lambda row: f"({row['player_name']}, {row['role']})", axis=1
    )

    # Put into a single-column DataFrame
    df_double = pd.DataFrame(doubles, columns=["double"])

    # Save to CSV
    df_double.to_csv(output_file, index=False)
    print(f"✅ Role doubles saved to {output_file}")
    
    # Print some examples
    print("\nExample player-role pairs:")
    for i in range(min(5, len(df_double))):
        print(f"  {df_double.iloc[i]['double']}")


# ==============================
# Enhanced Role Assignment with Balancing
# ==============================

def assign_balanced_roles(input_file="relevant_data.csv", output_file="role_data.csv", target_per_role=20):
    """Assign roles with better balance to ensure each position has enough players"""
    # Dynamic path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(input_file):
        input_file = os.path.join(script_dir, input_file)
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Load dataset
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} players from {input_file}")
    
    # Handle missing values
    df = df.dropna(subset=['ast', 'pts', 'reb', 'player_height'])
    print(f"After removing players with missing key stats: {len(df)} players")

    # Sort by different criteria for each position
    df_sorted_by_assists = df.sort_values('ast', ascending=False)
    df_sorted_by_points = df.sort_values('pts', ascending=False)
    df_sorted_by_rebounds = df.sort_values('reb', ascending=False)
    df_sorted_by_height = df.sort_values('player_height', ascending=False)
    
    assigned_players = set()
    roles = []
    
    # Assign Point Guards (top assist players who are shorter)
    height_median = df['player_height'].median()
    pg_candidates = df_sorted_by_assists[df_sorted_by_assists['player_height'] < height_median]
    count = 0
    for idx, row in pg_candidates.iterrows():
        if row['player_name'] not in assigned_players and count < target_per_role:
            roles.append([row["player_name"], "PG"])
            assigned_players.add(row['player_name'])
            count += 1
    
    # Assign Centers (tallest players with good rebounding)
    height_75 = df['player_height'].quantile(0.75)
    c_candidates = df_sorted_by_height[df_sorted_by_height['player_height'] >= height_75]
    count = 0
    for idx, row in c_candidates.iterrows():
        if row['player_name'] not in assigned_players and count < target_per_role:
            roles.append([row["player_name"], "C"])
            assigned_players.add(row['player_name'])
            count += 1
    
    # Assign Power Forwards (tall players with good rebounding, not as tall as centers)
    reb_median = df['reb'].median()
    pf_candidates = df[(df['player_height'] >= height_median) & 
                      (df['player_height'] < height_75) & 
                      (df['reb'] >= reb_median)].sort_values(['reb', 'player_height'], ascending=False)
    count = 0
    for idx, row in pf_candidates.iterrows():
        if row['player_name'] not in assigned_players and count < target_per_role:
            roles.append([row["player_name"], "PF"])
            assigned_players.add(row['player_name'])
            count += 1
    
    # Assign Shooting Guards (good scorers, guard size)
    pts_median = df['pts'].median()
    sg_candidates = df[(df['pts'] >= pts_median) & 
                      (df['player_height'] < height_median)].sort_values('pts', ascending=False)
    count = 0
    for idx, row in sg_candidates.iterrows():
        if row['player_name'] not in assigned_players and count < target_per_role:
            roles.append([row["player_name"], "SG"])
            assigned_players.add(row['player_name'])
            count += 1
    
    # Assign remaining players as Small Forwards
    for idx, row in df.iterrows():
        if row['player_name'] not in assigned_players:
            roles.append([row["player_name"], "SF"])
            assigned_players.add(row['player_name'])
    
    # Count final distribution
    role_counts = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    for player_name, role in roles:
        role_counts[role] += 1
    
    print(f"\nFinal role distribution:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} players")

    # Create final DataFrame
    df_roles = pd.DataFrame(roles, columns=["player_name", "role"])

    # Save to CSV
    df_roles.to_csv(output_file, index=False)
    print(f"✅ Balanced roles assigned and saved to {output_file}")
    
    return df_roles


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "datasets")
    
    # Check if input file exists
    input_file = os.path.join(dataset_dir, "relevant_data.csv")
    if os.path.exists(input_file):
        print("Using balanced role assignment...")
        # Use balanced role assignment for better results
        assign_balanced_roles(
            input_file=input_file,
            output_file=os.path.join(dataset_dir, "role_data.csv"),
            target_per_role=20
        )
        
        # Create role-based 2-tuple dataset
        create_role_double(
            input_file=os.path.join(dataset_dir, "role_data.csv"),
            output_file=os.path.join(dataset_dir, "role_double.csv")
        )
    else:
        print(f"Error: {input_file} not found!")
        print("Please run cleaner.py first to generate the relevant_data.csv file.")