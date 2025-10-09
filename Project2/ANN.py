import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Fixed import
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
import ast
import numpy as np

# ==============================
# Convert team strings back into numeric vectors
# ==============================

def team_to_vector_from_csv(team_string, df_players, role_encoder):
    """
    Convert a serialized team string into a numeric vector:
    - Roles are one-hot encoded
    - 1 binary duplicate flag (duplicate players or duplicate roles)
    """
    # Clean quotes if needed
    if isinstance(team_string, str) and team_string.startswith('"') and team_string.endswith('"'):
        team_string = team_string[1:-1]

    try:
        team_obj = ast.literal_eval(team_string)
    except Exception as e:
        # Try cleaning and parsing again
        cleaned = team_string.replace('""', '"')
        try:
            team_obj = ast.literal_eval(cleaned)
        except Exception as e2:
            print(f"Error parsing team string: {team_string}")
            print(f"Original error: {e}")
            print(f"Cleaned error: {e2}")
            return np.zeros(21)  # Return zero vector if parsing fails

    player_names = []
    role_list = []

    for player in team_obj:
        if not isinstance(player, (list, tuple)) or len(player) < 2:
            continue

        player_name = str(player[0]).strip("()'\" ")
        role = str(player[1]).strip("()'\" ")

        player_names.append(player_name)
        role_list.append(role)

    # Ensure we have exactly 5 players
    if len(role_list) != 5:
        print(f"Warning: Team has {len(role_list)} players instead of 5")
        # Pad with empty roles or truncate
        if len(role_list) < 5:
            role_list.extend(['UNKNOWN'] * (5 - len(role_list)))
        else:
            role_list = role_list[:5]

    try:
        # One-hot encode the 5 roles
        role_array = role_encoder.transform(np.array(role_list).reshape(-1, 1))
        # Flatten the one-hot encoded array
        role_array = role_array.flatten()
    except Exception as e:
        print(f"Error encoding roles {role_list}: {e}")
        # Return appropriate sized zero vector
        n_roles = len(role_encoder.categories_[0]) if hasattr(role_encoder, 'categories_') else 5
        role_array = np.zeros(n_roles * 5)

    # Duplicate flags
    has_duplicate_player = int(len(player_names) != len(set(player_names)))
    has_duplicate_role = int(len(role_list) != len(set(role_list)))

    # Final feature vector
    team_vector = np.concatenate([role_array, [has_duplicate_player, has_duplicate_role]])

    return team_vector

# ==============================
# Main ANN Code
# ==============================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.dirname(SCRIPT_DIR) else SCRIPT_DIR
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

    # Check if datasets exist
    train_path = os.path.join(DATASETS_DIR, "train_dataset.csv")
    test_path = os.path.join(DATASETS_DIR, "test_dataset.csv")

    if not os.path.exists(train_path):
        print(f"Error: Training dataset not found at {train_path}")
        print("Please run data_generator.py first to create the datasets")
        return

    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        print("Please run data_generator.py first to create the datasets")
        return

    # 1. Load train and test CSVs
    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Training set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    # Check label distribution
    print("\nTraining label distribution:")
    print(df_train['label'].value_counts())
    print("\nTest label distribution:")
    print(df_test['label'].value_counts())

    # Extract all roles for encoder fitting
    print("Extracting roles for encoding...")
    all_roles = []
    for df in [df_train, df_test]:
        for team_string in df["team"]:
            try:
                if isinstance(team_string, str) and team_string.startswith('"') and team_string.endswith('"'):
                    team_string = team_string[1:-1]
                team_obj = ast.literal_eval(team_string)
                for player in team_obj:
                    if isinstance(player, (list, tuple)) and len(player) >= 2:
                        role = str(player[1]).strip("()'\" ")
                        all_roles.append(role)
            except Exception as e:
                print(f"Error parsing team for role extraction: {e}")
                continue

    print(f"Found {len(all_roles)} role instances")
    unique_roles = list(set(all_roles))
    print(f"Unique roles: {unique_roles}")

    # Fit one-hot encoder on all roles
    role_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    role_encoder.fit(np.array(all_roles).reshape(-1, 1))

    print(f"Role encoder categories: {role_encoder.categories_[0]}")

    # Convert datasets into numeric vectors
    print("Converting teams to feature vectors...")
    X_train = []
    for i, team_string in enumerate(df_train["team"]):
        if i % 100 == 0:
            print(f"Processing training sample {i}/{len(df_train)}")
        vector = team_to_vector_from_csv(team_string, None, role_encoder)
        X_train.append(vector)
    
    X_train = np.array(X_train)
    y_train = df_train["label"].values

    X_test = []
    for i, team_string in enumerate(df_test["team"]):
        if i % 100 == 0:
            print(f"Processing test sample {i}/{len(df_test)}")
        vector = team_to_vector_from_csv(team_string, None, role_encoder)
        X_test.append(vector)
    
    X_test = np.array(X_test)
    y_test = df_test["label"].values

    print(f"Feature vector shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Check for any issues with the data
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print("Warning: NaN values found in feature vectors")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Build model
    input_dim = X_train_scaled.shape[1]
    print(f"Building model with input dimension: {input_dim}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=input_dim, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Model architecture:")
    model.summary()

    # 3. Train with early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # 4. Evaluate
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # 5. Predict
    predictions = model.predict(X_test_scaled[:10], verbose=0)
    print("\nSample predictions (first 10 test samples):")
    for i, (pred, actual) in enumerate(zip(predictions, y_test[:10])):
        print(f"Sample {i+1}: Predicted={pred[0]:.4f}, Actual={actual}, Predicted_class={int(pred[0] > 0.5)}")

    # 6. Save the trained model
    print("Saving trained model...")
    model.save("team_optimizer_model.h5")
    print("✅ Model saved as team_optimizer_model.h5")

    # 7. Plot training history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title("Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✅ Training plots saved as training_history.png")
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    main()