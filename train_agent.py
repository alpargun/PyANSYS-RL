import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from ansys_rl_env import AnsysSoftActuatorEnv

# --- CONFIGURATION ---
DAT_FILE = r"actuator_setup.dat" # Path to ANSYS solution .dat file
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
TOTAL_TIMESTEPS = 10000
TARGET_DEFORMATION = 0.02 # Target elongation in meters
CHECKPOINT_FREQ = 500 # Save model every N steps

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("="*50)
    print("Initializing ANSYS Environment...")
    print("="*50)
    # Initialize the environment
    env = AnsysSoftActuatorEnv(dat_path=DAT_FILE, target_deformation=TARGET_DEFORMATION)
    env = Monitor(env, LOG_DIR) # wrap it in Monitor for TensorBoard logging

    print("Setting up PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, # verbose=1 prints progress to console
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=64
    )

    # Checkpoint Callback (Saves every 500 steps)
    # Since FEA is slow and prone to crashing, save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, 
        save_path=MODEL_DIR, 
        name_prefix="ansys_ppo"
    )

    print("Starting Training Loop...")
    print("   Monitor progress using TensorBoard.")
    
    # Train for TOTAL_TIMESTEPS ------------------------------------------------
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
        print("Training Finished!")
        
        # Save the final model
        model.save(f"{MODEL_DIR}/final_ansys_model")
        print("Model saved successfully.")
    except KeyboardInterrupt:
        print("Training manually stopped. Saving current model...")
        model.save(f"{MODEL_DIR}/interrupted_model")    
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Attempt to save whatever we have
        model.save(f"{MODEL_DIR}/interrupted_model")
    finally:
        # Close ANSYS safely
        env.close()

if __name__ == "__main__":
    main()