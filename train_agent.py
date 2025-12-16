import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from ansys_rl_env import AnsysSoftActuatorEnv

# --- CONFIGURATION ---
DAT_FILE = r"actuator_setup.dat" # Path to ANSYS solution .dat file
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
TOTAL_TIMESTEPS = 1000
TARGET_DEFORMATION = 0.02 # Target elongation in meters
CHECKPOINT_FREQ = 500 # Save model every N steps

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --- REAL-TIME MONITORING CALLBACK ---
class RealTimeMonitoringCallback(BaseCallback):
    """
    Updates TQDM and TensorBoard EVERY STEP (instead of every episode).
    Crucial for slow FEA environments.
    """
    def __init__(self, check_freq=1, verbose=1):
        super(RealTimeMonitoringCallback, self).__init__(verbose)
        self.pba# --- REAL-TIME MONITORING CALLBACK ---
class RealTimeMonitoringCallback(BaseCallback):
    """
    Updates TQDM and TensorBoard EVERY STEP (instead of every episode).
    Crucial for slow FEA environments.
    """
    def __init__(self, check_freq=1, verbose=1):
        super(RealTimeMonitoringCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.pbar = None
        self.current_episode_reward = 0.0
        self.steps_in_episode = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'], desc="Training Progress")

    def _on_step(self) -> bool:
        # 1. GET DATA FROM THE CURRENT STEP
        # 'rewards' is an array (because of vectorized envs), so we take [0]
        current_reward = self.locals['rewards'][0]
        # 'infos' contains the extra dict we returned in step()
        current_info = self.locals['infos'][0] 
        
        # Accumulate reward for the current episode display
        self.current_episode_reward += current_reward
        self.steps_in_episode += 1

        # 2. UPDATE TENSORBOARD IMMEDIATELY
        # We log the 'instant' reward so you can see if the agent is getting closer
        self.logger.record("step_metrics/instant_reward", current_reward)
        self.logger.record("step_metrics/extension_mm", current_info.get('extension', 0) * 1000)
        self.logger.record("step_metrics/pressure_kPa", current_info.get('pressure', 0) / 1000)

        # 3. UPDATE PROGRESS BAR
        self.pbar.update(1)
        self.pbar.set_description(
            f"Rew: {current_reward:.2f} | Ext: {current_info.get('extension', 0)*1000:.1f}mm"
        )
        
        # 4. RESET TRACKERS IF EPISODE ENDS
        if self.locals['dones'][0]:
            self.logger.record("rollout/completed_episode_reward", self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.steps_in_episode = 0

        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


def main():
    print("="*50)
    print("Initializing ANSYS Environment...")
    print("="*50)
    # Initialize the environment
    env = AnsysSoftActuatorEnv(dat_path=DAT_FILE, target_deformation=TARGET_DEFORMATION)
    
    env = DummyVecEnv([lambda: Monitor(env, LOG_DIR)]) # in case we need multiple env instances
    #env = Monitor(env, LOG_DIR) # wrap it in Monitor for TensorBoard logging

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

    # TQDM Callback for Training Progress Bar
    progress_callback = RealTimeMonitoringCallback()

    print("Starting Training Loop...")
    print("   Monitor progress using TensorBoard.")
    
    # Train for TOTAL_TIMESTEPS ------------------------------------------------
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, progress_callback]
        )
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
        env.envs[0].close()

if __name__ == "__main__":
    main()