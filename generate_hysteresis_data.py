import numpy as np
import pandas as pd
import time
import csv
import os
from ansys_rl_env import AnsysSoftActuatorEnv

def generate_data_final():
    # --- CONFIG ---
    dat_path = "actuator_setup_viscoelasticity_1Pa.dat"
    output_file = "hysteresis_data_final.csv"
    
    # We want ~3000 steps per "Session"
    STEPS_PER_SESSION = 3000 
    
    # --- EPISODE MANAGEMENT ---
    current_episode_idx = 0
    file_exists = os.path.exists(output_file)
    
    if file_exists:
        try:
            df_check = pd.read_csv(output_file)
            if not df_check.empty and "episode_index" in df_check.columns:
                last_idx = df_check["episode_index"].iloc[-1]
                current_episode_idx = int(last_idx) + 1
                print(f"Found existing data. Starting Episode: {current_episode_idx}")
            else:
                file_exists = False
        except:
            file_exists = False

    if not file_exists:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["pressure", "extension", "episode_index"])
        print("Created new data file.")

    # --- INITIALIZE ENV ---
    env = AnsysSoftActuatorEnv(dat_path=dat_path, target_deformation=0.01, log_level="ERROR")
    obs, _ = env.reset()
    
    # Use bounds from Env to ensure consistency
    MIN_PRESSURE = env.min_pressure
    MAX_PRESSURE = env.max_pressure
    print(f"Environment Initialized. Range: {MIN_PRESSURE/1000:.1f} kPa to {MAX_PRESSURE/1000:.1f} kPa")

    # --- SIGNAL GENERATION ---
    
    # 1. Warm-up Cycles (Full Range: Min -> Max -> Min)
    time_cycle = np.linspace(0, 4*np.pi, 200) 
    
    amplitude = (MAX_PRESSURE - MIN_PRESSURE) / 2.0
    offset = (MAX_PRESSURE + MIN_PRESSURE) / 2.0
    
    clean_cycles = offset + amplitude * np.sin(time_cycle - np.pi/2) 

    # 2. Random "Rich" Signal
    random_len = STEPS_PER_SESSION - len(clean_cycles)
    random_signal = np.zeros(random_len)
    current_p = 0
    
    np.random.seed(42 + current_episode_idx) 
    
    for i in range(random_len):

        rand_val = np.random.rand()
        # Assign the current pressure (Hold)  
        if rand_val < 0.05:   
            random_signal[i] = current_p 
        # Sine wave segment (Small oscillations)
        elif rand_val < 0.2:     
            random_signal[i] = current_p + 2000 * np.sin(i * 0.5)
            # Update current_p so the walk continues from here
            current_p = random_signal[i]
        # Random Walk
        else:                             
            change = np.random.uniform(-10000, 10000) # Reduced to 10k for smoothness
            current_p = np.clip(current_p + change, MIN_PRESSURE, MAX_PRESSURE)
            random_signal[i] = current_p
    
    pressure_cmds = np.concatenate([clean_cycles, random_signal])
    print(f"Generated signal with {len(pressure_cmds)} steps.")

    # --- LIVE DATA COLLECTION ---
    start_time = time.time()
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for i, p_target in enumerate(pressure_cmds):
            
            # --- CALCULATE ACTION ---
            # Inverse of the Env's formula: 
            # P = Min + (Max - Min) * ((A + 1) / 2)
            # A = ( (P - Min) / (Max - Min) ) * 2 - 1
            
            p_range = MAX_PRESSURE - MIN_PRESSURE
            action_val = ((p_target - MIN_PRESSURE) / p_range) * 2.0 - 1.0
            
            # Clip just to be safe (floating point errors)
            action_val = np.clip(action_val, -1.0, 1.0)
            
            # Step
            obs, reward, terminated, truncated, info = env.step([action_val])
            extension_m = obs[0]
            
            # Write (Pressure, Extension, Episode_ID)
            writer.writerow([p_target, extension_m, current_episode_idx])
            
            # Flush to Disk
            f.flush() 
            os.fsync(f.fileno()) 
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                percent = ((i + 1) / len(pressure_cmds)) * 100
                print(f"Ep {current_episode_idx} | Step {i} ({percent:.0f}%) | P: {p_target:.0f} | Ext: {extension_m:.5f} | T: {elapsed:.0f}s")

    print(f"Episode {current_episode_idx} Complete.")
    env.close()

if __name__ == "__main__":
    generate_data_final()