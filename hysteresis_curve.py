import numpy as np
import matplotlib.pyplot as plt
from ansys_rl_env import AnsysSoftActuatorEnv

def sweep_pressure():
    # Initialize environment
    TARGET_DEFORMATION = 0.05 # 5 cm
    env = AnsysSoftActuatorEnv(
        dat_path=r"actuator_setup_viscoelasticity_1Pa_slowTerm.dat", 
        target_deformation=TARGET_DEFORMATION,
        dt=0.1,
        log_level="ERROR"
    )
    
    # --- 1. Define Hysteresis Cycle ---
    MIN_PRESSURE_KPA = 0.0
    MAX_PRESSURE_KPA = 120.0
    INCREMENT_KPA = 15.0
    
    # Create Loading phase (0 -> 120)
    steps_up = int((MAX_PRESSURE_KPA - MIN_PRESSURE_KPA) / INCREMENT_KPA) + 1
    loading_phase = np.linspace(MIN_PRESSURE_KPA, MAX_PRESSURE_KPA, steps_up)
    
    # Create Unloading phase (120 -> 0)
    # We remove the first element of unloading so we don't repeat the peak (120) twice
    unloading_phase = np.flip(loading_phase)[1:]
    
    # Combine into one full cycle
    pressures_kpa = np.concatenate([loading_phase, unloading_phase])
    extensions_cm = []

    print(f"Starting Hysteresis Pressure Sweep (0 -> {MAX_PRESSURE_KPA} -> 0 kPa)...")

    # Reset environment
    obs, _ = env.reset()
    
    # --- 2. Run the Sweep ---
    for i, p_kpa in enumerate(pressures_kpa):
        p_pa = p_kpa * 1000.0 # Convert kPa to Pa
        
        # Calculate action normalized to [-1, 1] based on 120kPa being the max here
        # Note: We ensure the denominator is at least the current pressure to avoid >1.0 actions
        # if the env.max_pressure happens to be lower than our sweep target.
        normalization_max = max(env.max_pressure, MAX_PRESSURE_KPA * 1000.0)
        action = (p_pa / normalization_max) * 2 - 1 
        
        # Step the environment
        obs, reward, done, truncated, info = env.step([action])
        cur_deform = obs[0] # in meters
        
        phase = "Loading" if i < len(loading_phase) else "Unloading"
        print(f"[{phase}] Pressure: {p_kpa:.1f} kPa | Deform: {cur_deform*100:.4f} cm")

        extensions_cm.append(cur_deform * 100)

    # --- 3. Plot Hysteresis Loop ---
    plt.figure(figsize=(10, 6))
    
    # Split data back for plotting distinct lines
    split_index = len(loading_phase)
    
    # Plot Loading (Green solid)
    plt.plot(pressures_kpa[:split_index], extensions_cm[:split_index], 
             marker='o', color='green', label='Loading (0 \u2192 120)')
    
    # Plot Unloading (Red dashed)
    # Include the peak point (split_index-1) so the lines connect
    plt.plot(pressures_kpa[split_index-1:], extensions_cm[split_index-1:], 
             marker='x', linestyle='--', color='red', label='Unloading (120 \u2192 0)')

    # Target line
    plt.axhline(y=TARGET_DEFORMATION * 100, color='gray', linestyle=':', label='Target')

    plt.grid(True)
    plt.xlabel('Pressure (kPa)')
    plt.ylabel('Extension (cm)')
    plt.title('Actuator Hysteresis Loop')
    plt.legend()
    
    # Save and show
    plt.savefig("actuator_hysteresis_curve.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    sweep_pressure()