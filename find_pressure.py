import numpy as np
import matplotlib.pyplot as plt
from ansys_rl_env import AnsysSoftActuatorEnv


def sweep_pressure():
    # Initialize environment
    TARGET_DEFORMATION = 0.05 # 5 cm
    env = AnsysSoftActuatorEnv(
        dat_path=r"actuator_setup_viscoelasticity_1Pa.dat", 
        target_deformation=TARGET_DEFORMATION,
        log_level="ERROR"
    )
    
    MIN_PRESSURE_KPA = env.min_pressure / 1000.0
    MAX_PRESSURE_KPA = env.max_pressure / 1000.0
    INCREMENT_KPA = 10
    num_steps = int((MAX_PRESSURE_KPA - MIN_PRESSURE_KPA) / INCREMENT_KPA) + 1
    pressures_kpa = np.linspace(MIN_PRESSURE_KPA, MAX_PRESSURE_KPA, num_steps)
    extensions_mm = []

    print("Starting Pressure Sweep...")

    # Reset environment
    obs, _ = env.reset()
    print(f"Initial Observation: {obs}")
    
    # Run for n_steps with random actions
    for p_kpa in pressures_kpa:
        p_pa = p_kpa * 1000.0 # Convert kPa to Pa
        action = (p_pa / env.max_pressure) * 2 - 1  # Normalize to [-1, 1]
        # Step the environment
        obs, reward, done, truncated, info = env.step([action])
        cur_deform = obs[0] # in meters
        print(f"  Applied Pressure: {info['pressure_applied']:.2f} Pa")
        print(f"  Resulting Deform: {cur_deform*100:.6f} cm")
        print(f"  Reward: {reward:.4f}")

        extensions_mm.append(cur_deform * 100)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(pressures_kpa, extensions_mm, marker='o', linestyle='-')
    plt.axhline(
        y=TARGET_DEFORMATION * 100, 
        color='r', 
        linestyle='--', 
        label=f'Target ({TARGET_DEFORMATION * 100} cm)'
    )
    plt.grid(True)
    plt.xlabel('Pressure (kPa)')
    plt.ylabel('Extension (cm)')
    plt.title('Actuator Characterization Curve')
    plt.legend()
    plt.savefig("actuator_characterization_curve.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    sweep_pressure()