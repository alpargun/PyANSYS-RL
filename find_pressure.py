import numpy as np
import matplotlib.pyplot as plt
from ansys_rl_env import AnsysSoftActuatorEnv


def sweep_pressure():
    # Initialize environment
    env = AnsysSoftActuatorEnv(
        dat_path=r"actuator_setup_150kPa.dat", 
        target_deformation=0.02, # 2cm target
        #actuation_axis='X'
    )
    
    # Define range: 0 kPa to 150 kPa in 10 kPa steps
    MIN_PRESSURE_KPA = 0
    MAX_PRESSURE_KPA = 10
    INCREMENT_KPA = 10
    num_steps = int((MAX_PRESSURE_KPA - MIN_PRESSURE_KPA) / INCREMENT_KPA) + 1
    pressures_kpa = np.linspace(MIN_PRESSURE_KPA, MAX_PRESSURE_KPA, num_steps)
    extensions_mm = []

    print("Starting Pressure Sweep...")
    
    for p_kpa in pressures_kpa:
        # Convert kPa to normalized action [-1, 1]
        # Formula: Action = (Pressure / Max_Pressure * 2) - 1
        # Assuming Max_Pressure in env is 150,000 Pa (adjust env if needed)
        
        # Calculate raw pressure in Pa
        p_pa = p_kpa * 1000.0
        
        # Manually invoke the solver logic (bypassing step() to ensure steady state)
        env.mapdl.prep7()
        env.mapdl.cmsel('S', 'Inner1new') # Explicitly select the internal cavity faces
        env.mapdl.sf('ALL', 'PRES', p_pa) # Apply pressure in Pascals
        env.mapdl.allsel() # Reselect everything so the solver can see the rest of the body
        
        env.mapdl.run('/SOLU')
        env.mapdl.antype('STATIC')
        env.mapdl.kbc(0) # Force Ramped Loading
        env.mapdl.nlgeom('ON')
        # Min 50 steps, Max 1000 steps (give it freedom to slow down), Min 20 steps
        env.mapdl.nsubst(50, 1000, 20)        
        env.mapdl.solve()
        env.mapdl.finish()
        
        # GET RESULT -----------------------------------------------------------
        # Check if solver converged
        if not env.mapdl.solution.converged:
            print(f"Pressure: {p_kpa:.0f} kPa -> FAILED (Diverged)")
            extensions_mm.append(0.0) # Record 0 for failure
            continue

        # Get result
        env.mapdl.post1()
        env.mapdl.set('LAST')
        env.mapdl.nsort('U', 'X')
        ext = env.mapdl.get_value('SORT', 0, 'MAX')
        
        # Safety check
        if ext is None: ext = 0.0
        
        # Print result
        print(f"Pressure: {p_kpa:.0f} kPa -> Extension: {ext*1000:.2f} mm")
        extensions_mm.append(ext * 1000)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(pressures_kpa, extensions_mm, marker='o', linestyle='-')
    plt.axhline(y=20, color='r', linestyle='--', label='Target (20mm)')
    plt.grid(True)
    plt.xlabel('Pressure (kPa)')
    plt.ylabel('Extension (mm)')
    plt.title('Actuator Characterization Curve')
    plt.legend()
    plt.show()

    env.close()

if __name__ == "__main__":
    sweep_pressure()