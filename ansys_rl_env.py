import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from ansys.mapdl.core import launch_mapdl


class AnsysSoftActuatorEnv(gym.Env):
    """
    Custom Environment built on gymnasium interface.
    Connects to ANSYS MAPDL to control a soft pneumatic actuator.
    """
    
    def __init__(self, dat_path, target_deformation=0.015):
        super(AnsysSoftActuatorEnv, self).__init__()
        
        # CONFIGURATION
        self.dat_path = dat_path # .dat file for the solution from ANSYS Mechanical
        self.target_deformation = target_deformation # Target elongation (meters)
        self.max_pressure = 100000.0 # Max pressure in Pascals (e.g. 100kPa)
        
        # Path to ANSYS Student Executable to make sure we use the license
        student_exe = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\ansys\bin\winx64\ansys252.exe"
        
        # Create a run folder for ANSYS logs
        run_dir = r"C:\Ansys_RL_Runs"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # LAUNCH ANSYS MAPDL ---------------------------------------------------
        print(f"Launching ANSYS from: {student_exe}")
        
        self.mapdl = launch_mapdl(
            loglevel="ERROR", # "INFO" for more details
            exec_file=student_exe,
            run_location=run_dir, # Run in a real folder, not Temp
            nproc=4, # Number of cores to use (max: 4 for Student license)
            additional_switches="-smp", # Student license needs SMP (Shared Memory)
            override=True, # Overwrite old lock files
            mode="grpc", # Standard connection mode
            start_timeout=60, # Wait 60s for the license popup/check
        )

        # LOAD ANSYS MODEL -----------------------------------------------------
        print(f"Loading Model from {self.dat_path}...")
        
        # Upload the file to the solver's working directory
        self.mapdl.upload(self.dat_path)

        self.mapdl.clear()
        
        # Enter Pre-processor first
        self.mapdl.prep7()
        
        # Now disable the shape warnings
        #self.mapdl.run('SHPP, OFF') # Disable shape checking errors
        #self.mapdl.run('/NERR, 0, -1') # Suppress error dialogs
        
        # Read the file
        self.mapdl.input(self.dat_path, verbose=True)

        self.mapdl.finish()
        self.mapdl.prep7()
        
        # Check Node Count
        node_count = self.mapdl.mesh.n_node
        print(f"Model Loaded. Node Count: {node_count}")
        if node_count > 128000:
            print("WARNING: Node count exceeds Student License limit (128k). Solver might crash.")
        
        # ACTION SPACE
        # Action: Continuous pressure value between 0 and Max Pressure
        self.action_space = spaces.Box(
            low=np.array([0.0]), 
            high=np.array([self.max_pressure]), 
            shape=(1,), # we control 1 variable (Pressure)
            dtype=np.float32
        )

        # OBSERVATION SPACE
        # Observation: Current Deformation in X
        self.observation_space = spaces.Box(
            low=np.array([-0.5]), # assuming max -0.5 m deformation
            high=np.array([0.5]), # assuming max +0.5 m deformation
            shape=(1,), 
            dtype=np.float32
        )
        

    def step(self, action):
        # Unpack Action and clip it to ensure it stays within bounds
        pressure_val = np.clip(action[0], 0, self.max_pressure)
        
        # Apply Loads in ANSYS
        self.mapdl.prep7()

        # Remove previous loads to be safe
        self.mapdl.sf('ALL', 'PRES', 0)

        # Select the faces for pressure (Named Selection 'Inner1new')
        self.mapdl.cmsel('S', 'Inner1new') 
        # Apply the new pressure (SF command in APDL)
        self.mapdl.sf('ALL', 'PRES', pressure_val)

        # Select everything again for solving
        self.mapdl.allsel()
        
        # Solve
        self.mapdl.run('/SOLU')
        self.mapdl.antype('STATIC') # Ensure Static Analysis #TODO do we need?
        self.mapdl.solve()
        self.mapdl.finish()
        
        # Get Observation (Deformation)
        self.mapdl.post1()
        self.mapdl.set('LAST') # Read last result set
        
        # Calculate max deformation across all nodes
        deformation_axis = 'X'
        
        # Sort nodes by deformation to find the max
        self.mapdl.nsort('U', deformation_axis) 
        
        # Get the maximum value from the sorted list
        max_deformation = self.mapdl.get_value('SORT', 0, 'MAX')

        # Handle solver failures
        if max_deformation is None or np.isnan(max_deformation):
            max_deformation = 0.0
            reward = -100.0 
            done = True
        else:
            # Calculate Reward
            # Compare the max deformation from the simulation to target deformation
            error = abs(self.target_deformation - max_deformation)
            reward = -(error * 100) 
            done = bool(error < 0.0001) # Done if within 0.1mm tolerance

        # Return info
        info = {"pressure_applied": pressure_val, "max_deformation": max_deformation}

        # Gymnasium requires observation, reward, terminated, truncated, info
        return np.array([max_deformation], dtype=np.float32), reward, done, False, info
    

    def reset(self, seed=None):
        """
        Reset the environment to an initial state. (Load 0)
        """
        super().reset(seed=seed)
        
        # In FEA, reset usually means unloading the structure
        self.mapdl.prep7()
        self.mapdl.sf('ALL', 'PRES', 0) # Remove pressure
        
        # Run a quick solve to zero out deformations
        self.mapdl.run('/SOLU')
        self.mapdl.solve()
        self.mapdl.finish()
        
        # Get the zeroed observation
        #self.mapdl.post1()
        #zero_def = self.mapdl.get_value('NODE', self.tip_node_id, 'U', 'X')
        #if zero_def is None: zero_def = 0.0
        
        return np.array([0.0], dtype=np.float32), {}


    def close(self):
        """
        Clean up ANSYS resources.
        """
        if self.mapdl:
            self.mapdl.exit()


# ==========================================
#   EXAMPLE USAGE (TESTING THE ENV)
# ==========================================
if __name__ == "__main__":
    # PATH to exported .dat file for the solution
    DAT_FILE = "actuator_setup.dat"
    
    # Create the environment
    # try/finally block to ensure ANSYS closes if code crashes
    env = None
    try:
        env = AnsysSoftActuatorEnv(dat_path=DAT_FILE, target_deformation=0.03)
        
        # Reset environment
        obs, _ = env.reset()
        print(f"Initial Observation: {obs}")
        
        # Run for n_steps with random actions
        N_STEPS = 1
        for i in range(N_STEPS):
            # Sample a random pressure from action space
            random_action = env.action_space.sample()
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(random_action)
            
            print(f"Step {i+1}:")
            print(f"  Applied Pressure: {info['pressure_applied']:.2f} Pa")
            print(f"  Resulting Deform: {obs[0]*100:.6f} cm")
            print(f"  Reward: {reward:.4f}")
            
            if done:
                print("Target Reached!")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if env:
            print("Closing ANSYS...")
            env.close()