from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import time

from ansys.mapdl.core import launch_mapdl

class AnsysSoftActuatorEnv(gym.Env):
    """
    Custom Environment built on gymnasium interface.
    Connects to ANSYS MAPDL to control a soft pneumatic actuator.
    """
    
    def __init__(self, dat_path, target_deformation=0.05, min_presure=0.0, max_pressure=120000.0, log_level="INFO"):
        super(AnsysSoftActuatorEnv, self).__init__()
        
        # CONFIGURATION
        self.dat_path = dat_path # .dat file for the solution from ANSYS Mechanical
        self.target_deformation = target_deformation # Target elongation (meters)

        # Pressure Range in Pascals (Vacuum to Expansion)
        self.min_pressure = min_presure 
        self.max_pressure = max_pressure

        self.actuation_axis = "X" # Axis along which deformation is measured
        self.current_step_count = 0   # Track steps for "done" logic
        self.pressure_step_limit = 15000.0 # Max change allowed per step (Pa)
        self.current_pressure = 0.0 # Track current state
        
        # TIME MANAGEMENT (The missing link for Hysteresis)
        self.sim_time = 0.0
        self.dt = 1  # How many sim steps corr. to 1 second of physics (1/N)

        # Path to ANSYS Student Executable to make sure we use the license
        student_exe = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\ansys\bin\winx64\ansys252.exe"
        
        # Create a timestamped run folder for ANSYS logs
        run_dir_base = r"C:\Ansys_RL_Runs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(run_dir_base, f"run_{timestamp}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # LAUNCH ANSYS MAPDL ---------------------------------------------------
        print(f"Launching ANSYS from: {student_exe}")
        self.mapdl = launch_mapdl(
            loglevel=log_level, # "INFO" for more details
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
        self.mapdl.upload(self.dat_path) # Upload the file to the solver's working dir

        self.mapdl.clear()
        self.mapdl.prep7(mute=True) # Enter Pre-processor
        self.mapdl.run('SHPP, OFF', mute=True) # Disable shape checking errors
        self.mapdl.run('/NERR, 0, -1', mute=True) # Suppress error dialogs
        self.mapdl.input(self.dat_path) # Read the file

        # INITIAL SETUP
        self.mapdl.prep7(mute=True) # Re-enter preprocessor again as dat file ends with FINISH
        self.mapdl.allsel(mute=True) # Ensure you can see/edit the whole model
        # FIX for INCREASE ERROR LIMIT (Num ERROR and WARNING messages exceeds 10000)
        # Allow up to 10 million warnings (effectively infinite)
        self.mapdl.run('/NERR, 10000000, 10000000, -1') # '-1' means "Do not abort"
        self.mapdl.run('/NOLIST', mute=True) # Disable "Element/Node" integrity check warning

        # ======================================================================
        # SIZE and MATERIAL CHECK ----------------------------------------------
        #self.check_material_properties(1)
        # CHECK SELF CONTACT
        #self.check_contact_status()
        # ======================================================================
        
        # Ensure Fixed Support
        self.mapdl.cmsel('S', 'FixedSupport')
        self.mapdl.d('ALL', 'ALL', 0)

        # Pre-calculate surface elements to apply pressure and save as a component
        self.mapdl.cmsel('S', 'Inner1new')
        self.mapdl.esln('S')
        self.mapdl.esel('R', 'ENAME', '', 154) 
        self.mapdl.cm('PRESSURE_ELEMS', 'ELEM') # Save as component
        self.mapdl.allsel(mute=True)

        # SAVE THE CLEAN STATE
        self.mapdl.save('base_state') # Saves base_state.db
        self.mapdl.finish()

        # ACTION SPACE
        self.action_space = spaces.Box(
            low=np.array([-1.0]), #low=np.array([0.0]), 
            high=np.array([1.0]), #high=np.array([self.max_pressure]),
            shape=(1,), # we control 1 variable (Pressure)
            dtype=np.float32
        )
        # OBSERVATION SPACE: Current Deformation in X
        self.observation_space = spaces.Box(
            low=np.array([-0.3]), # assuming max -0.3 m deformation
            high=np.array([0.3]), # assuming max +0.3 m deformation
            shape=(1,), 
            dtype=np.float32
        )
        
    def step(self, action):
        self.current_step_count += 1
        self.sim_time += self.dt # Advance time (Critical for Hysteresis)

        # Unpack Action and clip it to ensure it stays within bounds
        raw_action = np.clip(action[0], -1.0, 1.0)
        # P = Min + (Max - Min) * ( (Action + 1) / 2 )
        pressure_range = self.max_pressure - self.min_pressure
        norm_action = (raw_action + 1.0) / 2.0
        target_pressure = self.min_pressure + (pressure_range * norm_action)
        
        # 2. APPLY LIMITER (The Speed Fix)
        # Calculates the allowed change and updates self.current_pressure
        delta = target_pressure - self.current_pressure
        real_delta = np.clip(delta, -self.pressure_step_limit, self.pressure_step_limit)
        
        # We only ask ANSYS to solve for this smaller, safe step
        pressure_val = self.current_pressure + real_delta
        
        # --- ENTER SOLUTION PROCESSOR ---
        # NO PREP7 CALL HERE!
        self.mapdl.run('/SOLU')

        # 2. DYNAMIC SUBSTEPS (The Speed Fix)
        # We calculate how many steps we *actually* need for this specific delta.
        # Rule of thumb: 1 step per 2000 Pa is very safe.
        # 15,000 Pa -> ~7 steps. (Compare to 100 steps before!)
        optimal_substeps = max(2, int(abs(real_delta) / 2000))
        
        self.mapdl.nsubst(optimal_substeps, 1000, 2)

        # APPLY PRESSURE LOAD --------------------------------------------------
        #self.mapdl.cmsel('S', 'Inner1new') # Select the faces for pressure
        #self.mapdl.esln('S') # Select ALL elements attached to these nodes (Solids + Surfs)
        #self.mapdl.esel('R', 'ENAME', '', 154) # Filter out Surface Effect elements (154) to not double load
        # Apply Pressure to the remaining (Solid) elements
        #self.mapdl.sfe('ALL', 1, 'PRES', '', pressure_val) # Since the Surfs are hidden, the pressure only applies once.
        #self.mapdl.allsel() # Select everything again for solving

        # --- OPTIMIZED LOAD APPLICATION ---
        # Replaced 3 selection commands with 1 component selection
        self.mapdl.cmsel('S', 'PRESSURE_ELEMS') # PRESSURE_ELEMS
        self.mapdl.sfe('ALL', 1, 'PRES', '', pressure_val)
        self.mapdl.allsel() # Select everything again for solving
        
        self.mapdl.time(self.sim_time)
        self.mapdl.solve()
        
        # CHECK FOR SOLVER CONVERGENCE -----------------------------------------
        if not self.mapdl.solution.converged: # If diverged, garbage results
            print(f"DEBUG: Solver Diverged at {pressure_val:.0f} Pa")
            # Return current state as 0, Penalty -50, End Episode
            return np.array([0.0], dtype=np.float32), -50.0, True, False, {"pressure": pressure_val}

        self.current_pressure = pressure_val
        
        # GET OBSERVATION ------------------------------------------------------
        # # Get Observation (Deformation)
        # self.mapdl.post1()
        # self.mapdl.set('LAST') # Read the final converged substep
        # self.mapdl.allsel()
        
        # # Calculate max deformation across all nodes
        # self.mapdl.nsort('U', self.actuation_axis) # Sort nodes by deformation
        # # Measure Extension Relative to Base (Tip - Base)
        # # Get Tip (Min Displacement - assuming extension in -X)
        # tip_disp = self.mapdl.get_value('SORT', 0, 'MIN')
        # # Get Base (Max Displacement 0)
        # self.mapdl.cmsel('S', 'FixedSupport')
        # self.mapdl.nsort('U', self.actuation_axis)
        # base_disp = self.mapdl.get_value('SORT', 0, 'MAX')

        # --- OBSERVATION (Optimized *GET) ---
        self.mapdl.post1()
        self.mapdl.set('LAST') 
        
        # 1. Select ALL nodes (Original Accuracy)
        self.mapdl.allsel()
        
        # 2. Sort ALL nodes by X deformation
        self.mapdl.nsort('U', self.actuation_axis) 
        
        # 3. Get Tip Value (Min X) using *GET (Fast)
        # This replaces get_value('SORT', 0, 'MIN')
        self.mapdl.run('*GET, TIP_VAL, SORT, 0, MIN') 
        
        # 4. Get Base Value (Max X)
        self.mapdl.cmsel('S', 'FixedSupport')
        self.mapdl.nsort('U', self.actuation_axis)
        self.mapdl.run('*GET, BASE_VAL, SORT, 0, MAX') 
        
        # 5. Read variables directly from memory
        tip_disp = self.mapdl.parameters['TIP_VAL']
        base_disp = self.mapdl.parameters['BASE_VAL']

        cur_deformation = abs(tip_disp - base_disp)

        # CALCULATE REWARD -----------------------------------------------------
        # Compare the max deformation from the simulation to target deformation
        error = abs(self.target_deformation - cur_deformation)
        reward = -(error * 100)

        # Done logic
        #done = bool(error < 0.0001) # Done if within 0.1mm tolerance
        terminated = bool(error < 0.0005) # 0.5mm tolerance
        truncated = bool(self.current_step_count >= 100) # Force stop after 100 steps

        # Return info
        info = {"pressure_applied": pressure_val, "max_deformation": cur_deformation}
        print(f"Step {self.current_step_count} | reward: {reward}, deformation(cm): {cur_deformation*100}, pressure: {pressure_val}")
        # Gymnasium requires observation, reward, terminated, truncated, info
        return np.array([cur_deformation], dtype=np.float32), reward, terminated, truncated, info

    def reset(self, seed=None):
        """
        Reset the environment to an initial state. (Load 0)
        """
        super().reset(seed=seed)
        self.current_step_count = 0
        self.sim_time = 0.0 

        self.mapdl.finish()
        # Instead of solving physics, just reload the clean memory
        self.mapdl.resume('base_state') # Resume Clean State (No loads, Time=0)

        self.mapdl.run('/SOLU')
        
        self.mapdl.antype('STATIC') 
        self.mapdl.timint('ON')  # Hysteresis ON
        self.mapdl.lnsrch('ON')  # Line Search ON
        self.mapdl.pred('ON')
        self.mapdl.nlgeom('ON') 
        
        self.mapdl.run('SOLCONTROL, ON') # Enable Optimized Nonlinear Defaults
        self.mapdl.kbc(0)

        self.mapdl.neqit(50)

        # LOOSEN TOLERANCE
        # 5% tolerance prevents the solver from getting stuck in bisection loops
        #self.mapdl.cnvtol('F', '', 0.01, 2) 

        # 3. HIGH SUBSTEPS (Mechanical Standard)
        # Start with 50-100 steps. 
        # This ensures the first load increment is tiny (1-2 kPa).
        self.mapdl.autots('ON')
        self.mapdl.nsubst(20, 1000, 5)
        #self.mapdl.nsubst(100, 5000, 20) # works for 120kPa 
        
        self.mapdl.eqslv('SPARSE') # (Fastest for <10k nodes)
        
        # I/O Settings
        self.mapdl.outres('ERASE') 
        self.mapdl.outres('NSOL', 'LAST')
        
        self.mapdl.finish()
        
        return np.array([0.0], dtype=np.float32), {}

    def render(self):
        print("Rendering snapshot...")
        
        # Enter Post-Processor
        self.mapdl.post1()
        self.mapdl.set('LAST')
        self.mapdl.allsel()
        
        self.mapdl.show('PNG') # render to PNG format
        
        # Create the Plot
        # PLNSOL, U, X  -> Plot Nodal Solution, Deformation, X-Axis
        # 1, 1          -> Show undeformed shape (wireframe) + Deformed shape
        self.mapdl.plnsol('U', 'X', 1, 1) 
        
        self.mapdl.show('CLOSE')

    def check_material_properties(self, mat_id):
        """Helper to print what ANSYS sees for the material"""
        print("\nCHECKING MATERIAL PROPERTIES...")
        # Check Node Count
        self.mapdl.allsel()
        node_count = self.mapdl.mesh.n_node
        print(f"Model Loaded. Node Count: {node_count}")
        if node_count > 128000:
            print("WARNING: Node count exceeds Student License limit (128k). Solver might crash.")

        # Check model length
        self.mapdl.allsel()
        xmin = self.mapdl.get_value("NODE", 0, "MNLOC", "X")
        xmax = self.mapdl.get_value("NODE", 0, "MXLOC", "X")
        model_len = xmax - xmin
        print(f"\n--- CONFIGURING UNITS & MATERIAL ---")
        print(f"Detected Model Length: {model_len:.4f}")
        
        # Verify materials are correct (Hyperelastic for soft robots)
        print(f"--- VERIFYING MATERIAL {mat_id} ---")
        # Check Hyperelastic Table
        output = self.mapdl.run(f"TBLIST, ALL, {mat_id}")
        if "HYPER" in output or "NEO" in output:
            print(f"SUCCESS: Material {mat_id} is Hyperelastic (Rubber).")
            print(output) # Uncomment to see full table
        else:
            print(f"WARNING: Material {mat_id} looks like STEEL or Undefined.")

        # Check for Viscoelasticity (Prony Series)
        if "Prony" in output:
            print("SUCCESS: Prony Series table found (Viscoelasticity active).")
        else:
            print("CRITICAL WARNING: No 'PRONY' table found! Viscoelasticity is missing.")
            print("Without this, the material has no time-dependent behavior (hysteresis).")

        # Check density
        # *GET command: Get Density (DENS) for Material ID (mat_id)
        density = self.mapdl.get_value("MAT", mat_id, "DENS")
        print(f"Density: {density}")

    def check_contact_status(self):
        """Prints status of all contact pairs in the model."""
        print("\n--- CHECKING CONTACT DEFINITIONS ---")
        
        # 1. List all Element Types to see if Contact exists
        # Look for IDs like 170 (Target) and 173/174 (Contact)
        print("Element Types Defined:")
        self.mapdl.etlist()
        
        # 2. Check Contact Status
        # CNCHECK prints a report: Are pairs "Open" (not touching) or "Closed" (touching)?
        print("\nContact Pair Status:")
        output = self.mapdl.run("CNCHECK, DETAIL")
        print(output)
        
        # 3. Count Contact Elements
        # Select all Contact elements
        self.mapdl.esel("S", "ENAME", "", 173, 174)
        count = self.mapdl.get_value("ELEM", 0, "COUNT")
        print(f"\nTotal Contact Elements Active: {count}")
        self.mapdl.allsel()

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
    DAT_FILE = "actuator_setup_viscoelasticity_1Pa.dat"
    
    # Create the environment
    # try/finally block to ensure ANSYS closes if code crashes
    env = None
    try:
        env = AnsysSoftActuatorEnv(
            dat_path=DAT_FILE,
            #min_presure=0.0,
            #max_pressure=150000.0,
            target_deformation=0.03, 
            log_level="INFO"
        )
        
        # Reset environment
        print("Resetting environment...")
        start_reset = time.time()
        obs, _ = env.reset()
        end_reset = time.time()
        print(f"Reset took: {end_reset - start_reset:.4f} seconds")
        print(f"Initial Observation: {obs}")
        
        # Run for n_steps with random actions
        N_STEPS = 1
        for i in range(N_STEPS):
            # Sample a random pressure from action space
            random_action = [1.0] #env.action_space.sample()
            
            print(f"\n--- Starting Step {i+1} ---")
            start_time = time.time()

            # Step the environment
            obs, reward, done, truncated, info = env.step(random_action)
            
            # END TIMER
            end_time = time.time()
            duration = end_time - start_time
            print(f"Step {i+1} Calculation Time: {duration:.4f} seconds")

            print(f"Step {i+1}:")
            print(f"  Applied Pressure: {info['pressure_applied']:.2f} Pa")
            print(f"  Resulting Deform: {obs[0]*100:.6f} cm")
            print(f"  Reward: {reward:.4f}")
            env.render()
            if done:
                print("Target Reached!")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if env:
            print("Closing ANSYS...")
            env.close()