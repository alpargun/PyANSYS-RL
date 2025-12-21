import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from datetime import datetime

from ansys.mapdl.core import launch_mapdl

class AnsysSoftActuatorEnv(gym.Env):
    """
    Custom Environment built on gymnasium interface.
    Connects to ANSYS MAPDL to control a soft pneumatic actuator.
    """
    
    def __init__(self, dat_path, target_deformation=0.015, log_level="INFO"):
        super(AnsysSoftActuatorEnv, self).__init__()
        
        # CONFIGURATION
        self.dat_path = dat_path # .dat file for the solution from ANSYS Mechanical
        self.target_deformation = target_deformation # Target elongation (meters)
        self.max_pressure = 150000.0 # Max pressure in Pascals
        self.actuation_axis = "X" # Axis along which deformation is measured
        self.current_step_count = 0   # Track steps for "done" logic
        
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
        # Upload the file to the solver's working directory
        self.mapdl.upload(self.dat_path)
        self.mapdl.clear()
        self.mapdl.prep7() # Enter Pre-processor
        self.mapdl.run('SHPP, OFF') # Disable shape checking errors
        self.mapdl.run('/NERR, 0, -1') # Suppress error dialogs
        
        # Read the file
        self.mapdl.input(self.dat_path)

        # FIX for INCREASE ERROR LIMIT (The number of ERROR and WARNING messages exceeds 10000)
        # Allow up to 10 million warnings (effectively infinite)
        self.mapdl.run('/NERR, 10000000, 10000000, -1') # '-1' means "Do not abort"
        # Disable the "Element/Node" integrity check warning printing
        self.mapdl.run('/NOLIST')

        # SIZE and MATERIAL CHECK ----------------------------------------------
        # Check if the size of the model is in Meters or Millimeters.
        self.mapdl.allsel()
        xmin = self.mapdl.get_value("NODE", 0, "MNLOC", "X")
        xmax = self.mapdl.get_value("NODE", 0, "MXLOC", "X")
        model_len = xmax - xmin
        print(f"\n--- CONFIGURING UNITS & MATERIAL ---")
        print(f"Detected Model Length: {model_len:.4f}")
        # Verify materials are correct (Hyperelastic for soft robots)
        self.check_material_properties(1)
        # ----------------------------------------------------------------------
        
        # Check Node Count
        node_count = self.mapdl.mesh.n_node
        print(f"Model Loaded. Node Count: {node_count}")
        if node_count > 128000:
            print("WARNING: Node count exceeds Student License limit (128k). Solver might crash.")

        self.mapdl.finish()

        # ACTION SPACE
        # Action: Continuous pressure value between 0 and Max Pressure
        # The agent will output values between -1 and 1.
        # Scale this to 0 - 100,000 Pa later.
        self.action_space = spaces.Box(
            low=np.array([-1.0]), #low=np.array([0.0]), 
            high=np.array([1.0]), #high=np.array([self.max_pressure]),
            shape=(1,), # we control 1 variable (Pressure)
            dtype=np.float32
        )

        # OBSERVATION SPACE
        # Observation: Current Deformation in X
        self.observation_space = spaces.Box(
            low=np.array([-0.3]), # assuming max -0.3 m deformation
            high=np.array([0.3]), # assuming max +0.3 m deformation
            shape=(1,), 
            dtype=np.float32
        )
        
    def step(self, action):
        self.current_step_count += 1
        # Unpack Action and clip it to ensure it stays within bounds
        #pressure_val = np.clip(action[0], 0, self.max_pressure)
        raw_action = np.clip(action[0], -1.0, 1.0)
        pressure_val = ((raw_action + 1.0) / 2.0) * self.max_pressure
        
        self.mapdl.prep7()

        #  ANCHOR BASE (Robust Coordinate Selection) ---------------------------
        # Select nodes near X=0 and lock them
        self.mapdl.nsel('S', 'LOC', 'X', -0.001, 0.001)
        if self.mapdl.mesh.n_node > 0:
            self.mapdl.d('ALL', 'ALL', 0)
        # Select the fixed support
        self.mapdl.cmsel('S', 'FixedSupport')
        # 2. Lock it (Displacement = 0 in All Directions)
        # 'D' command applies displacement constraints
        self.mapdl.d('ALL', 'ALL', 0)

        # APPLY PRESSURE LOAD --------------------------------------------------
        # Select the faces for pressure (Named Selection 'Inner1new')
        self.mapdl.cmsel('S', 'Inner1new')
        # Select ALL elements attached to these nodes (Solids + Surfs)
        self.mapdl.esln('S')
        # THE FIX: Unselect Surface Effect Elements (Type 154)
        # This removes the "Skin" from the active set so we don't double-load it.
        # We are left with ONLY the Solid elements (Type 180-189).
        self.mapdl.esel('R', 'ENAME', '', 154)

        # Apply Pressure to the remaining (Solid) elements
        # Since the Surfs are hidden, the pressure only applies once.
        self.mapdl.sfe('ALL', 1, 'PRES', '', pressure_val)
        
        # Select everything again for solving
        self.mapdl.allsel()
        
        # RUN SOLVER -----------------------------------------------------------
        self.mapdl.run('/SOLU')
        self.mapdl.antype('STATIC') # Ensure Static Analysis
        # CRITICAL: Re-enforce Large Deflection & Substepping every step
        # This prevents the "Explosion" where deformation jumps to 900 meters.
        self.mapdl.nlgeom('ON') # Nonlinear Geometry
        # FASTER SOLVER
        self.mapdl.eqslv('SPARSE') # Sparse is usually best for nonlinear rubber models < 100k nodes

        # SOLVE (RAMP) ---------------------------------------------------------
        self.mapdl.kbc(0) # 0 = RAMPED loading (linear interpolation)
        #self.mapdl.time(1.0) # Set virtual "End Time" for the step
        # Max 100 steps if it struggles, min 5 steps if stable enough.
        self.mapdl.nsubst(5, 100, 5) # Old: nsubst(50, 1000, 50)
        self.mapdl.neqit(20) # If can't solve in 20 iters, cut the step size rather than grinding.

        # REDUCE FILE IO (for RL speed) -------------------------
        # Only write the LAST substep to the file (prevents disk bloat)
        ULTRA_FAST = True
        if not ULTRA_FAST:
            self.mapdl.outres('ALL', 'LAST')
        else:
            # --- THE ULTRA-FAST SETTING ---
            self.mapdl.outres('ERASE') # Clear previous output settings
            # Write ONLY Nodal Solution (NSOL) -> Displacement
            # We skip stresses (RSOL), strains (ESOL), etc.
            # We write ONLY the Last substep ('LAST')
            self.mapdl.outres('NSOL', 'LAST')
            # In Ultra-Fast setting, you cannot plot Stress or Strain (Von Mises) later,
            # because that data is not saved.
            # Displacement (plot_nodal_displacement) WORKS (all we need for RL).
        
        self.mapdl.solve()
        self.mapdl.finish()
        
        # CHECK FOR SOLVER CONVERGENCE -----------------------------------------
        # If ANSYS failed to solve (diverged), the results are garbage.
        if not self.mapdl.solution.converged:
            print(f"DEBUG: Solver Diverged at {pressure_val:.0f} Pa")
            # Return current state as 0, Penalty -50, End Episode
            return np.array([0.0], dtype=np.float32), -50.0, True, False, {"pressure": pressure_val}

        # GET OBSERVATION ------------------------------------------------------
        # Get Observation (Deformation)
        self.mapdl.post1()
        self.mapdl.set('LAST') # Read the final converged substep
        self.mapdl.allsel()
        # Calculate max deformation across all nodes
        self.mapdl.nsort('U', self.actuation_axis) # Sort nodes by deformation
        
        # Measure Extension Relative to Base (Tip - Base)
        # This is immune to base sliding.
        # Get Tip (Min Displacement - assuming extension in -X)
        tip_disp = self.mapdl.get_value('SORT', 0, 'MIN')        
        # Get Base (Max Displacement - usually 0)
        self.mapdl.nsel('S', 'LOC', 'X', -0.001, 0.001) # Check for sliding
        self.mapdl.cmsel('S', 'FixedSupport')
        self.mapdl.nsort('U', self.actuation_axis)
        base_disp = self.mapdl.get_value('SORT', 0, 'MAX')

        # Calculate pure extension magnitude
        cur_deformation = abs(tip_disp - base_disp)

        # ======================================================================
        # OLD EXTENSION CALCULATION --------------------------------------------
        # # Get BOTH extremes
        # max_deformation = self.mapdl.get_value('SORT', 0, 'MAX')
        # min_deformation = self.mapdl.get_value('SORT', 0, 'MIN')
        # # Take the largest magnitude (Absolute Value)
        # cur_deformation = max(abs(max_deformation), abs(min_deformation))
        # ======================================================================

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
        
        # In FEA, reset usually means unloading the structure
        self.mapdl.prep7()
        self.mapdl.sf('ALL', 'PRES', 0) # Remove pressure
        
        # Run a quick solve to zero out deformations
        self.mapdl.run('/SOLU')
        self.mapdl.antype('STATIC')
        #self.mapdl.kbc(1)        
        self.mapdl.nsubst(1, 1, 1)
        self.mapdl.solve()
        self.mapdl.finish()
        
        # Get the zeroed observation
        #self.mapdl.post1()
        #zero_def = self.mapdl.get_value('NODE', self.tip_node_id, 'U', 'X')
        #if zero_def is None: zero_def = 0.0
        
        return np.array([0.0], dtype=np.float32), {}

    def render(self):
        """Visualizes the robot state"""
        self.mapdl.allsel() # Ensure all entities are selected
        self.mapdl.post1()
        self.mapdl.set('LAST')
        print("Rendering...")
        self.mapdl.post_processing.plot_nodal_displacement(
            component=self.actuation_axis,
            show_edges=True,
            cmap="jet",
            cpos="xy",
            displacement_factor=1.0, # 1.0 = True Scale. Change to 0.0 to see undeformed.
            overlay_wireframe=True # Shows the original grey mesh outline for comparison
        )

    def check_material_properties(self, mat_id):
        """Helper to print what ANSYS sees for the material"""
        print(f"--- VERIFYING MATERIAL {mat_id} ---")
        # Check Hyperelastic Table
        output = self.mapdl.run(f"TBLIST, ALL, {mat_id}")
        print(output)
        if "HYPER" in output or "NEO" in output:
            print(f"SUCCESS: Material {mat_id} is Hyperelastic (Rubber).")
            # print(output) # Uncomment to see full table
        else:
            print(f"WARNING: Material {mat_id} looks like STEEL or Undefined.")


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
    DAT_FILE = "actuator_setup_150kPa.dat"
    
    # Create the environment
    # try/finally block to ensure ANSYS closes if code crashes
    env = None
    try:
        env = AnsysSoftActuatorEnv(
            dat_path=DAT_FILE, 
            target_deformation=0.03, 
            log_level="INFO"
        )
        
        # Reset environment
        obs, _ = env.reset()
        print(f"Initial Observation: {obs}")
        
        # Run for n_steps with random actions
        N_STEPS = 1
        for i in range(N_STEPS):
            # Sample a random pressure from action space
            random_action = [1.0] #env.action_space.sample()
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(random_action)
            
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