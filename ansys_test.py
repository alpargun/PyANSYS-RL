import os
from ansys.mapdl.core import launch_mapdl

# 1. Update this path to your exact executable location
student_exe = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\ansys\bin\winx64\ansys252.exe"

# 2. Use a clean folder (Student version often fails in Temp folders)
run_dir = r"C:\Ansys_Test_Run"
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

print("Attempting to launch ANSYS Student (Empty)...")

try:
    # Launch with Student-safe flags
    mapdl = launch_mapdl(
        exec_file=student_exe,
        run_location=run_dir,
        nproc=2,                 # Force 2 cores
        additional_switches="-smp", # Force Shared Memory
        override=True,
        loglevel="INFO"
    )
    print("\nSUCCESS! ANSYS Launched and is ready.")
    print(mapdl)
    
    mapdl.exit()
    print("ANSYS Closed normally.")

except Exception as e:
    print("\nFAILURE: ANSYS could not start.")
    print(f"Error details: {e}")