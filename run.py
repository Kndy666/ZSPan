import subprocess
import time

def run_script(script, **kwargs):
    cmd = ["python", script]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script}")
        return False
    return True

def run_main_scripts(satellite):
    # Define the scripts to be run sequentially
    sequential_scripts = ["main_SDE.py", "main_RND.py", "test.py"]

    # Parameters for main_SDE.py
    sde_params = {
        "lr": 0.0005,
        "epochs": 250,
        "batch_size": 1,
        "device": 'cuda',
        "satellite": satellite,
        "file_path": "../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5",
    }

    # Parameters for main_RND.py
    rnd_params = {
        "lr_fug": 0.0005,
        "lr_rsp": 0.001,
        "epochs": 140,
        "device": 'cuda',
        "satellite": satellite,
        "file_path": "../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5",
    }

    # Parameters for test.py
    test_params = {
        "satellite": satellite
    }

    # Run sequential scripts
    for script in sequential_scripts:
        if script == "main_SDE.py":
            if not run_script(script, **sde_params):
                return False
        elif script == "main_RND.py":
            if not run_script(script, **rnd_params):
                return False
        elif script == "test.py":
            if not run_script(script, **test_params):
                return False
        else:
            if not run_script(script):
                return False

    return True

if __name__ == "__main__":
    current_satellite = 'wv3'
    print(f'Satellite is {current_satellite}')
    t1 = time.time()
    if run_main_scripts(current_satellite):
        print("Running scripts completed successfully.")
    else:
        print("Script execution failed. Exiting...")
    t2 = time.time()
    print(f'Total time: {t2 - t1:.2f}s')
