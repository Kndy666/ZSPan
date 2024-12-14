import subprocess
import time
from tqdm import tqdm

def run_script(script, params):
    """
    Run a Python script with specified parameters.

    Args:
        script (str): The script name to execute.
        params (dict): The parameters to pass to the script.

    Returns:
        bool: True if the script runs successfully, False otherwise.
    """
    cmd = ["python", script]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            tqdm.write(f"Error running {script} with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        tqdm.write(f"Exception while running {script}: {e}")
        return False

def run_main_scripts(satellite, sample_range):
    """
    Run the main scripts sequentially for each sample.

    Args:
        satellite (str): Satellite type.
        sample_range (int or tuple): Single sample or range of samples to process.

    Returns:
        bool: True if all scripts run successfully, False otherwise.
    """
    success = True
    if isinstance(sample_range, int):
        samples = [sample_range]
    else:
        samples = range(sample_range[0], sample_range[1])

    for sample in tqdm(samples, desc="Samples", colour='green'):
        # Set scripts and parameters for each sample
        scripts_with_params = {
            "main_SDE.py": {
                "lr": 0.0005,
                "epochs": 250,
                "batch_size": 8,
                "device": "cuda",
                "satellite": satellite,
                "file_path": "../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5",
                "sample": sample
            },
            "main_RND.py": {
                "lr_fug": 0.0005,
                "lr_rsp": 0.001,
                "epochs": 250,
                "batch_size": 8,
                "device": "cuda",
                "satellite": satellite,
                "file_path": "../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5",
                "sample": sample
            },
            "test.py": {
                "satellite": satellite,
                "sample": sample
            }
        }
        # Run each script sequentially
        for script, params in scripts_with_params.items():
            if not run_script(script, params):
                tqdm.write(f"Execution failed for {script} at sample {sample}. Stopping pipeline.")
                success = False
                break
        if not success:
            break
    return success

if __name__ == "__main__":
    # Define satellite type
    current_satellite = "wv3"
    print(f"Satellite type: {current_satellite}")

    # Define sample range
    sample_range = (0, 20)

    # Measure execution time
    t_start = time.time()
    success = run_main_scripts(current_satellite, sample_range)
    t_end = time.time()

    # Final status
    if success:
        tqdm.write("All scripts ran successfully.")
    else:
        tqdm.write("Script execution failed.")
    tqdm.write(f"Total execution time: {t_end - t_start:.2f}s")
