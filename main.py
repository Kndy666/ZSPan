import time
import wandb
import toml
from tqdm import tqdm
from main_SDE import main_run as main_sde_run
from main_RND import main_run as main_rnd_run
from test import main_run as main_test_run
from evaluate import main_run as main_evaluate_run

def run_pipeline(config):
    """
    Run the entire pipeline for the given satellite and sample range, using the config loaded from TOML.
    """
    success = True
    common_config = config['common']

    if isinstance(config['common']['sample_range'], int):
        sample_range = [config['common']['sample_range']]
    else:
        sample_range = range(config['common']['sample_range'][0], config['common']['sample_range'][1])
    common_config.pop('sample_range', None)

    project_name = config['wandb']['project_name']
    run_name = config['wandb']['run_name']
    note = config['wandb']['note']
    enable_wandb = config['wandb'].get('enable_wandb', False)

    exclude_common_sde = config['sde'].get('exclude_common', [])
    exclude_common_rnd = config['rnd'].get('exclude_common', [])
    exclude_common_test = config['test'].get('exclude_common', [])
    exclude_common_evaluate = config['evaluate'].get('exclude_common', [])

    for exclude_key in exclude_common_sde:
        params_sde.pop(exclude_key, None)
    for exclude_key in exclude_common_rnd:
        params_rnd.pop(exclude_key, None)
    for exclude_key in exclude_common_test:
        params_test.pop(exclude_key, None)
    for exclude_key in exclude_common_evaluate:
        params_evaluate.pop(exclude_key, None)

    params_sde = {**common_config, **config['sde'], 'sample': sample_range, 'enable_wandb': enable_wandb}
    params_rnd = {**common_config, **config['rnd'], 'sample': sample_range, 'enable_wandb': enable_wandb}
    params_test = {**common_config, **config['test'], 'sample': sample_range, 'enable_wandb': enable_wandb}
    params_evaluate = {**common_config, **config['evaluate'], 'sample': sample_range, 'enable_wandb': enable_wandb}

    prefix_dict = {
        "sde": params_sde,
        "rnd": params_rnd,
        "test": params_test,
        "evaluate": params_evaluate
    }

    # Combine parameters into a config dictionary for W&B logging
    wb_config = {}
    for script, params in prefix_dict.items():
        for key, value in params.items():
            if key in ["batch_size", "epochs"]:
                new_key = f"{script}_{key}"
                wb_config[new_key] = value
            else:
                wb_config[key] = value

    # Log configuration to W&B
    if enable_wandb:
        run = wandb.init(project=project_name, name=run_name, config=wandb.helper.parse_config(wb_config, exclude=("file_path", "device", "enable_wandb", "sample")), notes=note)
        wandb.define_metric("epoch")

    # Run SDE, RND, Test models sequentially and then evaluate
    try:
        tqdm.write(f"Running SDE for sample {sample_range}...")
        main_sde_run(**params_sde)
        tqdm.write(f"Running RND for sample {sample_range}...")
        main_rnd_run(**params_rnd)
        tqdm.write(f"Running Test for sample {sample_range}...")
        main_test_run(**params_test)
        tqdm.write(f"Evaluating model for sample {sample_range}...")
        main_evaluate_run(**params_evaluate)
    except Exception as e:
        tqdm.write(f"Error for sample {sample_range}: {e}")
        success = False
    finally:
        if enable_wandb:
            run.finish()

    return success

def start_pipeline(config_file="config.toml"):
    """
    Start the entire pipeline and measure execution time.
    """
    t_start = time.time()

    config = toml.load(config_file)
    success = run_pipeline(config)
    
    t_end = time.time()

    if success:
        tqdm.write("All scripts ran successfully.")
    else:
        tqdm.write("Script execution failed.")
    tqdm.write(f"Total execution time: {t_end - t_start:.2f}s")

if __name__ == "__main__":
    start_pipeline()