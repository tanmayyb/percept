import argparse
import subprocess
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import signal
import time
import sys

class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.yellow = '\033[93m'  # ANSI escape code for yellow
        self.reset = '\033[0m'    # ANSI escape code to reset color

    def emit(self, record):
        try:
            msg = self.format(record)
            # Add yellow color to error messages
            if record.levelno >= logging.ERROR:
                msg = f"{self.yellow}{msg}{self.reset}"
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(plan_dir):
    """Setup logging configuration using a file handler and TqdmLoggingHandler."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(plan_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'run_plan_{timestamp}.log')
    
    # Create file handler for logging to file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Clear any existing handlers.
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Add our custom TqdmLoggingHandler for console output.
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Set up the root logger with both handlers.
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    return log_file

def read_plan(plan_dir):
    """Read the run.plan file and return workload paths."""
    plan_file = os.path.join(plan_dir, 'run.plan')
    if not os.path.exists(plan_file):
        raise FileNotFoundError(f"Could not find run.plan file in {plan_dir}")
    
    with open(plan_file, 'r') as f:
        plan = yaml.safe_load(f)
    
    return plan.get('workloads', [])

def run_workload(workload_path, duration):
    cmd = [
        'ros2', 'launch', 'mp_eval', 'launch_workload.py',
        f'workload:={workload_path}',
        'disable_metrics:=true'
    ]
    if duration:
        cmd.append(f'timed_run:={duration}')
    
    logging.info(f"Running workload: {workload_path}")
    logging.info(f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid  # For Unix. On Windows use creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    def sigint_handler(signum, frame):
        logging.info("User initiated SIGINT. Propagating SIGINT to the subprocess.")
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

    old_handler = signal.signal(signal.SIGINT, sigint_handler)
    
    try:
        stdout, stderr = process.communicate()
    except Exception as e:
        logging.error(f"Exception while waiting for workload: {e}")
        process.kill()
        stdout, stderr = process.communicate()
        return False
    finally:
        signal.signal(signal.SIGINT, old_handler)

    if process.returncode != 0:
        logging.error(f"Workload {workload_path} exited with code {process.returncode}")
        logging.error(f"Output:\n{stdout}")
        logging.error(f"Error:\n{stderr}")
        return False

    logging.info(f"Workload {workload_path} completed successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run a plan of workloads')
    parser.add_argument('plan_dir', type=str, help='Directory containing the run.plan file')
    parser.add_argument('--duration', type=float, help='Duration for each workload in seconds')
    parser.add_argument('--wait', type=float, default=10.0, help='Wait time between workloads in seconds')
    args = parser.parse_args()

    plan_dir = os.path.abspath(args.plan_dir)
    
    log_file = setup_logging(plan_dir)
    logging.info(f"Starting plan execution from directory: {plan_dir}")
    logging.info(f"Log file: {log_file}")

    try:
        workloads = read_plan(plan_dir)
        logging.info(f"Found {len(workloads)} workloads to execute")

        successful_workloads = 0
        # Explicitly set the output stream to sys.stderr if desired.
        with tqdm(total=len(workloads), desc="Running workloads", position=0, leave=True, file=sys.stderr) as pbar:
            for i, workload in enumerate(workloads):
                if run_workload(workload, args.duration):
                    successful_workloads += 1
                if args.wait > 0 and i < len(workloads) - 1:
                    logging.info(f"Waiting {args.wait} seconds before next workload...")
                    time.sleep(args.wait)
                pbar.update(1)

        logging.info(f"Plan execution completed. Successful workloads: {successful_workloads}/{len(workloads)}")
    except Exception as e:
        logging.error(f"Error executing plan: {str(e)}", exc_info=True)
        return 1

    return 0 if successful_workloads == len(workloads) else 1

if __name__ == '__main__':
    exit(main())
