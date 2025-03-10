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
import shutil
yellow = '\033[93m'
red = '\033[91m'
green = '\033[92m'
reset = '\033[0m'

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
    
    return plan.get('workloads', []), plan.get('runtime', [])

def copy_logs(eval_dir, workload_log_dir, timestamp):
    for log_file in ['planner.log', 'percept.log']:
        src = eval_dir / 'results' / log_file
        if os.path.exists(src):
            dst = workload_log_dir / f"{timestamp}_{log_file}"
            shutil.copy2(src, dst) 
    logging.info(f"Copied logs to {workload_log_dir}")

def run_workload(workload_path, plan_dir, duration, enable_metrics):
    cmd = [
        'ros2', 'launch', 'mp_eval', 'launch_workload.py',
        f'workload:={workload_path}',
        f'disable_metrics:={"false" if enable_metrics else "true"}'
    ]
    if duration:
        cmd.append(f'timed_run:={duration}')
    
    logging.info(f"{yellow}Running workload: {workload_path}{reset}")
    logging.info(f"Command: {' '.join(cmd)}")

    # Create timestamped workload results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workload_name = os.path.splitext(os.path.basename(workload_path))[0]
    eval_dir = Path(os.path.dirname(__file__))
    plan_dir_name = os.path.basename(plan_dir)
    workload_log_dir = eval_dir / 'results' / plan_dir_name / 'logs' / workload_name
    os.makedirs(workload_log_dir, exist_ok=True)
    
    # Create log file for subprocess output
    output_log = workload_log_dir / f"{timestamp}_mp_eval.log"
    with open(output_log, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            preexec_fn=os.setsid
        )

        def sigint_handler(signum, frame):
            logging.info(f"{yellow}User initiated SIGINT. Propagating SIGINT to the subprocess.{reset}")
            os.killpg(os.getpgid(process.pid), signal.SIGINT)

        old_handler = signal.signal(signal.SIGINT, sigint_handler)
        
        try:
            start_time = time.time()
            process.wait()  # Wait for process to complete
            actual_duration = time.time() - start_time
            
        except Exception as e:
            logging.error(f"Exception while waiting for workload: {e}")
            process.kill()
            return False
        finally:
            signal.signal(signal.SIGINT, old_handler)

    if process.returncode != 0:
        logging.error(f"Workload {workload_path} exited with code {process.returncode}")
        with open(output_log, 'r') as f:
            output = f.read()
            logging.error(f"See detailed output in: {output_log}")
        return False

    copy_logs(eval_dir, workload_log_dir, timestamp)
    # Check if workload ended significantly earlier than expected
    if duration and actual_duration < duration * 0.9:  # 10% threshold
        logging.warning(
            f"{red}Workload completed in {actual_duration:.1f}s, "
            f"significantly shorter than expected {duration:.1f}s{reset}"
        )
    else:
        logging.info(f"{green}Workload {workload_path} completed successfully.{reset}")    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run a plan of workloads')
    parser.add_argument('plan_dir', type=str, help='Directory containing the run.plan file')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration for each workload in seconds, 0.0 means dynamic loading')
    parser.add_argument('--enable_metrics', action='store_true', help='Enable metrics collection')
    parser.add_argument('--wait', type=float, default=10.0, help='Wait time between workloads in seconds')
    args = parser.parse_args()

    plan_dir = os.path.abspath(args.plan_dir)
    
    log_file = setup_logging(plan_dir)
    start_time = time.time()  # Start timing
    logging.info(f"Starting plan execution from directory: {plan_dir}")
    logging.info(f"Log file: {log_file}")

    try:
        workloads, runtimes = read_plan(plan_dir)
        if args.duration == 0.0 and len(runtimes) > len(workloads):
            logging.error(f"Number of runtimes ({len(runtimes)}) is greater than the number of workloads ({len(workloads)}). Cannot  implement runtime with dynamic loading!")
            return 1
        logging.info(f"Found {len(workloads)} workloads to execute")

        successful_workloads = 0
        # Explicitly set the output stream to sys.stderr if desired.
        with tqdm(total=len(workloads), desc="Running workloads", position=0, leave=True, file=sys.stderr) as pbar:
            for i, workload in enumerate(workloads):
                if args.duration == 0.0:
                    duration = runtimes[i]
                else:
                    duration = args.duration
                if run_workload(workload, plan_dir, duration, args.enable_metrics):
                    successful_workloads += 1
                if args.wait > 0 and i < len(workloads) - 1:
                    logging.info(f"Waiting {args.wait} seconds before next workload...")
                    logging.info(f"You can cancel the wait by pressing Ctrl+C")
                    try:
                        time.sleep(args.wait)
                    except KeyboardInterrupt:
                        logging.info(f"{red}Wait interrupted by user. Exiting.{reset}")
                        return 1
                pbar.update(1)

        logging.info(f"Plan execution completed. Successful workloads: {successful_workloads}/{len(workloads)}")
        total_time = time.time() - start_time  # Calculate total time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        logging.info(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    except Exception as e:
        logging.error(f"Error executing plan: {str(e)}", exc_info=True)
        return 1

    return 0 if successful_workloads == len(workloads) else 1

if __name__ == '__main__':
    exit(main())
