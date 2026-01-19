import subprocess
import sys
import time


def run_command(command):
    print(f"Running: {command}")
    start = time.time()
    result = subprocess.run(command, shell=True)
    end = time.time()

    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(result.returncode)
    else:
        print(f"Command finished in {end - start:.2f} seconds.\n")


def main():
    print("========================================")
    print("Starting Model Training Pipeline")
    print("========================================")

    run_command(f"{sys.executable} model/model_setup.py")

    # run_command(f"{sys.executable} model/model_test.py")

    print("========================================")
    print("Pipeline Completed Successfully")
    print("========================================")


if __name__ == "__main__":
    main()
