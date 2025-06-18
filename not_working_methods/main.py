import subprocess
import os

def run_extract_script():
    extract_command = ["python", "extract.py", "--input", "input"]
    print("Running extract.py...")
    result = subprocess.run(extract_command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running extract.py:")
        print(result.stderr)
        return False

    print("extract.py completed successfully.")
    return True

def run_eye_script():
    eye_command = ["python", "eye.py"]
    print("Running eye.py...")
    result = subprocess.run(eye_command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running eye.py:")
        print(result.stderr)
        return False

    print("eye.py completed successfully.")
    return True

def run_status_script():
    status_command = ["python", "status.py"]
    print("Running status.py...")
    result = subprocess.run(status_command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running status.py:")
        print(result.stderr)
        return False

    print("status.py completed successfully.")
    return True

if __name__ == "__main__":
    if not run_extract_script():
        print("Aborting due to extract.py failure.")
        exit(1)

    if not run_eye_script():
        print("Aborting due to eye.py failure.")
        exit(1)

    if not run_status_script():
        print("Aborting due to status.py failure.")
        exit(1)

    print("All scripts executed successfully.")