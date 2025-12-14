import argparse
import io
import os
import subprocess
import sys
import requests
import metric_caller
import src.url_class as url_class
from get_model_metrics import get_model_size, get_model_README, get_model_license
from src.json_output import build_model_output

# Prevent unending recursion if tests call run.py again
NESTED_TEST_GUARD = "__ECE461_TEST_SUITE_ACTIVE__"

def _run_tests_and_print_summary() -> None:
    """
    Discover and run unit tests, then print a one line summary
    including pass count and measured coverage.
    """
    import unittest
    import coverage

    os.environ[NESTED_TEST_GUARD] = "1"
    print("Running test suite...")

    sink = io.StringIO()
    cov = coverage.Coverage(
        branch=True,
        omit=[
            os.path.join(sys.prefix, "lib", "python*","*"),
            "*/site-packages/*",
            "*/dist-packages/*", 
            "tests/*"
        ],
    )

    cov.start()
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner(stream=sink, verbosity=0)
    result = runner.run(suite)
    cov.stop()
    cov.save()

    try:
        percent = cov.report(file=io.StringIO())
    except Exception:
        percent = 0.0

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print(
        f"{passed}/{total} test cases passed. "
        f"{int(round(percent))}% line coverage achieved."
    )
    sys.exit(0 if result.wasSuccessful() else 1)

def validate_github_token(token: str) -> bool:
    """Returns true if a GitHub token successfully authenticates to the API."""
    if not token:
        return False
    try:
        response = requests.get(
            "https://api.github.com/zen",
            headers={"Authorization": f"token {token}"},
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False

def validate_log_file_path(path: str) -> bool:
    """
    Returns true if path is nonempty, the file exists or can be created,
    and it can be opened in append mode.
    """
    if not path:
        return False
    try:
        dir_name = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if not os.access(dir_name, os.W_OK):
            return False
        
        # Check that the log file can be opened in append mode.
        with open(path, "a", encoding="ascii", errors="ignore"):
            pass
    except (OSError, IOError):
        return False
    return True

def main() -> int:
    log_level_str = os.getenv("LOG_LEVEL")
    log_file_path = os.getenv("LOG_FILE")
    github_token = os.getenv("GITHUB_TOKEN")
    
    # LOG_LEVEL: default to 0 if unset/invalid
    if log_level_str and log_level_str.isdigit() and int(log_level_str) in (0, 1, 2):
        verbosity_env = int(log_level_str)
    else:
        verbosity_env = 0

    # LOG_FILE: must always be a valid, writable path
    if log_file_path is not None:
        if not validate_log_file_path(log_file_path):
            print("ERROR: LOG_FILE is not set or is unwritable.", file=sys.stderr)
            return 1
        resolved_log_file_path = log_file_path
    else:
        # Default to ./log.txt if LOG_FILE is not set
        resolved_log_file_path = os.path.join(os.getcwd(), "log.txt")
        if not validate_log_file_path(resolved_log_file_path):
            print("ERROR: LOG_FILE is not set or is unwritable.", file=sys.stderr)
            return 1
        
    # GITHUB_TOKEN: if provided, it must be valid
    if github_token and not validate_github_token(github_token):
        print("ERROR: GITHUB_TOKEN is not set or invalid.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        prog="run", description="LLM Model Evaluator", add_help=False
    )

    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help=(
            "usage: run [-v | --verbose] [-h | --help] {install, test} | URL_FILE\n\n"
            "positional arguments:\n"
            "  install           Install any dependencies needed\n"
            "  test              Runs testing suite\n"
            "  URL_FILE          Absolute file location of set of URLs\n\n"
            "options:\n"
            "  -h, --help        Show this help message and exit\n"
            "  -v, --verbose     Enable verbose output"
        )
    )
    
    parser.add_argument(
        '-v', '--verbose', action='store_true', help=('Enable verbose output')
    )

    # Install command / test / URL file
    parser.add_argument(
        "target",
        type=str,
        help=("Choose 'install', 'test', or provide an absolute path to a URL file.")
    )

    args = parser.parse_args()

    # Final verbosity: env (if valid) > -v flag > 0
    verbosity = verbosity_env if verbosity_env is not None else (1 if args.verbose else 0)
    tasks_file = os.getenv("TASKS_FILE", "./tasks.txt")

    # Dispatch logic
    if args.target == "install":
        print("Installing dependencies...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "-r",
                "requirements.txt"
            ]
        )

    elif args.target == "test":
        # If any tests call run.py again, don't re-run the suite.
        if os.getenv(NESTED_TEST_GUARD) == "1":
            return 0
        
        _run_tests_and_print_summary()
        # Unreached in outer call because _run_tests_and_print_summary sys.exits
        return 0

    else:
        # Running URL FILE
        project_groups: list[url_class.ProjectGroup] = url_class.parse_project_file(
            args.target
        )
        metric_funcs = metric_caller.load_available_functions("src.metrics")

        for group in project_groups:
            if group.model is None:
                # Emit an empty model record so line counts match
                build_model_output(name="", category="MODEL", scores={}, latency={})
                continue

            namespace = group.model.namespace
            repo = group.model.repo
            rev = group.model.rev or "main"

            # Skip if namespace or repo is missing or empty
            if not repo:
                # Truly unusable: no repo at all -> still emit a minimal record
                build_model_output(name="", category="MODEL", scores={}, latency={})
                continue

            if not namespace:
                # Single-segment HF model: print a valid record (you can use defaults)
                build_model_output(name=repo, category="MODEL", scores={}, latency={})
                continue

            size = get_model_size(namespace, repo, rev)
            filename = get_model_README(namespace, repo, rev)
            license_value = get_model_license(namespace, repo, rev)

            # Safely build github_str
            github_str = ""
            code_obj = getattr(group, "code", None)
            if code_obj is not None:
                code_link = getattr(code_obj, "link", None)
                if isinstance(code_link, str):
                    github_str = code_link

            # Safely build dataset_name
            dataset_name = ""
            dataset_obj = getattr(group, "dataset", None)
            if dataset_obj is not None:
                dataset_repo = getattr(dataset_obj, "repo", None)
                if isinstance(dataset_repo, str):
                    dataset_name = dataset_repo

            # Build an input dictionary for metric functions
            input_dict = {
                "repo_owner": namespace,
                "repo_name": repo,
                "verbosity": verbosity,
                "model_size_bytes": size,
                "github_str": github_str,
                "dataset_name": dataset_name,
                "filename": filename,
                "license": license_value,
            }

            # Run all metrics defined in tasks.txt (or tasks_local.txt if using local Ollama) using the available functions
            scores, latency = metric_caller.run_concurrently_from_file(
                tasks_file, input_dict, metric_funcs, resolved_log_file_path
            )
            build_model_output(f"{repo}", "MODEL", scores, latency)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())