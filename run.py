import os
import sys
import argparse
import subprocess
import requests
import metric_caller
import src.url_class as url_class
from get_model_metrics import get_model_size, get_model_README, get_model_license
from src.classes.github_api import GitHubApi
from src.json_output import build_model_output


def validate_github_token(token: str) -> bool:
    """Checks if a GitHub token is valid by making a simple API call."""
    if not token:
        return False
    try:
        r = requests.get(
            "https://api.github.com/zen",
            headers={"Authorization": f"token {token}"},
            timeout=3
        )
        return r.status_code == 200
    except requests.RequestException:
        return False

def validate_log_file_path(path: str) -> bool:
    """Checks if the log file path is valid and the directory is writable."""
    if not path:
        return False
    try:
        dir_name = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if not os.access(dir_name, os.W_OK):
            return False
    except (OSError, IOError):
        return False
    return True

def main() -> int:
    log_level_str = os.getenv("LOG_LEVEL")
    log_file_path = os.getenv("LOG_FILE")
    github_token = os.getenv("GITHUB_TOKEN")
    #gen_ai_key = os.getenv('GEN_AI_STUDIO_API_KEY') # Used by a child module

    require_env = os.getenv("REQUIRE_STRICT_ENV", "0") == "1"
    
    # Strict mode: exit on bad/missing env
    if require_env:
        if log_level_str is None or not log_level_str.isdigit() or int(log_level_str) not in (0, 1, 2):
            # print("ERROR: LOG_LEVEL environment variable not set or invalid. Must be 0, 1, or 2.", file=sys.stderr)
            sys.exit(1)
        if not log_file_path or not validate_log_file_path(log_file_path):
            # print(f"ERROR: LOG_FILE environment variable not set or path is unwritable: '{log_file_path}'", file=sys.stderr)
            sys.exit(1)
        if not github_token or not validate_github_token(github_token):
            # print("ERROR: GITHUB_TOKEN environment variable not set or is invalid.", file=sys.stderr)
            sys.exit(1)
        # if not gen_ai_key:
        #     # print("ERROR: GEN_AI_STUDIO_API_KEY environment variable not set.", file=sys.stderr)
        #     sys.exit(1)
        GitHubApi.verify_token(github_token)

    # Non-strict mode: safe defaults
    verbosity_env = (
        int(log_level_str)
        if (log_level_str and log_level_str.isdigit() and int(log_level_str) in (0, 1, 2))
        else None
    )
    if log_file_path and validate_log_file_path(log_file_path):
        resolved_log_file_path = log_file_path
    else:
        resolved_log_file_path = os.path.join(os.getcwd(), "log.txt")
        os.makedirs(os.path.dirname(resolved_log_file_path), exist_ok=True)

    # If a token is provided in non-strict mode, validate gently
    if not require_env and github_token:
        if not validate_github_token(github_token):
            print("Warning: Provided GITHUB_TOKEN appears invalid; continuing without it.", file=sys.stderr)
            github_token = None
        else:
            GitHubApi.verify_token(github_token)

    parser = argparse.ArgumentParser(
        prog="run",
        description="LLM Model Evaluator",
        add_help=False
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
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
        '-v', '--verbose',
        action='store_true',
        help=('Enable verbose output')
    )

    # Install command
    parser.add_argument(
        "target",
        type=str,
        help=("Choose 'install', 'test', or provide an absolute path to a URL file.")
    )

    args = parser.parse_args()

    # Final verbosity: env (if valid) > -v flag > 0
    verbosity = verbosity_env if verbosity_env is not None else (1 if args.verbose else 0)

    # --- Dispatch logic ---
    if args.target == "install":
        # if args.verbose:
        #     print("Verbose: Installing dependencies...")  
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"])

    elif args.target == "test":
        if args.verbose:
            print("Verbose: Running tests...")
        print("Running test suite...")

    else:
        # Running URL FILE
        project_groups: list[url_class.ProjectGroup] = url_class.parse_project_file(args.target)
        x = metric_caller.load_available_functions("src.metrics")

        for i in project_groups:
            if i.model is None:
                if args.verbose:
                    print("Skipping a project with no model info.")
                continue

            namespace = i.model.namespace
            repo = i.model.repo
            rev = i.model.rev or "main"

            # Skip if namespace or repo is missing or empty
            if not repo:
                # truly unusable: no repo at all -> still emit a minimal record
                build_model_output(name="", category="MODEL", scores={}, latency={})
                continue

            if not namespace:
                # single-segment HF model: print a valid record (you can use defaults)
                build_model_output(name=repo, category="MODEL", scores={}, latency={})
                continue

            size = get_model_size(namespace, repo, rev)
            filename = get_model_README(namespace, repo, rev)
            license_value = get_model_license(namespace, repo, rev)

            # Safely build github_str
            github_str = ""
            code_obj = getattr(i, "code", None)
            if code_obj is not None:
                code_link = getattr(code_obj, "link", None)
                if isinstance(code_link, str):
                    github_str = code_link

            # Safely build dataset_name
            dataset_name = ""
            dataset_obj = getattr(i, "dataset", None)
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

            # Run all metrics defined in tasks.txt using the available functions
            scores, latency = metric_caller.run_concurrently_from_file(
                "./tasks.txt", input_dict, x, resolved_log_file_path
            )
            build_model_output(f"{repo}", "model", scores, latency)
    
    return 0

if __name__ == "__main__":
    main()