import os
import time
import re
from typing import Tuple, Optional, Dict, List
from urllib.parse import urlparse
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.classes.github_api import GitHubApi


def reviewedness_metric(github_str: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calculates the fraction of code (not weights) introduced through reviewed pull requests.
    
    This metric evaluates code review practices by analyzing:
    - What percentage of code changes went through pull requests
    - What percentage of those PRs had actual code reviews
    
    Args:
        github_str (str): GitHub repository URL (e.g., "https://github.com/owner/repo")
        verbosity (int): The verbosity level (0=silent, 1=INFO, 2=DEBUG)
        log_queue (multiprocessing.Queue): The queue for centralized logging
    
    Returns:
        A tuple containing:
        - The reviewedness score: -1 if no GitHub repo, otherwise 0.0-1.0
        - The total time spent (float)
    """
    pid = os.getpid()
    start_time = time.perf_counter()
    score = -1.0  # Default for no GitHub repo
    
    try:
        # Normalize input
        github_str_norm = (github_str or "").strip()
        
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Reviewedness: Starting analysis...")
        
        # Check if we have a GitHub URL - return 0.0 instead of -1.0 to not penalize
        if not github_str_norm or not _is_github_url(github_str_norm):
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [INFO] Reviewedness: No GitHub repository linked -> score=0.0")
            time_taken = time.perf_counter() - start_time
            return 0.0, time_taken
        
        # Parse GitHub URL to extract owner and repo
        owner, repo = _parse_github_url(github_str_norm)
        
        if not owner or not repo:
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [INFO] Reviewedness: Could not parse GitHub URL -> score=0.0")
            time_taken = time.perf_counter() - start_time
            return 0.0, time_taken
        
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Analyzing {owner}/{repo}")
        
        # Initialize GitHub API client
        try:
            gh_api = GitHubApi(owner, repo)
            gh_api.set_bearer_token_from_env("GITHUB_TOKEN")
        except Exception as e:
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [INFO] Reviewedness: GitHub API setup failed -> score=0.0")
            time_taken = time.perf_counter() - start_time
            return 0.0, time_taken
        
        # Get repository statistics
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Reviewedness: Fetching repository data...")
        
        stats = _calculate_reviewedness_stats(gh_api, owner, repo, verbosity, log_queue, pid)
        
        if stats is None:
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [INFO] Reviewedness: Could not fetch repository data -> score=0.0")
            score = 0.0
        else:
            score = stats['reviewedness_score']
            
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [INFO] Reviewedness: Final score = {score:.2f}")
            
            if verbosity >= 2:
                log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Total commits = {stats['total_commits']}")
                log_queue.put(f"[{pid}] [DEBUG] Reviewedness: PR commits = {stats['pr_commits']}")
                log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Reviewed PRs = {stats['reviewed_prs']}/{stats['total_prs']}")
    
    except Exception as e:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] calculating reviewedness: {e}")
        score = 0.0
    
    time_taken = time.perf_counter() - start_time
    
    if verbosity >= 1:
        log_queue.put(f"[{pid}] [INFO] Finished reviewedness calculation. Score={score:.2f}, Time={time_taken:.3f}s")
    
    return score, time_taken


def _is_github_url(url: str) -> bool:
    """Check if the URL is a valid GitHub URL."""
    url_lower = url.lower()
    return url_lower.startswith("https://github.com/") or url_lower.startswith("http://github.com/")


def _parse_github_url(url: str) -> Tuple[str, str]:
    """
    Parse a GitHub URL to extract owner and repo name.
    
    Examples:
        https://github.com/owner/repo -> ("owner", "repo")
        https://github.com/owner/repo.git -> ("owner", "repo")
        https://github.com/owner/repo/tree/branch -> ("owner", "repo")
    
    Returns:
        Tuple of (owner, repo) or ("", "") if parsing fails
    """
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].replace(".git", "")
            return owner, repo
    except Exception:
        pass
    
    return "", ""


def _calculate_reviewedness_stats(
    gh_api: GitHubApi, 
    owner: str, 
    repo: str,
    verbosity: int,
    log_queue,
    pid: int
) -> Optional[Dict[str, any]]:
    """
    Calculate reviewedness statistics for a GitHub repository.
    
    This function:
    1. Gets all merged pull requests
    2. Checks which PRs had reviews
    3. Gets commits and identifies which came from reviewed PRs
    4. Calculates the fraction of code changes that were reviewed
    
    Returns:
        Dictionary with stats or None if data cannot be fetched
    """
    try:
        # Fetch all merged pull requests
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Fetching pull requests...")
        
        pr_response = gh_api.get_repo_pulls(state="closed")
        
        if not pr_response or not isinstance(pr_response, dict):
            return None
        
        all_prs = pr_response.get('data', [])
        if not isinstance(all_prs, list):
            all_prs = [] if all_prs is None else [all_prs]
        
        # Filter for merged PRs only
        merged_prs = [pr for pr in all_prs if pr.get('merged_at') is not None]
        
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Found {len(merged_prs)} merged PRs")
        
        if not merged_prs:
            # No PRs means all commits are direct - low reviewedness
            return {
                'reviewedness_score': 0.0,
                'total_commits': 0,
                'pr_commits': 0,
                'total_prs': 0,
                'reviewed_prs': 0
            }
        
        # Check which PRs had reviews
        reviewed_prs = []
        total_additions = 0
        total_deletions = 0
        reviewed_additions = 0
        reviewed_deletions = 0
        
        # Sample PRs for efficiency (max 100 most recent)
        sample_size = min(100, len(merged_prs))
        sampled_prs = merged_prs[:sample_size]
        
        for pr in sampled_prs:
            pr_number = pr.get('number')
            additions = pr.get('additions', 0)
            deletions = pr.get('deletions', 0)
            
            total_additions += additions
            total_deletions += deletions
            
            # Check if PR had reviews
            has_review = _check_pr_has_reviews(gh_api, owner, repo, pr_number, verbosity, log_queue, pid)
            
            if has_review:
                reviewed_prs.append(pr_number)
                reviewed_additions += additions
                reviewed_deletions += deletions
        
        # Calculate reviewedness score
        total_changes = total_additions + total_deletions
        reviewed_changes = reviewed_additions + reviewed_deletions
        
        if total_changes == 0:
            score = 0.0
        else:
            score = reviewed_changes / total_changes
        
        return {
            'reviewedness_score': round(score, 2),
            'total_commits': len(merged_prs),
            'pr_commits': len(merged_prs),
            'total_prs': len(sampled_prs),
            'reviewed_prs': len(reviewed_prs)
        }
    
    except Exception as e:
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Error calculating stats: {e}")
        return None


def _check_pr_has_reviews(
    gh_api: GitHubApi,
    owner: str,
    repo: str,
    pr_number: int,
    verbosity: int,
    log_queue,
    pid: int
) -> bool:
    """
    Check if a pull request has code reviews.
    
    Returns:
        True if the PR has at least one review, False otherwise
    """
    try:
        # Build URL for PR reviews endpoint
        url = f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        
        # Make API request
        response = gh_api.get(url)
        
        if not response or not isinstance(response, dict):
            return False
        
        reviews = response.get('data', [])
        if not isinstance(reviews, list):
            reviews = [] if reviews is None else [reviews]
        
        # Check if there are any actual reviews (not just comments)
        # Valid review states: APPROVED, CHANGES_REQUESTED, COMMENTED
        actual_reviews = [
            r for r in reviews 
            if r.get('state') in ['APPROVED', 'CHANGES_REQUESTED']
        ]
        
        return len(actual_reviews) > 0
    
    except Exception as e:
        if verbosity >= 2:
            log_queue.put(f"[{pid}] [DEBUG] Reviewedness: Error checking PR#{pr_number} reviews: {e}")
        return False
