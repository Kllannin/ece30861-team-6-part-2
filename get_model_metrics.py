from src.classes.hugging_face_api import HuggingFaceApi  # adjust import to where your class is saved
from typing import Optional
import requests

def get_model_size(namespace: str, repo: str, rev: str = "main") -> float:
    """
    Calculate the total size (bytes) of all files in a Hugging Face model.
    
    If any API call fails, return 0.0
    """
    try:
        # Input validation
        if not namespace or not repo:
            return 0.0
            
        api = HuggingFaceApi(namespace, repo, rev)
        files_info = api.get_model_files_info()
        
        # Ensure files_info is a list
        if not isinstance(files_info, list):
            return 0.0
            
        # Sum file sizes, missing sizes are zero
        total_size = float(sum((f.get("size") or 0) for f in files_info if isinstance(f, dict)))
        return total_size
    except Exception as e:
        # On failure, return zero (inaccessible/missing model)
        # Could log error here for debugging: print(f"Error getting model size for {namespace}/{repo}: {e}")
        return 0.0

def get_model_README(namespace: str, repo: str, rev: str = "main") -> str:
    api = HuggingFaceApi(namespace, repo, rev)

    path_or_paths = api.download_file("model_file_download", "README.md")

    # Normalize union type (str | list[str]) to str
    if isinstance(path_or_paths, list):
        README_filepath = path_or_paths[0] if path_or_paths else ""
    else:
        README_filepath = path_or_paths

    return README_filepath

def get_model_license(namespace: str, repo: str, rev: str = "main") -> str:
    """
    Get license information from HuggingFace model with proper error handling.
    
    Args:
        namespace: Model namespace (e.g., "openai-community")
        repo: Repository name (e.g., "gpt2") 
        rev: Revision/branch (default: "main")
    
    Returns:
        License string or empty string if not found/error
    """
    
    # Input validation
    if not namespace or not repo:
        return ""
    
    try:
        # Try multiple approaches to get license
        
        # Approach 1: Check model info API
        license_info = _get_license_from_api(namespace, repo)
        if license_info:
            return license_info
        
        # Approach 2: Check tags (your original approach)
        license_info = _get_license_from_tags(namespace, repo)
        if license_info:
            return license_info
            
        # Approach 3: Fallback to README parsing
        license_info = _get_license_from_readme(namespace, repo, rev)
        if license_info:
            return license_info
            
    except Exception as e:
        print(f"Error getting license for {namespace}/{repo}: {e}")
        
    return ""

def _get_license_from_api(namespace: str, repo: str) -> Optional[str]:
    """Get license from HuggingFace API model info"""
    try:
        url = f"https://huggingface.co/api/models/{namespace}/{repo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Check multiple possible license fields
            return (data.get('cardData', {}).get('license') or 
                   data.get('license') or 
                   data.get('model_license') or
                   data.get('modelLicense'))
    except Exception:
        pass
    return None

def _get_license_from_tags(namespace: str, repo: str) -> Optional[str]:
    """Get license from model tags (original approach with error handling)"""
    try:
        url = f"https://huggingface.co/api/models/{namespace}/{repo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            tags = data.get("tags", [])
            
            # Look for license in tags
            for tag in tags:
                if isinstance(tag, str):
                    # Handle multiple possible license tag formats
                    if tag.startswith("license:"):
                        return tag.split("license:", 1)[-1]
                    elif tag.startswith("licence:"):  # British spelling
                        return tag.split("licence:", 1)[-1]
                    elif "license" in tag.lower():
                        return tag
    except Exception:
        pass
    return None

def _get_license_from_readme(namespace: str, repo: str, rev: str) -> Optional[str]:
    """Extract license from README file as fallback"""
    try:
        url = f"https://huggingface.co/{namespace}/{repo}/raw/{rev}/README.md"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            readme_text = response.text
            # Simple license extraction from README
            if "license" in readme_text.lower():
                # Extract license section - simplified example
                lines = readme_text.split('\n')
                for i, line in enumerate(lines):
                    if "license" in line.lower() and i + 1 < len(lines):
                        return lines[i + 1].strip()
    except Exception:
        pass
    return None

if __name__ == "__main__":
    # Test with multiple models
    test_models = [
        ("openai-community", "gpt2"),
        ("google", "bert-base-uncased"),
        ("microsoft", "DialoGPT-medium")
    ]
    
    for namespace, repo in test_models:
        license_info = get_model_license(namespace, repo)
        print(f"License for {namespace}/{repo}: {license_info or 'Not found'}")