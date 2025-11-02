from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urlparse

@dataclass
class Code:
    link: str
    namespace: str = ""

@dataclass
class Dataset:
    link: str
    namespace: str = ""
    repo: str = ""
    rev: str = ""

@dataclass
class Model:
    # www.huggingface.co\namespace\repo\rev
    link: str 
    namespace: str = ""
    repo: str = ""
    rev: str = ""

@dataclass
class ProjectGroup:
    code: Optional[Code] = None
    dataset: Optional[Dataset] = None
    model: Optional[Model] = None

def parse_huggingface_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a Hugging Face model URL and return (namespace, repo, rev).

    The path format is typically `/namespace/repo` or `namespace/repo/tree/rev`.

    If the URL is missing a namespace or repo part, this function returns empty string.
    The revision defaults to "main" unless a `/tree/<rev>` segment is present.
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    ''' # Require at least namespace and repo. Return empty strings if not present
    if len(parts) < 2:
        return "", "", ""
    namespace, repo = parts[0], parts[1]
    rev = "main"

    # Detect a revision in URLs like `/namespace/repo/tree/<rev>`
    if len(parts) >= 4 and parts[2] == "tree":
        rev = parts[3]
    return namespace, repo, rev'''
    if len(parts) >= 2:
        namespace, repo = parts[0], parts[1]
        rev = "main"
        if len(parts) >= 4 and parts[2] == "tree":
            rev = parts[3]
        return namespace, repo, rev

    if len(parts) == 1:
        # e.g., https://huggingface.co/distilbert-base-uncased-distilled-squad
        repo = parts[0]
        rev = "main"
        return "", repo, rev

    return "", "", ""

def parse_dataset_url(url: str) -> str:
    """
    Parse a dataset URL and return the appropriate identifier for loading.
    
    - Hugging Face datasets: returns only the repo name
        Example: 
            https://huggingface.co/datasets/stanfordnlp/imdb -> "imdb"
            https://huggingface.co/datasets/glue -> "glue"
    
    - GitHub repos: returns the full URL (used directly for git clone)
        Example:
            https://github.com/zalandoresearch/fashion-mnist -> "https://github.com/zalandoresearch/fashion-mnist"
    
    Raises:
        ValueError: if the URL is not recognized.
    """
    parsed = urlparse(url)

    # Case: Hugging Face dataset
    if "huggingface.co" in parsed.netloc:
        parts = parsed.path.strip("/").split("/")
        # Expect `datasets/<...>`, but could be different
        if parts:
            # If the first segment is 'datasets', use the last part as the repo name
            if parts[0] == "datasets":
                return parts[-1] if parts else ""
            # If the path does not start with 'datasets', still return last part
            return parts[-1] if parts else ""
        # No path segments -> return empty string
        return ""

    # Case: GitHub dataset
    if "github.com" in parsed.netloc:
        return url  # keep full URL for git clone

    # Case: Unknown host
    raise ValueError(f"Unsupported dataset URL: {url}")
    

def parse_project_file(filepath: str) -> List[ProjectGroup]:
    """
    Parse a text file where each line has format:
        code_link,dataset_link,model_link

    Each line corresponds to a grouped set of links.
    Empty fields are allowed (e.g., ',,http://model.com').

    Args:
        filepath: Path to the input file.

    Returns:
        A list of ProjectGroup objects containing Code, Dataset, and/or Model.
    """
    project_groups: List[ProjectGroup] = []
    path = Path(filepath)

    # Read the project file line by line. Each line is comma-separated:
    # code_link,dataset_link,model_link. Missing entries are treated as empty.
    with path.open("r", encoding="ASCII") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:  # skip empty lines
                continue
            # Split into up to three parts and pad with empty strings
            parts = [p.strip() for p in line.split(",")]
            while len(parts) < 3:
                parts.append("")
            code_link, dataset_link, model_link = parts

            code: Optional[Code] = None
            dataset: Optional[Dataset] = None
            model: Optional[Model] = None
            
            # Code link: store as is if non-empty
            if code_link:
                code = Code(code_link)
            
            # Dataset link: parse via helper, on failure leave as None
            if dataset_link:
                try:
                    dataset_repo = parse_dataset_url(dataset_link)
                    dataset = Dataset(dataset_link, namespace="",
                    repo=dataset_repo, rev="")
                except Exception:
                    dataset = None

            # Model link: parse Hugging Face URL, on failure leave as None
            if model_link:
                try:
                    ns, rp, rev = parse_huggingface_url(model_link)
                    model = Model(model_link, ns, rp, rev)
                except Exception:
                    model = None
                
            project_groups.append(ProjectGroup(code=code, dataset=dataset,
            model=model))

    return project_groups

parse_hf_dataset_url_repo = parse_dataset_url

def main():
    # Point to your test file
    filepath = Path("tests/test.txt")

    # Parse file
    groups = parse_project_file(str(filepath))

    # Print results
    print("Parsed project groups:\n")
    for i, group in enumerate(groups, start=1):
        print(f"Group {i}: {group}")


if __name__ == "__main__":
    url = "https://huggingface.co/openai-community/gpt2"
    ns, rp, rev = parse_huggingface_url(url)
    print(f"Namespace: {ns}, Repo: {rp}")
    # Output: Namespace: openai-community, Repo: gpt2