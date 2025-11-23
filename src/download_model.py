"""
Model Download Module for ECE30861 Team 6 Registry

Handles downloading models from HuggingFace Hub.

Integrates with the registry database to verify model access and retrieve metadata.
"""

import os
import io
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO, List
from urllib.parse import urlparse
import logging

# HuggingFace support
try:
    from huggingface_hub import snapshot_download, hf_hub_download, model_info, list_repo_files
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    snapshot_download = None
    hf_hub_download = None
    model_info = None
    list_repo_files = None

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Model downloader for HuggingFace Hub models.
    
    Flow:
    1. User requests download (model_id or artifact_id)
    2. Verify access permissions (optional, based on auth)
    3. Search database/registry for model metadata
    4. Retrieve files from HuggingFace Hub
    5. Stream data to user as HTTP download response
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        artifacts_db: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the ModelDownloader.
        
        Args:
            cache_dir: Local cache directory for downloads
            artifacts_db: Reference to ARTIFACTS registry (in-memory database)
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "model_registry"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Reference to the artifacts database (can be passed from API)
        self.artifacts_db = artifacts_db or {}
        
        if not HAS_HF_HUB:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface-hub")
        else:
            logger.info("HuggingFace Hub client initialized")
    
    def verify_model_access(
        self, 
        model_id: str, 
        user_token: Optional[str] = None
    ) -> bool:
        """
        Verify if user has access to download the model.
        
        Args:
            model_id: Model identifier (artifact ID or model name)
            user_token: Optional authentication token
            
        Returns:
            True if access granted, False otherwise
        """
        # For now, basic check: model exists in registry
        # Can be extended with user permissions, access control lists, etc.
        
        if model_id in self.artifacts_db:
            # Model exists in registry
            return True
        
        # If not in registry, it might be a HuggingFace model (external)
        # Allow access to external models for now
        return True
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata from database/registry.
        
        Args:
            model_id: Model identifier (artifact ID)
            
        Returns:
            Model metadata dictionary or None if not found
        """
        metadata = self.artifacts_db.get(model_id)
        
        # If not in local registry, try to get from HuggingFace
        if not metadata and HAS_HF_HUB:
            try:
                # Extract HuggingFace model ID from URL or use as-is
                hf_model_id = self._extract_hf_model_id(model_id)
                info = model_info(hf_model_id)
                metadata = {
                    'hf_model_id': hf_model_id,
                    'name': hf_model_id.split('/')[-1],
                    'type': 'model',
                    'url': f"https://huggingface.co/{hf_model_id}"
                }
            except Exception as e:
                logger.error(f"Error getting HuggingFace metadata: {e}")
        
        return metadata
    
    def _extract_hf_model_id(self, model_identifier: str) -> str:
        """
        Extract HuggingFace model ID from various formats.
        
        Args:
            model_identifier: URL or model ID
            
        Returns:
            HuggingFace model ID (e.g., 'openai-community/gpt2')
        """
        # If it's a URL, parse it
        if model_identifier.startswith('http'):
            parsed = urlparse(model_identifier)
            parts = parsed.path.strip('/').split('/')
            # Handle formats like: /namespace/repo or /namespace/repo/tree/main
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            elif len(parts) == 1:
                return parts[0]
        
        # Otherwise, assume it's already a model ID
        return model_identifier
    
    def list_model_files(self, model_id: str, revision: str = "main") -> List[str]:
        """
        List all files for a HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID
            revision: Model revision/branch
            
        Returns:
            List of file paths in the model repository
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return []
        
        try:
            hf_model_id = self._extract_hf_model_id(model_id)
            files = list_repo_files(hf_model_id, revision=revision)
            return files
        except Exception as e:
            logger.error(f"Error listing model files: {e}")
            return []
    
    def download_model_package(
        self, 
        model_id: str,
        revision: str = "main",
        output_path: Optional[str] = None,
        format: str = "directory"
    ) -> Optional[str]:
        """
        Download complete model package from HuggingFace.
        
        Args:
            model_id: HuggingFace model identifier
            revision: Model revision/branch
            output_path: Optional output path for package
            format: Package format ('zip' or 'directory')
            
        Returns:
            Path to downloaded package or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            hf_model_id = self._extract_hf_model_id(model_id)
            
            # Download to directory first
            if output_path is None:
                output_dir = os.path.join(
                    self.cache_dir, 
                    "models",
                    hf_model_id.replace("/", "_")
                )
            else:
                output_dir = output_path if format == "directory" else tempfile.mkdtemp()
            
            logger.info(f"Downloading HuggingFace model: {hf_model_id}")
            model_path = snapshot_download(
                repo_id=hf_model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            
            # If ZIP format requested, create archive
            if format == "zip":
                zip_path = output_path or os.path.join(
                    self.cache_dir,
                    f"{hf_model_id.replace('/', '_')}.zip"
                )
                return self._create_zip_from_directory(model_path, zip_path)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model package: {e}")
            return None
    
    def download_specific_file(
        self,
        model_id: str,
        filename: str,
        revision: str = "main",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Download a specific file from a HuggingFace model.
        
        Args:
            model_id: HuggingFace model identifier
            filename: Name of file to download
            revision: Model revision/branch
            output_path: Optional output file path
            
        Returns:
            Path to downloaded file or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            hf_model_id = self._extract_hf_model_id(model_id)
            
            logger.info(f"Downloading file '{filename}' from {hf_model_id}")
            file_path = hf_hub_download(
                repo_id=hf_model_id,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir,
                local_dir=output_path if output_path else None
            )
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading file {filename}: {e}")
            return None
    
    def stream_model_file(
        self,
        model_id: str,
        filename: str,
        revision: str = "main"
    ) -> Optional[BinaryIO]:
        """
        Stream a model file for HTTP download response.
        
        Args:
            model_id: HuggingFace model identifier
            filename: Name of file to stream
            revision: Model revision/branch
            
        Returns:
            File-like binary stream or None on error
        """
        # Download file first, then open for streaming
        file_path = self.download_specific_file(model_id, filename, revision)
        
        if file_path and os.path.exists(file_path):
            try:
                return open(file_path, 'rb')
            except Exception as e:
                logger.error(f"Error opening file for streaming: {e}")
                return None
        
        return None
    
    def _create_zip_from_directory(
        self,
        source_dir: str,
        output_path: str
    ) -> Optional[str]:
        """
        Create a ZIP package from a directory.
        
        Args:
            source_dir: Source directory to zip
            output_path: Output ZIP file path
            
        Returns:
            Path to created ZIP file or None on error
        """
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname=arcname)
            
            logger.info(f"Created ZIP package: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating ZIP package: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a HuggingFace model without downloading.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Dictionary of model information or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            hf_model_id = self._extract_hf_model_id(model_id)
            info = model_info(hf_model_id)
            
            return {
                "id": info.id,
                "name": hf_model_id.split('/')[-1],
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "model_size": getattr(info, 'model_size', None),
                "safetensors": getattr(info, 'safetensors', None),
                "author": info.author if hasattr(info, 'author') else None,
                "last_modified": str(info.lastModified) if hasattr(info, 'lastModified') else None
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None


class HuggingFaceDownloader:
    """
    Legacy HuggingFace downloader for external models.
    Kept for backward compatibility.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_model(self, model_id: str, revision: str = "main") -> Optional[str]:
        """
        Download a complete HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID
            revision: Model revision/branch
            
        Returns:
            Path to downloaded model directory or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_dir=os.path.join(
                    self.cache_dir, "models", model_id.replace("/", "_")
                ),
                local_dir_use_symlinks=False
            )
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            return None
    
    def download_specific_file(
        self, 
        model_id: str, 
        filename: str, 
        revision: str = "main"
    ) -> Optional[str]:
        """
        Download a specific file from a HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID
            filename: File to download
            revision: Model revision/branch
            
        Returns:
            Path to downloaded file or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir
            )
            return file_path
        except Exception as e:
            logger.error(f"Error downloading file {filename} from {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model without downloading it.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Dictionary of model information or None on error
        """
        if not HAS_HF_HUB:
            logger.error("huggingface_hub not installed")
            return None
        
        try:
            info = model_info(model_id)
            return {
                "id": info.id,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "model_size": getattr(info, 'model_size', None),
                "safetensors": getattr(info, 'safetensors', None)
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None