"""
Utilities for directory management and file operations.
"""
import os
import shutil
import json
from pathlib import Path
from typing import Union, Dict, Any, List

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path: Path to the directory
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path

def clean_dir(path: Union[str, Path]) -> Path:
    """
    Clean a directory by removing all its contents.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path: Path to the cleaned directory
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        file_path: Path to the output file
    """
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Dict[str, Any]: Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        return json.load(f)

def find_files(directory: Union[str, Path], extensions: List[str]) -> List[Path]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to look for
        
    Returns:
        List[Path]: List of found files
    """
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        files.extend(directory.glob(f'*{ext}'))
        
    return files
