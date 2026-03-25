"""
Utilities for extracting audio from video files.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Union

def extract_audio_from_video(
    video_path: Union[str, Path], 
    output_dir: Union[str, Path] = None,
    format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1
) -> str:
    """
    Extract audio from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted audio file (defaults to same as video)
        format: Audio format (default: wav)
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)
        
    Returns:
        str: Path to the extracted audio file
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    audio_filename = f"{video_path.stem}.{format}"
    audio_path = output_dir / audio_filename
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # Disable video recording
        "-ar", str(sample_rate),  # Audio sample rate
        "-ac", str(channels),  # Audio channels
        str(audio_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(audio_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error extracting audio: {e.stderr.decode()}")

def extract_audio_from_videos(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    video_extensions: List[str] = [".mp4", ".MP4", ".avi", ".mov", ".MOV", ".mkv"]
) -> List[str]:
    """
    Extract audio from all video files in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted audio files (defaults to same as videos)
        format: Audio format (default: wav)
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)
        video_extensions: List of video file extensions to process
        
    Returns:
        List[str]: Paths to the extracted audio files
    """
    video_dir = Path(video_dir)
    
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")
    
    if output_dir is None:
        output_dir = video_dir
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files in the directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
        
    audio_paths = []
    for video_file in video_files:
        try:
            audio_path = extract_audio_from_video(
                video_file, 
                output_dir, 
                format, 
                sample_rate, 
                channels
            )
            audio_paths.append(audio_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            
    return audio_paths
