"""Speech separation module using SepFormer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

from audio_models.external.repo_manager import ensure_repo

class SpeechSeparator:
    """Speech separator using SepFormer from SpeechBrain."""
    
    def __init__(self, device: str = "cpu", chunk_duration: float = 30.0):
        """
        Initialize the speech separation model.
        
        Args:
            device: Device to run the model on ("cpu" or "cuda")
            chunk_duration: Duration of each chunk in seconds for memory-efficient processing
        """
        self.device = device
        self.model = None
        self.sample_rate = 8000  # SepFormer default
        self.chunk_duration = chunk_duration
        
    def _import_sepformer(self):
        """Ensure SpeechBrain is importable from the managed clone."""

        try:
            from speechbrain.inference import SepformerSeparation  # type: ignore
            return SepformerSeparation
        except ImportError:
            repo_path = ensure_repo("speechbrain")
            repo_root = Path(repo_path)
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            try:
                from speechbrain.inference import SepformerSeparation  # type: ignore
                return SepformerSeparation
            except ImportError as exc:  # pragma: no cover - defensive path
                raise ImportError(
                    "Unable to import SpeechBrain even after cloning the repository. "
                    "Run `pip install -e external/audio/speechbrain` inside the audio_models environment."
                ) from exc

    def _load_model(self):
        """Load the SepFormer model from SpeechBrain."""
        SepformerSeparation = self._import_sepformer()
        try:
            print("Loading SepFormer model...")
            self.model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-libri3mix",
                run_opts={"device": self.device},
            )
            print("SepFormer model loaded successfully")
        except Exception as e:  # pragma: no cover - runtime guard
            print(f"Error loading SepFormer model: {e}")
            raise
    
    def separate(self, audio_path: str, output_dir: str = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Separate speakers from an audio file using memory-efficient chunked processing.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save separated audio files (if None, files are not saved)
            
        Returns:
            Tuple[List[np.ndarray], List[str]]: List of separated audio waveforms and paths to saved files
        """
        print(f"Starting speech separation for: {audio_path}")
        
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        try:
            print("Loading and preprocessing audio...")
            # Load original audio for both separation and feature extraction
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded: {waveform.shape}, sample rate: {sample_rate}")
            duration = waveform.shape[1]/sample_rate
            print(f"Duration: {duration:.1f} seconds")
            
            # Store original audio for feature extraction (before any processing)
            original_waveform = waveform.clone()
            original_sample_rate = sample_rate
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print("Converted to mono")
            
            # Remove batch dimension if present and ensure proper shape
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            
            # Resample if needed for separation model
            if sample_rate != self.sample_rate:
                print(f"Resampling from {sample_rate} to {self.sample_rate}")
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            print("Performing speech separation...")
            
            # Check if we need chunked processing for memory efficiency
            duration_seconds = waveform.shape[-1] / self.sample_rate
            print(f"Audio duration: {duration_seconds:.1f} seconds")
            
            if duration_seconds > self.chunk_duration:
                print(f"Using chunked processing (chunks of {self.chunk_duration}s)")
                separated_sources = self._separate_chunked(waveform)
                print(f"Chunked processing returned {len(separated_sources)} separated sources")
            else:
                print("Using direct processing")
                # Perform separation using batch method for better handling
                with torch.no_grad():
                    separated_sources = self.model.separate_batch(waveform)
                print(f"Raw separated sources shape: {separated_sources.shape}")
            
            # Handle different output formats from SepFormer
            if isinstance(separated_sources, list):
                # Already processed as list from chunked processing
                processed_sources = separated_sources
            elif isinstance(separated_sources, torch.Tensor):
                if separated_sources.dim() == 3:
                    # SepFormer outputs [batch, samples, sources] - transpose to [batch, sources, samples]
                    if separated_sources.shape[2] < separated_sources.shape[1]:
                        separated_sources = separated_sources.transpose(1, 2)
                        print(f"Transposed to: {separated_sources.shape}")
                    batch_size, num_sources, num_samples = separated_sources.shape
                elif separated_sources.dim() == 2:
                    # Shape might be [samples, sources] - transpose and add batch dimension
                    separated_sources = separated_sources.transpose(0, 1).unsqueeze(0)
                    batch_size, num_sources, num_samples = separated_sources.shape
                else:
                    print(f"Unexpected tensor shape: {separated_sources.shape}")
                    # Fallback to old method
                    est_sources = self.model.separate_file(audio_path)
                    processed_sources = [source.cpu().numpy() for source in est_sources]
                    print(f"Fallback separation completed. Found {len(processed_sources)} sources")
                
                # Filter sources by energy and convert to numpy (only if we have tensor)
                if separated_sources.dim() >= 2:
                    processed_sources = []
                    max_sources = min(num_sources, 3)  # Limit to 3 speakers maximum
                    energy_threshold = 0.001
                    
                    for i in range(max_sources):
                        source_audio = separated_sources[0, i, :].cpu().numpy()
                        rms_energy = np.sqrt(np.mean(source_audio**2))
                        max_amplitude = np.max(np.abs(source_audio))
                        
                        print(f"Source {i+1} - RMS: {rms_energy:.6f}, Max amplitude: {max_amplitude:.6f}")
                        
                        if rms_energy > energy_threshold:
                            # Normalize audio to prevent clipping
                            if max_amplitude > 0:
                                source_audio = source_audio / max_amplitude * 0.95
                            processed_sources.append(source_audio)
                        else:
                            print(f"Skipped source {i + 1} (too quiet, RMS: {rms_energy:.6f})")
            else:
                print("Unknown separated_sources format")
                processed_sources = []
            
            print(f"Separation completed. Found {len(processed_sources)} valid sources")
            
            # Save separated sources if output_dir is provided
            output_paths = []
            if output_dir:
                output_dir = Path(output_dir)
                base_name = Path(audio_path).stem
                
                # Create a subfolder for this specific audio file
                audio_out_dir = output_dir / base_name
                os.makedirs(audio_out_dir, exist_ok=True)
                print(f"Saving separated sources to: {audio_out_dir}")
                
                # Save separated sources with HuggingFace SepFormer naming convention
                for i, source in enumerate(processed_sources):
                    output_path = str(audio_out_dir / f"source{i+1}hat.wav")
                    
                    # Ensure proper tensor shape for saving
                    source_tensor = torch.tensor(source)
                    
                    # Handle different tensor shapes
                    if source_tensor.dim() == 1:
                        source_tensor = source_tensor.unsqueeze(0)  # Add channel dimension
                    elif source_tensor.dim() == 2:
                        # If shape is [samples, channels], transpose to [channels, samples]
                        if source_tensor.shape[0] > source_tensor.shape[1]:
                            source_tensor = source_tensor.transpose(0, 1)
                    
                    print(f"Saving source {i+1} with shape: {source_tensor.shape}")
                    torchaudio.save(
                        output_path,
                        source_tensor,
                        self.sample_rate
                    )
                    output_paths.append(output_path)
                    print(f"Saved: {output_path}")
                    
            return processed_sources, output_paths
            
        except Exception as e:
            print(f"Error during speech separation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_feature_dict(self, audio_path: str, output_dir: str = None) -> dict:
        """
        Get a dictionary of separated speech features.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save separated audio files
            
        Returns:
            dict: Dictionary with separated sources and original audio
        """
        separated_sources, source_paths = self.separate(audio_path, output_dir)
        
        result = {}
        
        # Add separated sources 
        for i, (source, path) in enumerate(zip(separated_sources, source_paths)):
            source_key = f"source{i+1}hat"
            result[source_key] = source
            
            if output_dir:
                result[f"{source_key}_path"] = path
                
        return result
    
    def get_original_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Get the original audio for feature extraction work.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Original audio waveform and sample rate
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to numpy and return original audio without any modifications
        # This preserves the original quality and format for feature extraction
        original_audio = waveform.cpu().numpy()
        
        return original_audio, sample_rate
    
    def _separate_chunked(self, waveform: torch.Tensor) -> List[np.ndarray]:
        """
        Process long audio files in chunks to avoid memory issues.
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            List[np.ndarray]: List of separated sources
        """
        try:
            chunk_size = int(self.chunk_duration * self.sample_rate)
            num_samples = waveform.shape[-1]
            num_chunks = (num_samples + chunk_size - 1) // chunk_size
            
            print(f"Processing {num_chunks} chunks of {self.chunk_duration}s each")
            
            # Store separated sources for each chunk
            all_chunk_sources = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_samples)
                
                # Extract chunk
                chunk = waveform[:, start_idx:end_idx]
                
                print(f"Processing chunk {i+1}/{num_chunks}")
                
                # Process chunk
                chunk_sources = self._process_chunk(chunk)
                
                if chunk_sources:
                    all_chunk_sources.append(chunk_sources)
            
            if not all_chunk_sources:
                return []
            
            # Concatenate all chunks for each source
            num_sources = len(all_chunk_sources[0])
            final_sources = []
            
            for source_idx in range(num_sources):
                # Collect all chunks for this source
                source_chunks = []
                for chunk_sources in all_chunk_sources:
                    if source_idx < len(chunk_sources):
                        source_chunks.append(chunk_sources[source_idx])
                
                if source_chunks:
                    # Concatenate chunks
                    concatenated_source = np.concatenate(source_chunks, axis=0)
                    final_sources.append(concatenated_source)
            
            return final_sources
            
        except Exception as e:
            print(f"Error in chunked separation: {e}")
            return []

    def _process_chunk(self, waveform_chunk: torch.Tensor) -> List[np.ndarray]:
        """
        Process a single audio chunk for separation.
        
        Args:
            waveform_chunk: Audio chunk tensor
            
        Returns:
            List[np.ndarray]: Separated sources for this chunk
        """
        try:
            with torch.no_grad():
                # Clear GPU cache if using CUDA
                if self.device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                separated_sources = self.model.separate_batch(waveform_chunk)
                
                # Handle different output formats from SepFormer
                if isinstance(separated_sources, torch.Tensor):
                    if separated_sources.dim() == 3:
                        # SepFormer outputs [batch, samples, sources] - transpose to [batch, sources, samples]
                        if separated_sources.shape[2] < separated_sources.shape[1]:
                            separated_sources = separated_sources.transpose(1, 2)
                        batch_size, num_sources, num_samples = separated_sources.shape
                    elif separated_sources.dim() == 2:
                        # Shape might be [samples, sources] - transpose and add batch dimension
                        separated_sources = separated_sources.transpose(0, 1).unsqueeze(0)
                        batch_size, num_sources, num_samples = separated_sources.shape
                    else:
                        return []
                    
                    # Convert to numpy and filter by energy
                    processed_sources = []
                    max_sources = min(num_sources, 3)  # Limit to 3 speakers maximum
                    energy_threshold = 0.001
                    
                    for i in range(max_sources):
                        source_audio = separated_sources[0, i, :].cpu().numpy()
                        rms_energy = np.sqrt(np.mean(source_audio**2))
                        
                        if rms_energy > energy_threshold:
                            # Normalize audio to prevent clipping
                            max_amplitude = np.max(np.abs(source_audio))
                            if max_amplitude > 0:
                                source_audio = source_audio / max_amplitude * 0.95
                            processed_sources.append(source_audio)
                    
                    return processed_sources
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []
        
        return []
