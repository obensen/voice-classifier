"""
Audio loading and preprocessing utilities for voice classification.

This module handles audio file loading, validation, preprocessing and metadata extraction
with support for various audio formats and robust error handling.
"""

import os
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import mutagen
from mutagen.id3 import ID3NoHeaderError

from app.config import get_audio_config, Settings
from app.models.schemas import AudioData, AudioMetadata, ProcessingError
from app.utils.cache import cache_manager

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

class AudioLoader:
    """
    Audio file loader with comprehensive validation, preprocessing and metadata extraction.
    
    Features:
    - Multi-format support (WAV, MP3, FLAC, OGG, M4A, AAC)
    - File validation (size, duration, format)
    - Audio preprocessing (normalization, resampling, noise reduction)
    - Metadata extraction
    - Async processing with thread pool
    - Memory efficient processing
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize AudioLoader with configuration."""
        self.config = config or get_audio_config()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Audio processing settings
        self.max_file_size = getattr(self.config, 'max_file_size_mb', 50) * 1024 * 1024
        self.max_duration = getattr(self.config, 'max_duration_seconds', 300)
        self.target_sample_rate = getattr(self.config, 'target_sample_rate', 16000)
        self.normalize_audio = getattr(self.config, 'normalize_audio', True)
        self.apply_noise_reduction = getattr(self.config, 'apply_noise_reduction', True)
        
        logger.info(f"AudioLoader initialized with target_sr={self.target_sample_rate}")
    
    async def load_audio(self, file_path: str) -> AudioData:
        """
        Load and process audio file asynchronously.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioData: Processed audio data with metadata
            
        Raises:
            AudioProcessingError: If file cannot be processed
        """
        try:
            # Check cache first
            file_hash = await self._get_file_hash(file_path)
            cache_key = f"audio_data:{file_hash}"
            
            cached_data = cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Audio data loaded from cache for {file_path}")
                return cached_data
            
            # Validate file
            await self._validate_file(file_path)
            
            # Load audio in thread pool
            audio_data, sample_rate = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._load_audio_sync, file_path
            )
            
            # Preprocess audio
            processed_audio = await self._preprocess_audio(audio_data, sample_rate)
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path, processed_audio, sample_rate)
            
            # Create AudioData object
            result = AudioData(
                audio_data=processed_audio,
                sample_rate=self.target_sample_rate,
                metadata=metadata,
                file_hash=file_hash
            )
            
            # Cache result
            cache_manager.set(cache_key, result, ttl=300)  # 5 minutes
            
            logger.info(f"Audio loaded successfully: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {str(e)}")
            raise AudioProcessingError(f"Audio loading failed: {str(e)}") from e
    
    async def validate_format(self, file_path: str) -> bool:
        """
        Validate if file format is supported.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if format is supported
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            return file_extension in self.SUPPORTED_FORMATS
        except Exception:
            return False
    
    async def _validate_file(self, file_path: str) -> None:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to audio file
            
        Raises:
            AudioProcessingError: If validation fails
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise AudioProcessingError(
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit "
                f"({self.max_file_size / 1024 / 1024}MB)"
            )
        
        # Check format
        if not await self.validate_format(file_path):
            raise AudioProcessingError(
                f"Unsupported format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Check if file is readable and not corrupted
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._validate_audio_file, file_path
            )
        except Exception as e:
            raise AudioProcessingError(f"File validation failed: {str(e)}") from e
    
    def _validate_audio_file(self, file_path: str) -> None:
        """Validate audio file by attempting to read basic info."""
        try:
            # Try with librosa first
            duration = librosa.get_duration(path=file_path)
            if duration > self.max_duration:
                raise AudioProcessingError(
                    f"Audio duration ({duration:.1f}s) exceeds limit ({self.max_duration}s)"
                )
        except Exception:
            # Try with pydub as fallback
            try:
                audio = AudioSegment.from_file(file_path)
                duration = len(audio) / 1000.0
                if duration > self.max_duration:
                    raise AudioProcessingError(
                        f"Audio duration ({duration:.1f}s) exceeds limit ({self.max_duration}s)"
                    )
            except CouldntDecodeError as e:
                raise AudioProcessingError(f"Corrupted or invalid audio file") from e
    
    def _load_audio_sync(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Synchronous audio loading using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load with librosa (handles most formats well)
            audio_data, sample_rate = librosa.load(
                file_path, 
                sr=None,  # Keep original sample rate initially
                mono=True,  # Convert to mono
                dtype=np.float32
            )
            
            logger.debug(f"Loaded audio: shape={audio_data.shape}, sr={sample_rate}")
            return audio_data, sample_rate
            
        except Exception as e:
            # Fallback to pydub for problematic formats
            logger.warning(f"Librosa failed, trying pydub: {str(e)}")
            return self._load_with_pydub(file_path)
    
    def _load_with_pydub(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Fallback audio loading using pydub.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load with pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert to numpy array
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            if audio.sample_width == 2:  # 16-bit
                audio_data = audio_data / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                audio_data = audio_data / 2147483648.0
            elif audio.sample_width == 1:  # 8-bit
                audio_data = (audio_data - 128) / 128.0
            
            return audio_data, audio.frame_rate
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio with pydub: {str(e)}") from e
    
    async def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio data asynchronously.
        
        Args:
            audio_data: Raw audio data
            sample_rate: Original sample rate
            
        Returns:
            np.ndarray: Processed audio data
        """
        try:
            # Run preprocessing in thread pool
            processed = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._preprocess_sync, audio_data, sample_rate
            )
            
            logger.debug(f"Audio preprocessed: shape={processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise AudioProcessingError(f"Preprocessing failed: {str(e)}") from e
    
    def _preprocess_sync(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Synchronous audio preprocessing.
        
        Args:
            audio_data: Raw audio data
            sample_rate: Original sample rate
            
        Returns:
            np.ndarray: Processed audio data
        """
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.target_sample_rate
            )
        
        # Normalize audio
        if self.normalize_audio:
            # RMS normalization
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                audio_data = audio_data / rms * 0.1  # Target RMS of 0.1
            
            # Peak normalization (ensure no clipping)
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
        
        # Apply noise reduction (simple spectral gating)
        if self.apply_noise_reduction:
            audio_data = self._reduce_noise(audio_data)
        
        # Trim silence
        audio_data, _ = librosa.effects.trim(
            audio_data, 
            top_db=20,  # Trim silence below -20dB
            frame_length=2048,
            hop_length=512
        )
        
        return audio_data
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Simple noise reduction using spectral gating.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            np.ndarray: Noise-reduced audio data
        """
        try:
            # Compute spectral features
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Estimate noise floor (bottom 20% of magnitudes)
            noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
            
            # Apply spectral gating (suppress frequencies below noise floor * 1.5)
            gate_threshold = noise_floor * 1.5
            mask = magnitude > gate_threshold
            
            # Apply mask with smooth transitions
            gated_stft = stft * mask
            
            # Reconstruct audio
            return librosa.istft(gated_stft, hop_length=512)
            
        except Exception:
            # If noise reduction fails, return original audio
            logger.warning("Noise reduction failed, using original audio")
            return audio_data
    
    async def _extract_metadata(
        self, 
        file_path: str, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> AudioMetadata:
        """
        Extract audio metadata asynchronously.
        
        Args:
            file_path: Path to audio file
            audio_data: Processed audio data
            sample_rate: Audio sample rate
            
        Returns:
            AudioMetadata: Extracted metadata
        """
        try:
            # Run metadata extraction in thread pool
            metadata = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._extract_metadata_sync, file_path, audio_data, sample_rate
            )
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {str(e)}")
            # Return basic metadata if extraction fails
            return AudioMetadata(
                duration=len(audio_data) / sample_rate,
                sample_rate=sample_rate,
                channels=1,
                format=Path(file_path).suffix.lower(),
                file_size=Path(file_path).stat().st_size
            )
    
    def _extract_metadata_sync(
        self, 
        file_path: str, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> AudioMetadata:
        """
        Synchronous metadata extraction.
        
        Args:
            file_path: Path to audio file
            audio_data: Processed audio data
            sample_rate: Audio sample rate
            
        Returns:
            AudioMetadata: Extracted metadata
        """
        path = Path(file_path)
        
        # Basic metadata from processed audio
        duration = len(audio_data) / sample_rate
        file_size = path.stat().st_size
        
        # Additional metadata from file
        extra_metadata = {}
        
        try:
            # Try to extract metadata using mutagen
            audio_file = mutagen.File(file_path)
            if audio_file:
                # Extract common tags
                if hasattr(audio_file, 'info'):
                    info = audio_file.info
                    extra_metadata.update({
                        'bitrate': getattr(info, 'bitrate', None),
                        'original_duration': getattr(info, 'length', duration),
                        'original_channels': getattr(info, 'channels', 1)
                    })
                
                # Extract tags
                tags = dict(audio_file.tags) if audio_file.tags else {}
                extra_metadata['tags'] = tags
                
        except (ID3NoHeaderError, Exception) as e:
            logger.debug(f"Could not extract detailed metadata: {str(e)}")
        
        # Compute audio characteristics
        try:
            # RMS energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )[0]
            
            extra_metadata.update({
                'rms_energy': float(rms_energy),
                'zcr_mean': float(np.mean(zcr)),
                'spectral_centroid_mean': float(np.mean(spectral_centroid))
            })
            
        except Exception as e:
            logger.debug(f"Could not compute audio characteristics: {str(e)}")
        
        return AudioMetadata(
            duration=duration,
            sample_rate=sample_rate,
            channels=1,  # Always mono after processing
            format=path.suffix.lower(),
            file_size=file_size,
            **extra_metadata
        )
    
    async def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate file hash for caching.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            str: SHA256 hash of file
        """
        try:
            def _calculate_hash():
                hash_sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
            
            file_hash = await asyncio.get_event_loop().run_in_executor(
                self.executor, _calculate_hash
            )
            
            return file_hash
            
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {str(e)}")
            # Fallback to file path + size + mtime
            path = Path(file_path)
            stat = path.stat()
            fallback_data = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.sha256(fallback_data.encode()).hexdigest()
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Global instance for reuse
_audio_loader_instance: Optional[AudioLoader] = None

def get_audio_loader(config: Optional[Settings] = None) -> AudioLoader:
    """
    Get AudioLoader singleton instance.
    
    Args:
        config: Optional configuration override
        
    Returns:
        AudioLoader: Singleton instance
    """
    global _audio_loader_instance
    
    if _audio_loader_instance is None:
        _audio_loader_instance = AudioLoader(config)
    
    return _audio_loader_instance