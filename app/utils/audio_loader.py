# app/utils/audio_loader.py
# Ses dosyası yükleme, doğrulama ve ön işleme sistemi

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
import tempfile
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from app.config import settings

logger = logging.getLogger(__name__)

class AudioLoadError(Exception):
    """Ses dosyası yükleme hatası"""
    pass

class AudioValidationError(Exception):
    """Ses dosyası doğrulama hatası"""
    pass

class AudioLoader:
    """
    Ses dosyası yükleme ve ön işleme sınıfı
    Farklı formatları destekler ve audio preprocessing yapar
    """
    
    def __init__(self):
        """AudioLoader'ı başlat"""
        self.supported_formats = settings.supported_audio_formats
        self.target_sample_rate = settings.default_sample_rate
        self.max_duration = settings.max_audio_duration_seconds
        self.max_file_size = settings.max_file_size_bytes
        
        logger.info(f"AudioLoader initialized - Formats: {self.supported_formats}")
        logger.info(f"Target sample rate: {self.target_sample_rate}Hz")
        logger.info(f"Max duration: {self.max_duration}s, Max size: {self.max_file_size//1024//1024}MB")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ses dosyasını doğrula
        
        Args:
            file_path: Ses dosyası yolu
            
        Returns:
            Dict: Dosya bilgileri
            
        Raises:
            AudioValidationError: Doğrulama hatası
        """
        try:
            file_path = Path(file_path)
            
            # Dosya varlığı kontrolü
            if not file_path.exists():
                raise AudioValidationError(f"File not found: {file_path}")
            
            # Dosya boyutu kontrolü
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise AudioValidationError(
                    f"File too large: {file_size//1024//1024}MB > {self.max_file_size//1024//1024}MB"
                )
            
            # Format kontrolü
            file_extension = file_path.suffix.lower().lstrip('.')
            if file_extension not in self.supported_formats:
                raise AudioValidationError(
                    f"Unsupported format: {file_extension}. Supported: {self.supported_formats}"
                )
            
            # Temel audio bilgileri
            try:
                info = sf.info(str(file_path))
                duration = info.frames / info.samplerate
                
                # Süre kontrolü
                if duration > self.max_duration:
                    raise AudioValidationError(
                        f"Audio too long: {duration:.1f}s > {self.max_duration}s"
                    )
                
                validation_result = {
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "format": file_extension,
                    "duration": duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "valid": True
                }
                
                logger.info(f"File validated: {file_path.name} - {duration:.1f}s, {info.samplerate}Hz")
                return validation_result
                
            except Exception as e:
                # soundfile başarısız olursa pydub ile dene
                try:
                    audio_segment = AudioSegment.from_file(str(file_path))
                    duration = len(audio_segment) / 1000.0  # milisaniyeden saniyeye
                    
                    if duration > self.max_duration:
                        raise AudioValidationError(
                            f"Audio too long: {duration:.1f}s > {self.max_duration}s"
                        )
                    
                    validation_result = {
                        "file_path": str(file_path),
                        "file_size": file_size,
                        "format": file_extension,
                        "duration": duration,
                        "sample_rate": audio_segment.frame_rate,
                        "channels": audio_segment.channels,
                        "frames": int(duration * audio_segment.frame_rate),
                        "valid": True,
                        "loaded_with": "pydub"
                    }
                    
                    logger.info(f"File validated with pydub: {file_path.name}")
                    return validation_result
                    
                except Exception as inner_e:
                    raise AudioValidationError(f"Cannot read audio file: {e}, {inner_e}")
        
        except AudioValidationError:
            raise
        except Exception as e:
            raise AudioValidationError(f"Validation error: {e}")
    
    def load_audio(
        self, 
        file_path: str, 
        target_sr: Optional[int] = None,
        mono: bool = True,
        normalize: bool = None
    ) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükle ve işle
        
        Args:
            file_path: Ses dosyası yolu
            target_sr: Hedef sample rate (None = default)
            mono: Mono'ya dönüştür mü
            normalize: Normalize et mi (None = config'den al)
            
        Returns:
            Tuple[np.ndarray, int]: (audio_data, sample_rate)
            
        Raises:
            AudioLoadError: Yükleme hatası
        """
        try:
            # Dosyayı doğrula
            validation_info = self.validate_file(file_path)
            
            # Parametreleri hazırla
            target_sample_rate = target_sr or self.target_sample_rate
            should_normalize = normalize if normalize is not None else settings.normalize_audio
            
            logger.info(f"Loading audio: {Path(file_path).name}")
            
            # Audio'yu yükle
            try:
                # Önce librosa ile dene (daha güvenilir preprocessing)
                audio_data, sample_rate = librosa.load(
                    file_path,
                    sr=target_sample_rate,
                    mono=mono,
                    dtype=np.float32
                )
                
                logger.debug(f"Loaded with librosa: {audio_data.shape}, {sample_rate}Hz")
                
            except Exception as e:
                logger.warning(f"Librosa failed, trying soundfile: {e}")
                
                # soundfile ile dene
                try:
                    audio_data, sample_rate = sf.read(file_path, dtype=np.float32)
                    
                    # Mono dönüşüm
                    if mono and len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # Sample rate dönüşümü
                    if sample_rate != target_sample_rate:
                        audio_data = librosa.resample(
                            audio_data, 
                            orig_sr=sample_rate, 
                            target_sr=target_sample_rate
                        )
                        sample_rate = target_sample_rate
                        
                except Exception as inner_e:
                    logger.warning(f"Soundfile failed, trying pydub: {inner_e}")
                    
                    # Son çare: pydub
                    audio_data, sample_rate = self._load_with_pydub(
                        file_path, target_sample_rate, mono
                    )
            
            # Ön işleme
            audio_data = self._preprocess_audio(audio_data, should_normalize)
            
            # Son kontroller
            if len(audio_data) == 0:
                raise AudioLoadError("Loaded audio is empty")
            
            duration = len(audio_data) / sample_rate
            logger.info(f"Audio loaded successfully: {duration:.2f}s, {sample_rate}Hz, shape: {audio_data.shape}")
            
            return audio_data, sample_rate
            
        except AudioValidationError:
            raise AudioLoadError(f"Validation failed for {file_path}")
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise AudioLoadError(f"Audio loading failed: {e}")
    
    def _load_with_pydub(
        self, 
        file_path: str, 
        target_sample_rate: int, 
        mono: bool
    ) -> Tuple[np.ndarray, int]:
        """Pydub ile ses dosyası yükleme (son çare)"""
        try:
            audio_segment = AudioSegment.from_file(file_path)
            
            # Mono dönüşüm
            if mono:
                audio_segment = audio_segment.set_channels(1)
            
            # Sample rate dönüşüm
            if audio_segment.frame_rate != target_sample_rate:
                audio_segment = audio_segment.set_frame_rate(target_sample_rate)
            
            # NumPy array'e dönüştür
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize et (16-bit'den float32'ye)
            if audio_segment.sample_width == 2:  # 16-bit
                audio_data = audio_data / 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit
                audio_data = audio_data / 2147483648.0
            
            # Stereo ise reshape et
            if not mono and audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
            
            logger.info(f"Loaded with pydub: {audio_segment.frame_rate}Hz")
            return audio_data, target_sample_rate
            
        except Exception as e:
            raise AudioLoadError(f"Pydub loading failed: {e}")
    
    def _preprocess_audio(self, audio_data: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Audio ön işleme
        
        Args:
            audio_data: Ham audio verisi
            normalize: Normalize edilsin mi
            
        Returns:
            np.ndarray: İşlenmiş audio verisi
        """
        processed_audio = audio_data.copy()
        
        try:
            # Normalize
            if normalize:
                max_val = np.abs(processed_audio).max()
                if max_val > 0:
                    processed_audio = processed_audio / max_val
                    logger.debug("Audio normalized")
            
            # Silence removal (eğer aktifse)
            if settings.remove_silence:
                processed_audio = self._remove_silence(processed_audio)
            
            # Noise reduction (eğer aktifse)
            if settings.noise_reduction:
                processed_audio = self._reduce_noise(processed_audio)
            
            # NaN ve inf değerleri temizle
            processed_audio = np.nan_to_num(processed_audio)
            
            # Değer aralığını kontrol et
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            
            logger.debug("Audio preprocessing completed")
            return processed_audio
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, returning original audio")
            return audio_data
    
    def _remove_silence(self, audio_data: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Sessizlikleri kaldır (basit implementasyon)"""
        try:
            # Librosa ile silence detection
            intervals = librosa.effects.split(audio_data, top_db=top_db)
            
            if len(intervals) == 0:
                return audio_data
            
            # Sessiz olmayan kısımları birleştir
            non_silent_audio = []
            for start, end in intervals:
                non_silent_audio.append(audio_data[start:end])
            
            if non_silent_audio:
                result = np.concatenate(non_silent_audio)
                logger.debug(f"Silence removed: {len(audio_data)} -> {len(result)} samples")
                return result
            else:
                return audio_data
                
        except Exception as e:
            logger.warning(f"Silence removal failed: {e}")
            return audio_data
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Basit noise reduction (spektral subtraction benzeri)"""
        try:
            # Bu basit bir implementasyon - üretimde daha gelişmiş yöntemler kullanılabilir
            
            # RMS tabanlı gürültü tahmini (ilk %10 sessizlik varsayımı)
            noise_sample_size = int(len(audio_data) * 0.1)
            noise_sample = audio_data[:noise_sample_size]
            noise_level = np.sqrt(np.mean(noise_sample**2))
            
            # Eşik altındaki değerleri azalt
            threshold = noise_level * 2
            mask = np.abs(audio_data) > threshold
            
            processed = audio_data.copy()
            processed[~mask] *= 0.1  # Sessiz kısımları azalt
            
            logger.debug(f"Noise reduction applied, threshold: {threshold:.4f}")
            return processed
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def save_processed_audio(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_path: str,
        format: str = "wav"
    ) -> str:
        """
        İşlenmiş audio'yu kaydet
        
        Args:
            audio_data: Audio verisi
            sample_rate: Sample rate
            output_path: Çıktı dosya yolu
            format: Dosya formatı
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format'a göre kaydet
            if format.lower() == "wav":
                sf.write(str(output_path), audio_data, sample_rate)
            else:
                # Pydub ile diğer formatlar
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1]
                )
                audio_segment.export(str(output_path), format=format)
            
            logger.info(f"Audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise AudioLoadError(f"Save failed: {e}")
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Ses dosyası hakkında detaylı bilgi al
        
        Args:
            file_path: Ses dosyası yolu
            
        Returns:
            Dict: Detaylı audio bilgileri
        """
        try:
            validation_info = self.validate_file(file_path)
            
            # Ek bilgiler
            file_path = Path(file_path)
            
            # Audio özelliklerini analiz et
            audio_data, sample_rate = self.load_audio(file_path, normalize=False)
            
            # Temel istatistikler
            rms_energy = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.abs(audio_data).max()
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            # Spektral özellikler
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0])
            
            info = {
                **validation_info,
                "audio_stats": {
                    "rms_energy": float(rms_energy),
                    "max_amplitude": float(max_amplitude),
                    "zero_crossing_rate": float(zero_crossing_rate),
                    "spectral_centroid": float(spectral_centroid),
                    "spectral_bandwidth": float(spectral_bandwidth),
                    "dynamic_range": float(max_amplitude / (rms_energy + 1e-8))
                }
            }
            
            logger.info(f"Audio info extracted for: {file_path.name}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            raise AudioLoadError(f"Info extraction failed: {e}")
    
    def batch_validate(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Birden fazla dosyayı toplu doğrula
        
        Args:
            file_paths: Dosya yolları listesi
            
        Returns:
            Dict: Dosya adı -> validation sonucu mapping
        """
        results = {}
        
        for file_path in file_paths:
            try:
                file_name = Path(file_path).name
                results[file_name] = self.validate_file(file_path)
                results[file_name]["status"] = "valid"
                
            except AudioValidationError as e:
                file_name = Path(file_path).name if file_path else "unknown"
                results[file_name] = {
                    "valid": False,
                    "status": "invalid",
                    "error": str(e)
                }
            except Exception as e:
                file_name = Path(file_path).name if file_path else "unknown"
                results[file_name] = {
                    "valid": False,
                    "status": "error",
                    "error": f"Unexpected error: {e}"
                }
        
        logger.info(f"Batch validation completed: {len(results)} files")
        return results

# ================================
# UTILITY FUNCTIONS
# ================================

def create_audio_loader() -> AudioLoader:
    """AudioLoader singleton factory"""
    return AudioLoader()

# Global instance
audio_loader = create_audio_loader()

# ================================
# CONVENIENCE FUNCTIONS
# ================================

def load_audio_file(
    file_path: str, 
    target_sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = None
) -> Tuple[np.ndarray, int]:
    """
    Convenience function - ses dosyası yükleme
    
    Args:
        file_path: Ses dosyası yolu
        target_sr: Hedef sample rate
        mono: Mono dönüşüm
        normalize: Normalize
        
    Returns:
        Tuple[np.ndarray, int]: (audio_data, sample_rate)
    """
    return audio_loader.load_audio(file_path, target_sr, mono, normalize)

def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function - ses dosyası doğrulama
    
    Args:
        file_path: Ses dosyası yolu
        
    Returns:
        Dict: Doğrulama sonucu
    """
    return audio_loader.validate_file(file_path)

def get_audio_file_info(file_path: str) -> Dict[str, Any]:
    """
    Convenience function - ses dosyası bilgisi
    
    Args:
        file_path: Ses dosyası yolu
        
    Returns:
        Dict: Detaylı ses bilgisi
    """
    return audio_loader.get_audio_info(file_path)

# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    # Test ve örnek kullanım
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        try:
            print(f"Testing AudioLoader with: {test_file}")
            
            # Validate
            validation = validate_audio_file(test_file)
            print(f"Validation: {validation}")
            
            # Load
            audio_data, sr = load_audio_file(test_file)
            print(f"Loaded: {audio_data.shape}, {sr}Hz")
            
            # Info
            info = get_audio_file_info(test_file)
            print(f"Info: {info['audio_stats']}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python audio_loader.py <audio_file>")
        print("AudioLoader module loaded successfully!")