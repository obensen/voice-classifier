# app/services/whisper_language_service.py
# OpenAI Whisper ile dil tespiti servisi

import whisper
import numpy as np
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
import soundfile as sf
import os

from app.models.schemas import Language
from app.config import settings

logger = logging.getLogger(__name__)

class WhisperLanguageServiceError(Exception):
    """Whisper dil tespiti hatası"""
    pass

class WhisperLanguageService:
    """
    OpenAI Whisper kullanarak dil tespiti servisi
    Audio dosyalarından konuşulan dili tespit eder
    """
    
    def __init__(self):
        """WhisperLanguageService'i başlat"""
        self.model = None
        self.model_size = settings.whisper_model_size
        self.model_path = settings.whisper_model_path
        self.device = settings.model_device
        
        # Desteklenen diller (Whisper'ın desteklediği diller)
        self.supported_languages = {
            'en': Language.ENGLISH,
            'tr': Language.TURKISH,
            'es': Language.SPANISH,
            'fr': Language.FRENCH,
            'de': Language.GERMAN,
            'it': Language.ITALIAN,
            'ru': Language.RUSSIAN,
            'zh': Language.CHINESE,
            'ja': Language.JAPANESE,
            'ar': Language.ARABIC,
        }
        
        # Language code mapping (ISO 639-1 to full names)
        self.language_names = {
            'en': 'English',
            'tr': 'Turkish',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic'
        }
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        
        # Stats
        self.detection_count = 0
        self.model_loaded = False
        
        logger.info(f"WhisperLanguageService initialized - Model: {self.model_size}, Device: {self.device}")
    
    async def initialize_model(self):
        """Whisper modelini lazy loading ile yükle"""
        if self.model_loaded:
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Model yükleme (async wrapper)
            loop = asyncio.get_event_loop()
            
            if self.model_path and os.path.exists(self.model_path):
                # Local model path varsa kullan
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: whisper.load_model(self.model_path)
                )
                logger.info(f"Loaded local Whisper model from: {self.model_path}")
            else:
                # Standard model yükle
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: whisper.load_model(self.model_size, device=self.device)
                )
                logger.info(f"Loaded Whisper model: {self.model_size} on {self.device}")
            
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise WhisperLanguageServiceError(f"Model loading failed: {e}")
    
    async def analyze(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Audio'dan dil tespiti yap
        
        Args:
            audio_data: Audio sinyal verisi
            sample_rate: Sample rate
            features: Önceden çıkarılmış features (opsiyonel)
            
        Returns:
            Dict: Dil tespiti sonuçları
        """
        try:
            # Model yükle
            await self.initialize_model()
            
            # Audio'yu geçici dosyaya kaydet (Whisper dosya bekliyor)
            temp_audio_path = await self._save_temp_audio(audio_data, sample_rate)
            
            try:
                # Whisper ile analiz yap
                result = await self._run_whisper_detection(temp_audio_path)
                
                # Sonucu işle
                processed_result = self._process_whisper_result(result)
                
                # İstatistikleri güncelle
                self.detection_count += 1
                
                logger.info(f"Language detection completed: {processed_result['language']}")
                return processed_result
                
            finally:
                # Geçici dosyayı temizle
                self._cleanup_temp_file(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise WhisperLanguageServiceError(f"Detection failed: {e}")
    
    async def detect_language_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Dosyadan direkt dil tespiti (audio_loader kullanmadan)
        
        Args:
            file_path: Audio dosya yolu
            
        Returns:
            Dict: Dil tespiti sonuçları
        """
        try:
            await self.initialize_model()
            
            # Whisper direkt dosyayı işle
            result = await self._run_whisper_detection(file_path)
            processed_result = self._process_whisper_result(result)
            
            self.detection_count += 1
            logger.info(f"Direct file language detection: {processed_result['language']}")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Direct file detection failed: {e}")
            raise WhisperLanguageServiceError(f"File detection failed: {e}")
    
    async def batch_detect_languages(
        self, 
        file_paths: List[str],
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Toplu dil tespiti
        
        Args:
            file_paths: Audio dosya yolları
            max_concurrent: Maksimum paralel işlem
            
        Returns:
            List[Dict]: Her dosya için dil tespiti sonuçları
        """
        try:
            await self.initialize_model()
            
            # Semaphore ile paralel işlem sınırla (Whisper memory-intensive)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def detect_single_file(file_path: str) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await self.detect_language_from_file(file_path)
                        return {
                            "file_name": Path(file_path).name,
                            "file_path": file_path,
                            "success": True,
                            **result
                        }
                    except Exception as e:
                        logger.warning(f"Batch detection failed for {file_path}: {e}")
                        return {
                            "file_name": Path(file_path).name,
                            "file_path": file_path,
                            "success": False,
                            "language": Language.UNKNOWN,
                            "confidence": 0.0,
                            "error": str(e)
                        }
            
            # Parallel processing
            tasks = [detect_single_file(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Exception handling
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch task exception: {result}")
                    final_results.append({
                        "success": False,
                        "error": str(result),
                        "language": Language.UNKNOWN
                    })
                else:
                    final_results.append(result)
            
            successful = sum(1 for r in final_results if r.get("success"))
            logger.info(f"Batch language detection: {successful}/{len(file_paths)} successful")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch language detection failed: {e}")
            raise WhisperLanguageServiceError(f"Batch detection failed: {e}")
    
    # ================================
    # PRIVATE METHODS
    # ================================
    
    async def _save_temp_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Audio'yu geçici dosyaya kaydet"""
        try:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.wav',
                dir=settings.temp_audio_path
            ) as temp_file:
                temp_path = temp_file.name
            
            # Audio'yu kaydet
            sf.write(temp_path, audio_data, sample_rate)
            
            logger.debug(f"Temp audio saved: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save temp audio: {e}")
            raise WhisperLanguageServiceError(f"Temp audio save failed: {e}")
    
    def _cleanup_temp_file(self, file_path: str):
        """Geçici dosyayı temizle"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Temp file cleaned: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    async def _run_whisper_detection(self, audio_path: str) -> Dict[str, Any]:
        """Whisper ile dil tespiti çalıştır"""
        try:
            loop = asyncio.get_event_loop()
            
            # Whisper transcription (language detection)
            result = await loop.run_in_executor(
                None,
                lambda: whisper.transcribe(
                    self.model, 
                    audio_path,
                    task="transcribe",
                    verbose=False
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise WhisperLanguageServiceError(f"Whisper transcription failed: {e}")
    
    def _process_whisper_result(self, whisper_result: Dict[str, Any]) -> Dict[str, Any]:
        """Whisper sonucunu işle ve formatla"""
        try:
            # Detected language
            detected_language_code = whisper_result.get('language', 'unknown')
            
            # Language mapping
            if detected_language_code in self.supported_languages:
                language_enum = self.supported_languages[detected_language_code]
            else:
                language_enum = Language.UNKNOWN
                detected_language_code = 'unknown'
            
            # Confidence estimation (Whisper doesn't provide direct confidence)
            # We estimate based on text length and consistency
            text = whisper_result.get('text', '').strip()
            segments = whisper_result.get('segments', [])
            
            confidence = self._estimate_confidence(text, segments, detected_language_code)
            
            # Alternative languages (mock - Whisper doesn't provide alternatives)
            alternatives = self._generate_alternatives(detected_language_code, confidence)
            
            # Language name
            language_name = self.language_names.get(detected_language_code, 'Unknown')
            
            result = {
                "language": language_enum,
                "confidence": confidence,
                "detected_language": detected_language_code,
                "detected_language_name": language_name,
                "transcribed_text": text,
                "alternatives": alternatives,
                "segments_count": len(segments),
                "whisper_result": {
                    "original_language": detected_language_code,
                    "text_length": len(text),
                    "segments": len(segments)
                }
            }
            
            logger.debug(f"Processed Whisper result: {detected_language_code} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process Whisper result: {e}")
            # Fallback result
            return {
                "language": Language.UNKNOWN,
                "confidence": 0.0,
                "detected_language": "unknown",
                "detected_language_name": "Unknown",
                "transcribed_text": "",
                "alternatives": [],
                "segments_count": 0,
                "whisper_result": {},
                "error": str(e)
            }
    
    def _estimate_confidence(
        self, 
        text: str, 
        segments: List[Dict], 
        language_code: str
    ) -> float:
        """
        Whisper confidence tahmini (Whisper direkt confidence vermez)
        Text kalitesi ve segment consistency'sine dayalı tahmin
        """
        try:
            base_confidence = 0.5
            
            # Text length factor
            text_length = len(text.strip())
            if text_length > 50:
                base_confidence += 0.2
            elif text_length > 20:
                base_confidence += 0.1
            elif text_length < 5:
                base_confidence -= 0.3
            
            # Segments consistency
            if segments:
                # Average probability from segments (if available)
                segment_probs = []
                for segment in segments:
                    if 'avg_logprob' in segment:
                        # Convert log prob to approximate probability
                        prob = np.exp(segment['avg_logprob'])
                        segment_probs.append(prob)
                
                if segment_probs:
                    avg_segment_prob = np.mean(segment_probs)
                    base_confidence += (avg_segment_prob - 0.5) * 0.4
            
            # Language specific adjustments
            if language_code in ['en', 'tr']:
                # Common languages, usually higher accuracy
                base_confidence += 0.1
            elif language_code in ['zh', 'ar', 'ja']:
                # More challenging languages
                base_confidence -= 0.05
            
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, base_confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Confidence estimation failed: {e}")
            return 0.5
    
    def _generate_alternatives(
        self, 
        detected_language: str, 
        confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Alternative diller oluştur (Whisper'ın verdiği primary sonuca dayalı tahmin)
        """
        try:
            alternatives = []
            
            # Primary detection'ı ekle
            if detected_language in self.language_names:
                alternatives.append({
                    "language": detected_language,
                    "language_name": self.language_names[detected_language],
                    "confidence": confidence
                })
            
            # Common alternatives based on primary detection
            common_alternatives = {
                'tr': [('en', 0.15), ('de', 0.08), ('ar', 0.05)],
                'en': [('tr', 0.12), ('es', 0.10), ('fr', 0.08)],
                'de': [('en', 0.18), ('tr', 0.10), ('fr', 0.07)],
                'es': [('en', 0.15), ('fr', 0.12), ('it', 0.08)],
                'fr': [('en', 0.16), ('es', 0.11), ('de', 0.07)],
                'ar': [('tr', 0.10), ('en', 0.08), ('fr', 0.05)],
                'zh': [('en', 0.12), ('ja', 0.08), ('tr', 0.05)],
                'ja': [('zh', 0.10), ('en', 0.08), ('tr', 0.04)],
                'ru': [('en', 0.14), ('de', 0.08), ('tr', 0.06)]
            }
            
            if detected_language in common_alternatives:
                for alt_lang, alt_conf in common_alternatives[detected_language]:
                    if alt_lang in self.language_names:
                        # Adjust confidence based on primary confidence
                        adjusted_conf = alt_conf * (1 - confidence)
                        alternatives.append({
                            "language": alt_lang,
                            "language_name": self.language_names[alt_lang],
                            "confidence": round(adjusted_conf, 3)
                        })
            
            # Sort by confidence
            alternatives.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return top 3
            return alternatives[:3]
            
        except Exception as e:
            logger.warning(f"Alternative generation failed: {e}")
            return []
    
    # ================================
    # SERVICE MANAGEMENT
    # ================================
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Servis istatistikleri"""
        return {
            "model_loaded": self.model_loaded,
            "model_size": self.model_size,
            "device": self.device,
            "detection_count": self.detection_count,
            "supported_languages": len(self.supported_languages),
            "supported_language_codes": list(self.supported_languages.keys()),
            "confidence_thresholds": {
                "high": self.high_confidence_threshold,
                "medium": self.medium_confidence_threshold
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Servis sağlık kontrolü"""
        try:
            # Model yükleme kontrolü
            if not self.model_loaded:
                await self.initialize_model()
            
            return {
                "status": "healthy",
                "model_ready": self.model_loaded,
                "model_size": self.model_size,
                "device": self.device,
                "detection_count": self.detection_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_ready": False
            }
    
    async def cleanup(self):
        """Servis temizleme"""
        try:
            logger.info("Cleaning up WhisperLanguageService...")
            
            # Model'i memory'den temizle
            if self.model is not None:
                del self.model
                self.model = None
            
            # CUDA cache temizle (eğer kullanılıyorsa)
            if self.device.startswith('cuda'):
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
                except ImportError:
                    pass
            
            self.model_loaded = False
            logger.info("WhisperLanguageService cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# ================================
# CONVENIENCE FUNCTIONS
# ================================

# Global service instance
_whisper_service_instance = None

def get_whisper_language_service() -> WhisperLanguageService:
    """Global WhisperLanguageService instance'ı al"""
    global _whisper_service_instance
    if _whisper_service_instance is None:
        _whisper_service_instance = WhisperLanguageService()
    return _whisper_service_instance

async def detect_language_from_audio(
    audio_data: np.ndarray, 
    sample_rate: int
) -> Dict[str, Any]:
    """Convenience function - audio'dan dil tespiti"""
    service = get_whisper_language_service()
    return await service.analyze(audio_data, sample_rate)

async def detect_language_from_file(file_path: str) -> Dict[str, Any]:
    """Convenience function - dosyadan dil tespiti"""
    service = get_whisper_language_service()
    return await service.detect_language_from_file(file_path)

# ================================
# EXAMPLE USAGE & TESTING
# ================================

if __name__ == "__main__":
    import sys
    
    async def test_whisper_service():
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            
            try:
                print(f"Testing WhisperLanguageService with: {test_file}")
                
                # Servis oluştur
                service = get_whisper_language_service()
                
                # Health check
                health = await service.health_check()
                print(f"Health check: {health}")
                
                # Dosyadan dil tespiti
                print(f"\n=== Language Detection ===")
                result = await service.detect_language_from_file(test_file)
                print(f"Detected language: {result['detected_language_name']} ({result['confidence']:.3f})")
                print(f"Enum: {result['language']}")
                print(f"Text sample: {result['transcribed_text'][:100]}...")
                
                if result['alternatives']:
                    print(f"Alternatives:")
                    for alt in result['alternatives']:
                        print(f"  - {alt['language_name']}: {alt['confidence']:.3f}")
                
                # Stats
                stats = service.get_service_stats()
                print(f"\n=== Service Stats ===")
                print(f"Detections performed: {stats['detection_count']}")
                print(f"Model: {stats['model_size']} on {stats['device']}")
                
            except Exception as e:
                print(f"Test failed: {e}")
        else:
            print("Usage: python whisper_language_service.py <audio_file>")
            print("WhisperLanguageService module loaded successfully!")
    
    # Async test çalıştır
    import asyncio
    asyncio.run(test_whisper_service())