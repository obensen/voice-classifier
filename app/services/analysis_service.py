# app/services/analysis_service.py
# Ana analiz servis koordinatörü - tüm ses analizi işlemlerini yönetir

import asyncio
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Local imports
from app.models.schemas import (
    AnalysisResult, Gender, AgeGroup, Language, Tone, Emotion
)
from app.utils.audio_loader import AudioLoader, AudioLoadError
from app.utils.feature_extract import AudioFeatureExtractor
from app.utils.cache import cache, metrics
from app.config import settings

logger = logging.getLogger(__name__)

class AnalysisServiceError(Exception):
    """Analiz servisi hatası"""
    pass

class AnalysisService:
    """
    Ana analiz servis sınıfı
    Tüm ses analizi işlemlerini koordine eder ve yönetir
    """
    
    def __init__(self):
        """AnalysisService'i başlat"""
        self.audio_loader = AudioLoader()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Alt servisler (lazy loading)
        self._language_service = None
        self._gender_service = None
        self._age_service = None
        self._tone_service = None
        self._emotion_service = None
        
        # Service durumu
        self.services_initialized = False
        self.analysis_count = 0
        self.total_processing_time = 0.0
        
        logger.info("AnalysisService initialized")
    
    async def initialize_services(self):
        """Tüm alt servisleri lazy loading ile başlat"""
        if self.services_initialized:
            return
        
        try:
            logger.info("Initializing analysis sub-services...")
            start_time = time.time()
            
            # Alt servisleri import et ve başlat
            # Şu an için mock implementasyonlar kullanacağız
            # Gerçek implementasyonlar tamamlandığında aşağıdaki satırlar açılacak:
            
            # from app.services.whisper_language_service import WhisperLanguageService
            # from app.services.simple_gender_service import SimpleGenderService
            # from app.services.simple_age_service import SimpleAgeService
            # from app.services.simple_tone_service import SimpleToneService
            # from app.services.simple_emotion_service import SimpleEmotionService
            
            # self._language_service = WhisperLanguageService()
            # self._gender_service = SimpleGenderService()
            # self._age_service = SimpleAgeService()
            # self._tone_service = SimpleToneService()
            # self._emotion_service = SimpleEmotionService()
            
            # Mock services (geçici)
            self._language_service = MockLanguageService()
            self._gender_service = MockGenderService()
            self._age_service = MockAgeService()
            self._tone_service = MockToneService()
            self._emotion_service = MockEmotionService()
            
            self.services_initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"All analysis services initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise AnalysisServiceError(f"Service initialization failed: {e}")
    
    async def analyze_audio_file(
        self, 
        file_path: str, 
        analysis_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> AnalysisResult:
        """
        Ses dosyasını tam analiz et
        
        Args:
            file_path: Ses dosyası yolu
            analysis_types: Hangi analizlerin yapılacağı (None = tümü)
            use_cache: Cache kullanılsın mı
            
        Returns:
            AnalysisResult: Analiz sonuçları
        """
        try:
            start_time = time.time()
            
            # Cache kontrolü
            if use_cache:
                cache_key = f"{file_path}_{Path(file_path).stat().st_mtime}"
                cached_result = cache.get(cache_key, "full_analysis")
                
                if cached_result:
                    metrics.hit()
                    logger.info(f"Cache hit for full analysis: {Path(file_path).name}")
                    return cached_result
                
                metrics.miss()
            
            # Servisleri başlat
            await self.initialize_services()
            
            # Audio dosyasını yükle
            logger.info(f"Loading audio file: {Path(file_path).name}")
            audio_data, sample_rate = self.audio_loader.load_audio(file_path)
            
            # Feature extraction
            logger.info("Extracting features...")
            features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
            
            # Hangi analizlerin yapılacağını belirle
            if analysis_types is None:
                analysis_types = ["language", "gender", "age", "tone", "emotion"]
            
            # Parallel analysis
            analysis_tasks = []
            
            if "language" in analysis_types:
                analysis_tasks.append(self._analyze_language(audio_data, sample_rate, features))
            if "gender" in analysis_types:
                analysis_tasks.append(self._analyze_gender(audio_data, sample_rate, features))
            if "age" in analysis_types:
                analysis_tasks.append(self._analyze_age(audio_data, sample_rate, features))
            if "tone" in analysis_types:
                analysis_tasks.append(self._analyze_tone(audio_data, sample_rate, features))
            if "emotion" in analysis_types:
                analysis_tasks.append(self._analyze_emotion(audio_data, sample_rate, features))
            
            # Paralel analiz çalıştır
            logger.info(f"Running {len(analysis_tasks)} parallel analyses...")
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Sonuçları birleştir
            final_result = self._combine_analysis_results(analysis_results, analysis_types, features)
            
            # İstatistikleri güncelle
            processing_time = time.time() - start_time
            self.analysis_count += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Analysis completed in {processing_time:.2f}s for {Path(file_path).name}")
            
            # Cache'e kaydet
            if use_cache:
                cache.set(cache_key, "full_analysis", final_result, ttl=300)
            
            return final_result
            
        except AudioLoadError:
            raise AnalysisServiceError("Audio file could not be loaded")
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            raise AnalysisServiceError(f"Analysis failed: {e}")
    
    async def analyze_single_aspect(
        self, 
        file_path: str, 
        aspect: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Tek bir analiz türü yap (gender, age, language, tone, emotion)
        
        Args:
            file_path: Ses dosyası yolu
            aspect: Analiz türü
            use_cache: Cache kullanılsın mı
            
        Returns:
            Dict: Analiz sonucu
        """
        try:
            # Cache kontrolü
            if use_cache:
                cache_key = f"{file_path}_{Path(file_path).stat().st_mtime}"
                cached_result = cache.get(cache_key, aspect)
                
                if cached_result:
                    metrics.hit()
                    logger.info(f"Cache hit for {aspect} analysis: {Path(file_path).name}")
                    return cached_result
                
                metrics.miss()
            
            # Servisleri başlat
            await self.initialize_services()
            
            # Audio yükle ve feature çıkar
            audio_data, sample_rate = self.audio_loader.load_audio(file_path)
            
            # Aspect'e göre özelleşmiş feature extraction
            if aspect in ["gender", "age", "emotion"]:
                features = self.feature_extractor.extract_specialized_features(audio_data, sample_rate, aspect)
            else:
                features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
            
            # İlgili analizi yap
            if aspect == "language":
                result = await self._analyze_language(audio_data, sample_rate, features)
            elif aspect == "gender":
                result = await self._analyze_gender(audio_data, sample_rate, features)
            elif aspect == "age":
                result = await self._analyze_age(audio_data, sample_rate, features)
            elif aspect == "tone":
                result = await self._analyze_tone(audio_data, sample_rate, features)
            elif aspect == "emotion":
                result = await self._analyze_emotion(audio_data, sample_rate, features)
            else:
                raise ValueError(f"Unknown analysis aspect: {aspect}")
            
            # Cache'e kaydet
            if use_cache:
                cache.set(cache_key, aspect, result, ttl=300)
            
            logger.info(f"{aspect.capitalize()} analysis completed for {Path(file_path).name}")
            return result
            
        except Exception as e:
            logger.error(f"{aspect.capitalize()} analysis failed: {e}")
            raise AnalysisServiceError(f"{aspect.capitalize()} analysis failed: {e}")
    
    async def batch_analyze(
        self, 
        file_paths: List[str],
        analysis_types: Optional[List[str]] = None,
        max_concurrent: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Toplu ses dosyası analizi
        
        Args:
            file_paths: Ses dosya yolları
            analysis_types: Analiz türleri
            max_concurrent: Maksimum paralel işlem sayısı
            
        Returns:
            List[Dict]: Her dosya için analiz sonuçları
        """
        try:
            logger.info(f"Starting batch analysis for {len(file_paths)} files")
            
            # Semaphore ile paralel işlem sayısını sınırla
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def analyze_single_file(file_path: str) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await self.analyze_audio_file(file_path, analysis_types)
                        return {
                            "file_name": Path(file_path).name,
                            "file_path": file_path,
                            "success": True,
                            "result": result,
                            "error": None
                        }
                    except Exception as e:
                        logger.warning(f"Failed to analyze {file_path}: {e}")
                        return {
                            "file_name": Path(file_path).name,
                            "file_path": file_path,
                            "success": False,
                            "result": None,
                            "error": str(e)
                        }
            
            # Tüm dosyaları paralel işle
            batch_tasks = [analyze_single_file(fp) for fp in file_paths]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Sonuçları filtrele
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis task failed: {result}")
                    final_results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    final_results.append(result)
            
            # İstatistikleri logla
            successful = sum(1 for r in final_results if r.get("success"))
            logger.info(f"Batch analysis completed: {successful}/{len(file_paths)} successful")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise AnalysisServiceError(f"Batch analysis failed: {e}")
    
    # ================================
    # PRIVATE ANALYSIS METHODS
    # ================================
    
    async def _analyze_language(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        """Dil analizi"""
        try:
            result = await self._language_service.analyze(audio_data, sample_rate, features)
            return {
                "language": result["language"],
                "confidence": result["confidence"],
                "detected_language": result.get("detected_language", ""),
                "alternatives": result.get("alternatives", [])
            }
        except Exception as e:
            logger.warning(f"Language analysis failed: {e}")
            return {
                "language": Language.UNKNOWN,
                "confidence": 0.0,
                "detected_language": "unknown",
                "alternatives": []
            }
    
    async def _analyze_gender(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        """Cinsiyet analizi"""
        try:
            result = await self._gender_service.analyze(audio_data, sample_rate, features)
            return {
                "gender": result["gender"],
                "confidence": result["confidence"],
                "probabilities": result.get("probabilities", {})
            }
        except Exception as e:
            logger.warning(f"Gender analysis failed: {e}")
            return {
                "gender": Gender.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    async def _analyze_age(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        """Yaş analizi"""
        try:
            result = await self._age_service.analyze(audio_data, sample_rate, features)
            return {
                "age_group": result["age_group"],
                "confidence": result["confidence"],
                "probabilities": result.get("probabilities", {}),
                "estimated_age_range": result.get("estimated_age_range", "")
            }
        except Exception as e:
            logger.warning(f"Age analysis failed: {e}")
            return {
                "age_group": AgeGroup.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {},
                "estimated_age_range": "unknown"
            }
    
    async def _analyze_tone(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        """Ton analizi"""
        try:
            result = await self._tone_service.analyze(audio_data, sample_rate, features)
            return {
                "tone": result["tone"],
                "confidence": result["confidence"],
                "probabilities": result.get("probabilities", {}),
                "tone_characteristics": result.get("tone_characteristics", [])
            }
        except Exception as e:
            logger.warning(f"Tone analysis failed: {e}")
            return {
                "tone": Tone.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {},
                "tone_characteristics": []
            }
    
    async def _analyze_emotion(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        """Duygu analizi"""
        try:
            result = await self._emotion_service.analyze(audio_data, sample_rate, features)
            return {
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "probabilities": result.get("probabilities", {}),
                "emotional_intensity": result.get("emotional_intensity", 0.0),
                "valence": result.get("valence", 0.0),  # Pozitif/negatif
                "arousal": result.get("arousal", 0.0)   # Enerji seviyesi
            }
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return {
                "emotion": Emotion.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {},
                "emotional_intensity": 0.0,
                "valence": 0.0,
                "arousal": 0.0
            }
    
    def _combine_analysis_results(
        self, 
        analysis_results: List[Any], 
        analysis_types: List[str],
        features: Dict[str, Any]
    ) -> AnalysisResult:
        """Analiz sonuçlarını birleştir"""
        try:
            # Varsayılan değerler
            combined = {
                "gender": Gender.UNKNOWN,
                "age_group": AgeGroup.UNKNOWN,
                "language": Language.UNKNOWN,
                "tone": Tone.UNKNOWN,
                "emotion": Emotion.UNKNOWN,
                "confidence": {},
                "features": features
            }
            
            # Sonuçları birleştir
            for i, analysis_type in enumerate(analysis_types):
                if i < len(analysis_results) and not isinstance(analysis_results[i], Exception):
                    result = analysis_results[i]
                    
                    if analysis_type == "language" and result:
                        combined["language"] = result.get("language", Language.UNKNOWN)
                        combined["confidence"]["language"] = result.get("confidence", 0.0)
                        
                    elif analysis_type == "gender" and result:
                        combined["gender"] = result.get("gender", Gender.UNKNOWN)
                        combined["confidence"]["gender"] = result.get("confidence", 0.0)
                        
                    elif analysis_type == "age" and result:
                        combined["age_group"] = result.get("age_group", AgeGroup.UNKNOWN)
                        combined["confidence"]["age_group"] = result.get("confidence", 0.0)
                        
                    elif analysis_type == "tone" and result:
                        combined["tone"] = result.get("tone", Tone.UNKNOWN)
                        combined["confidence"]["tone"] = result.get("confidence", 0.0)
                        
                    elif analysis_type == "emotion" and result:
                        combined["emotion"] = result.get("emotion", Emotion.UNKNOWN)
                        combined["confidence"]["emotion"] = result.get("confidence", 0.0)
                else:
                    # Hata durumunda confidence 0 yap
                    combined["confidence"][analysis_type] = 0.0
            
            return AnalysisResult(**combined)
            
        except Exception as e:
            logger.error(f"Failed to combine analysis results: {e}")
            # En temel sonucu döndür
            return AnalysisResult(
                gender=Gender.UNKNOWN,
                age_group=AgeGroup.UNKNOWN,
                language=Language.UNKNOWN,
                tone=Tone.UNKNOWN,
                emotion=Emotion.UNKNOWN,
                confidence={
                    "gender": 0.0,
                    "age_group": 0.0,
                    "language": 0.0,
                    "tone": 0.0,
                    "emotion": 0.0
                },
                features=features if features else {}
            )
    
    # ================================
    # SERVICE STATISTICS & MANAGEMENT
    # ================================
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Servis istatistiklerini al"""
        avg_processing_time = (
            self.total_processing_time / self.analysis_count 
            if self.analysis_count > 0 else 0.0
        )
        
        return {
            "services_initialized": self.services_initialized,
            "analysis_count": self.analysis_count,
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "cache_hits": metrics.hits,
            "cache_misses": metrics.misses,
            "cache_hit_rate": round(metrics.get_hit_rate(), 3),
            "available_services": [
                "language", "gender", "age", "tone", "emotion"
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Servis sağlık kontrolü"""
        try:
            await self.initialize_services()
            
            health_status = {
                "status": "healthy",
                "services_ready": self.services_initialized,
                "sub_services": {
                    "language_service": self._language_service is not None,
                    "gender_service": self._gender_service is not None,
                    "age_service": self._age_service is not None,
                    "tone_service": self._tone_service is not None,
                    "emotion_service": self._emotion_service is not None
                },
                "audio_loader": True,
                "feature_extractor": True
            }
            
            # Alt servislerin sağlığını kontrol et
            all_healthy = all(health_status["sub_services"].values())
            if not all_healthy:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "services_ready": False
            }
    
    async def cleanup(self):
        """Servis temizleme"""
        try:
            logger.info("Cleaning up AnalysisService...")
            
            # Alt servisleri temizle
            if hasattr(self._language_service, 'cleanup'):
                await self._language_service.cleanup()
            if hasattr(self._gender_service, 'cleanup'):
                await self._gender_service.cleanup()
            if hasattr(self._age_service, 'cleanup'):
                await self._age_service.cleanup()
            if hasattr(self._tone_service, 'cleanup'):
                await self._tone_service.cleanup()
            if hasattr(self._emotion_service, 'cleanup'):
                await self._emotion_service.cleanup()
            
            # Servisleri sıfırla
            self._language_service = None
            self._gender_service = None
            self._age_service = None
            self._tone_service = None
            self._emotion_service = None
            
            self.services_initialized = False
            logger.info("AnalysisService cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# ================================
# MOCK SERVICES (TEMPORARY)
# ================================

class MockLanguageService:
    """Geçici mock dil servisi"""
    
    async def analyze(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        # Basit mock implementasyon
        await asyncio.sleep(0.1)  # Simüle edilen işlem süresi
        
        # Features'dan basit heuristik
        duration = features.get("duration", 0)
        
        if duration > 10:
            language = Language.TURKISH
            confidence = 0.85
        elif duration > 5:
            language = Language.ENGLISH
            confidence = 0.78
        else:
            language = Language.UNKNOWN
            confidence = 0.45
        
        return {
            "language": language,
            "confidence": confidence,
            "detected_language": language.value,
            "alternatives": [
                {"language": "tr", "confidence": 0.85},
                {"language": "en", "confidence": 0.12},
                {"language": "de", "confidence": 0.03}
            ]
        }

class MockGenderService:
    """Geçici mock cinsiyet servisi"""
    
    async def analyze(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        
        # F0 tabanlı basit mock
        f0_mean = features.get("f0_mean", 150)
        
        if f0_mean > 180:
            gender = Gender.FEMALE
            confidence = 0.82
        elif f0_mean > 120:
            gender = Gender.MALE
            confidence = 0.79
        else:
            gender = Gender.UNKNOWN
            confidence = 0.35
        
        return {
            "gender": gender,
            "confidence": confidence,
            "probabilities": {
                "male": 0.21 if gender == Gender.FEMALE else 0.79,
                "female": 0.82 if gender == Gender.FEMALE else 0.18,
                "unknown": 0.03
            }
        }

class MockAgeService:
    """Geçici mock yaş servisi"""
    
    async def analyze(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        await asyncio.sleep(0.08)
        
        # Voice stability tabanlı mock
        voice_stability = features.get("voice_stability", 0.5)
        speech_rate = features.get("speech_rate", 3.0)
        
        if voice_stability > 0.7 and speech_rate > 4:
            age_group = AgeGroup.YOUNG
            confidence = 0.74
            age_range = "18-25"
        elif voice_stability > 0.6:
            age_group = AgeGroup.ADULT
            confidence = 0.81
            age_range = "26-50"
        elif speech_rate < 2:
            age_group = AgeGroup.SENIOR
            confidence = 0.68
            age_range = "60+"
        else:
            age_group = AgeGroup.ADULT
            confidence = 0.55
            age_range = "25-60"
        
        return {
            "age_group": age_group,
            "confidence": confidence,
            "estimated_age_range": age_range,
            "probabilities": {
                "child": 0.05,
                "young": 0.25,
                "adult": 0.55,
                "senior": 0.15
            }
        }

class MockToneService:
    """Geçici mock ton servisi"""
    
    async def analyze(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        await asyncio.sleep(0.06)
        
        # Energy ve pitch variation tabanlı mock
        energy_mean = features.get("energy_mean", 0.03)
        pitch_variation = features.get("pitch_variation", 0.1)
        
        if energy_mean > 0.05 and pitch_variation > 0.15:
            tone = Tone.EXCITED
            confidence = 0.76
        elif energy_mean < 0.02:
            tone = Tone.CALM
            confidence = 0.71
        elif pitch_variation < 0.05:
            tone = Tone.FORMAL
            confidence = 0.68
        else:
            tone = Tone.NEUTRAL
            confidence = 0.62
        
        return {
            "tone": tone,
            "confidence": confidence,
            "probabilities": {
                "formal": 0.25,
                "neutral": 0.35,
                "calm": 0.20,
                "excited": 0.15,
                "friendly": 0.05
            },
            "tone_characteristics": [
                f"Energy level: {'high' if energy_mean > 0.04 else 'moderate' if energy_mean > 0.02 else 'low'}",
                f"Pitch variation: {'high' if pitch_variation > 0.12 else 'moderate' if pitch_variation > 0.06 else 'low'}"
            ]
        }

class MockEmotionService:
    """Geçici mock duygu servisi"""
    
    async def analyze(self, audio_data, sample_rate, features) -> Dict[str, Any]:
        await asyncio.sleep(0.07)
        
        # Multi-feature mock
        energy = features.get("energy_mean", 0.03)
        pitch_range = features.get("pitch_range", 50)
        tempo = features.get("tempo", 120)
        
        # Basit emotion classification
        if energy > 0.05 and pitch_range > 80 and tempo > 130:
            emotion = Emotion.EXCITED
            confidence = 0.78
            valence = 0.8
            arousal = 0.9
        elif energy < 0.02 and pitch_range < 30:
            emotion = Emotion.SAD
            confidence = 0.72
            valence = 0.2
            arousal = 0.3
        elif pitch_range > 100:
            emotion = Emotion.HAPPY
            confidence = 0.69
            valence = 0.75
            arousal = 0.65
        else:
            emotion = Emotion.NEUTRAL
            confidence = 0.58
            valence = 0.5
            arousal = 0.5
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                "happy": 0.25,
                "sad": 0.15,
                "neutral": 0.35,
                "excited": 0.20,
                "angry": 0.05
            },
            "emotional_intensity": confidence,
            "valence": valence,
            "arousal": arousal
        }

# ================================
# GLOBAL SERVICE INSTANCE
# ================================

# Singleton pattern
_analysis_service_instance = None

def get_analysis_service() -> AnalysisService:
    """Global AnalysisService instance'ı al"""
    global _analysis_service_instance
    if _analysis_service_instance is None:
        _analysis_service_instance = AnalysisService()
    return _analysis_service_instance

# ================================
# CONVENIENCE FUNCTIONS
# ================================

async def analyze_audio_file(file_path: str, analysis_types: Optional[List[str]] = None) -> AnalysisResult:
    """Convenience function - ses dosyası analizi"""
    service = get_analysis_service()
    return await service.analyze_audio_file(file_path, analysis_types)

async def analyze_single_aspect(file_path: str, aspect: str) -> Dict[str, Any]:
    """Convenience function - tek aspectli analiz"""
    service = get_analysis_service()
    return await service.analyze_single_aspect(file_path, aspect)

# ================================
# EXAMPLE USAGE & TESTING
# ================================

if __name__ == "__main__":
    import sys
    
    async def test_analysis_service():
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            
            try:
                print(f"Testing AnalysisService with: {test_file}")
                
                # Servis örneği oluştur
                service = get_analysis_service()
                
                # Health check
                health = await service.health_check()
                print(f"Health check: {health}")
                
                # Tam analiz
                print("\n=== Full Analysis ===")
                result = await service.analyze_audio_file(test_file)
                print(f"Result: {result}")
                
                # Tek aspect analizler
                print(f"\n=== Individual Aspects ===")
                for aspect in ["gender", "language", "emotion"]:
                    try:
                        aspect_result = await service.analyze_single_aspect(test_file, aspect)
                        print(f"{aspect.capitalize()}: {aspect_result}")
                    except Exception as e:
                        print(f"{aspect.capitalize()} failed: {e}")
                
                # İstatistikler
                stats = service.get_service_stats()
                print(f"\n=== Service Stats ===")
                print(f"Stats: {stats}")
                
            except Exception as e:
                print(f"Test failed: {e}")
        else:
            print("Usage: python analysis_service.py <audio_file>")
            print("AnalysisService module loaded successfully!")
    
    # Async test çalıştır
    import asyncio
    asyncio.run(test_analysis_service())