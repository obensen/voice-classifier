"""
Advanced Age Classification Service for Voice Analysis

Bu servis, ses özelliklerini kullanarak konuşmacının yaş grubunu tespit eder.
Fundamental frequency (F0), formant frequencies, spektral özellikler ve
ses kalitesi metriklerini kullanarak hibrit bir yaklaşım benimser.

Features:
- F0 (Fundamental Frequency) analysis - yaş ile ters korelasyon
- Formant frequency extraction (F1, F2, F3) - vocal tract length
- Spectral features (centroid, rolloff, bandwidth) - voice aging
- Voice quality metrics (jitter, shimmer, HNR) - stability
- ML Classification + Heuristic rules
- Multi-metric confidence scoring
- Async processing with caching
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

from app.models.schemas import (
    AgeCategory, AgeAnalysisResult, AnalysisDetail,
    AudioFeatures, ProcessingError
)
from app.config import get_settings
from app.utils.cache import cache_manager
from app.utils.feature_extract import FeatureExtractor

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class SimpleAgeService:
    """
    Advanced Age Classification Service
    
    Yaş gruplarını tespit etmek için çoklu akustik özellik kullanır:
    - F0 analysis: Fundamental frequency patterns
    - Formant analysis: Vocal tract characteristics  
    - Spectral analysis: Frequency distribution patterns
    - Voice quality: Stability and aging indicators
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_extractor = None
        self.scaler = None
        self.classifier = None
        self._model_loaded = False
        self._lock = asyncio.Lock()
        
        # Age classification thresholds (from research)
        self.age_thresholds = {
            'f0': {
                'child_min': 200.0,      # Children: 200-400 Hz
                'child_max': 400.0,
                'teen_min': 150.0,       # Teens: 150-250 Hz  
                'teen_max': 250.0,
                'young_min': 120.0,      # Young adults: 120-200 Hz
                'young_max': 200.0,
                'middle_min': 100.0,     # Middle-aged: 100-180 Hz
                'middle_max': 180.0,
                'senior_max': 120.0      # Seniors: < 120 Hz (typically)
            },
            'formant': {
                'f1_child_min': 800.0,   # Higher formants for children
                'f1_adult_max': 700.0,   # Lower formants for adults
                'f2_child_min': 2000.0,
                'f2_adult_max': 1800.0
            },
            'quality': {
                'jitter_threshold': 0.02,    # Jitter increases with age
                'shimmer_threshold': 0.15,   # Shimmer increases with age
                'hnr_threshold': 15.0        # HNR decreases with age
            }
        }
        
        # Confidence calculation weights
        self.confidence_weights = {
            'f0_consistency': 0.3,
            'formant_reliability': 0.25,
            'spectral_stability': 0.2,
            'voice_quality': 0.15,
            'ml_confidence': 0.1
        }

    async def _ensure_model_loaded(self) -> None:
        """ML modelini lazy loading ile yükler"""
        if self._model_loaded:
            return
            
        async with self._lock:
            if self._model_loaded:
                return
                
            try:
                logger.info("Age classification modelini yükleniyor...")
                
                # Feature extractor'ı başlat
                if not self.feature_extractor:
                    self.feature_extractor = FeatureExtractor()
                
                # ML modelini yükle veya oluştur
                model_path = Path(self.settings.models_dir) / "age_classifier.joblib"
                scaler_path = Path(self.settings.models_dir) / "age_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    # Pre-trained model varsa yükle
                    self.classifier = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Pre-trained age model yüklendi")
                else:
                    # Basit model oluştur (production'da gerçek data ile train edilecek)
                    self.classifier = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.scaler = StandardScaler()
                    logger.warning("Mock age model oluşturuldu - production'da train edilmeli")
                
                self._model_loaded = True
                logger.info("Age classification servisi hazır")
                
            except Exception as e:
                logger.error(f"Age model yükleme hatası: {e}")
                # Fallback: sadece heuristic rules kullan
                self._model_loaded = True

    async def analyze_age(
        self, 
        audio_file: str, 
        features: Optional[AudioFeatures] = None
    ) -> AgeAnalysisResult:
        """
        Ses dosyasından yaş grubu analizi yapar
        
        Args:
            audio_file: Analiz edilecek ses dosyası yolu
            features: Önceden çıkarılmış özellikler (optional)
            
        Returns:
            AgeAnalysisResult: Yaş analizi sonucu
        """
        try:
            # Cache kontrolü
            cache_key = f"age_analysis:{Path(audio_file).stem}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Age analysis cache hit: {audio_file}")
                return AgeAnalysisResult(**cached_result)

            logger.info(f"Age analysis başlatılıyor: {audio_file}")
            await self._ensure_model_loaded()
            
            # Feature extraction
            if not features:
                if not self.feature_extractor:
                    self.feature_extractor = FeatureExtractor()
                features = await self.feature_extractor.extract_features(audio_file)
            
            # Akustik analiz yap
            acoustic_analysis = await self._analyze_acoustic_features(audio_file, features)
            
            # Age classification
            age_category = await self._classify_age(acoustic_analysis)
            
            # Confidence hesaplama
            confidence = await self._calculate_confidence(acoustic_analysis)
            
            # Estimated age range
            age_range = self._get_age_range(age_category)
            
            # Analysis details
            details = self._create_analysis_details(acoustic_analysis, age_category)
            
            result = AgeAnalysisResult(
                age_category=age_category,
                confidence=confidence,
                estimated_age_range=age_range,
                analysis_details=details,
                acoustic_features=acoustic_analysis
            )
            
            # Cache sonucu
            await cache_manager.set(
                cache_key, 
                result.dict(), 
                ttl=self.settings.cache_ttl_seconds
            )
            
            logger.info(f"Age analysis tamamlandı: {age_category.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Age analysis hatası: {e}")
            return self._create_error_result(str(e))

    async def _analyze_acoustic_features(
        self, 
        audio_file: str, 
        features: AudioFeatures
    ) -> Dict[str, Any]:
        """Yaş tespiti için akustik özellikler analiz eder"""
        try:
            # Audio dosyasını yükle
            y, sr = librosa.load(audio_file, sr=None)
            
            analysis = {}
            
            # 1. Fundamental Frequency (F0) Analysis
            f0_analysis = await self._analyze_f0(y, sr)
            analysis['f0'] = f0_analysis
            
            # 2. Formant Analysis
            formant_analysis = await self._analyze_formants(y, sr)
            analysis['formants'] = formant_analysis
            
            # 3. Spectral Features
            spectral_analysis = await self._analyze_spectral_features(features)
            analysis['spectral'] = spectral_analysis
            
            # 4. Voice Quality Metrics
            quality_analysis = await self._analyze_voice_quality(y, sr)
            analysis['quality'] = quality_analysis
            
            # 5. Additional Features
            analysis['additional'] = {
                'duration': len(y) / sr,
                'energy_mean': float(np.mean(librosa.feature.rms(y=y))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y)))
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Acoustic analysis hatası: {e}")
            return {}

    async def _analyze_f0(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Fundamental frequency (F0) analizi"""
        try:
            # Pitch extraction using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
            
            # Extract F0 values
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            if not f0_values:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'range': 0.0}
            
            f0_array = np.array(f0_values)
            
            return {
                'mean': float(np.mean(f0_array)),
                'std': float(np.std(f0_array)),
                'median': float(np.median(f0_array)),
                'min': float(np.min(f0_array)),
                'max': float(np.max(f0_array)),
                'range': float(np.max(f0_array) - np.min(f0_array)),
                'consistency': float(1.0 - (np.std(f0_array) / np.mean(f0_array)) if np.mean(f0_array) > 0 else 0.0)
            }
            
        except Exception as e:
            logger.error(f"F0 analysis hatası: {e}")
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'range': 0.0}

    async def _analyze_formants(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Formant frequency analizi (vocal tract characteristics)"""
        try:
            # LPC analysis for formant estimation
            # Bu basitleştirilmiş bir yaklaşım - production'da daha gelişmiş yöntemler kullanılabilir
            
            # Pre-emphasis filter
            pre_emphasis = 0.97
            emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Windowing
            frame_size = int(0.025 * sr)  # 25ms frames
            frame_stride = int(0.01 * sr)  # 10ms stride
            
            formant_estimates = []
            
            # Basit formant estimation using spectral peaks
            for i in range(0, len(emphasized) - frame_size, frame_stride):
                frame = emphasized[i:i + frame_size]
                
                # FFT
                fft = np.fft.rfft(frame * np.hanning(len(frame)))
                magnitude = np.abs(fft)
                
                # Find peaks (simplified formant estimation)
                freqs = np.fft.rfftfreq(len(frame), 1/sr)
                
                # Look for first 3 formants in typical ranges
                f1_range = (200, 1000)  # F1 typical range
                f2_range = (900, 3000)  # F2 typical range
                f3_range = (2000, 4000) # F3 typical range
                
                f1_idx = np.where((freqs >= f1_range[0]) & (freqs <= f1_range[1]))[0]
                f2_idx = np.where((freqs >= f2_range[0]) & (freqs <= f2_range[1]))[0]
                f3_idx = np.where((freqs >= f3_range[0]) & (freqs <= f3_range[1]))[0]
                
                if len(f1_idx) > 0 and len(f2_idx) > 0:
                    f1 = freqs[f1_idx[np.argmax(magnitude[f1_idx])]]
                    f2 = freqs[f2_idx[np.argmax(magnitude[f2_idx])]]
                    f3 = freqs[f3_idx[np.argmax(magnitude[f3_idx])]] if len(f3_idx) > 0 else 0
                    
                    formant_estimates.append([f1, f2, f3])
            
            if not formant_estimates:
                return {'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0}
            
            formants = np.array(formant_estimates)
            
            return {
                'f1_mean': float(np.mean(formants[:, 0])),
                'f2_mean': float(np.mean(formants[:, 1])),
                'f3_mean': float(np.mean(formants[:, 2])) if formants.shape[1] > 2 else 0.0,
                'f1_std': float(np.std(formants[:, 0])),
                'f2_std': float(np.std(formants[:, 1])),
                'formant_dispersion': float(np.mean(formants[:, 1] - formants[:, 0]))
            }
            
        except Exception as e:
            logger.error(f"Formant analysis hatası: {e}")
            return {'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0}

    async def _analyze_spectral_features(self, features: AudioFeatures) -> Dict[str, float]:
        """Spektral özellikler analizi (aging indicators)"""
        try:
            spectral_data = features.spectral_features
            
            return {
                'centroid_mean': spectral_data.get('spectral_centroid_mean', 0.0),
                'centroid_std': spectral_data.get('spectral_centroid_std', 0.0),
                'rolloff_mean': spectral_data.get('spectral_rolloff_mean', 0.0),
                'bandwidth_mean': spectral_data.get('spectral_bandwidth_mean', 0.0),
                'contrast_mean': spectral_data.get('spectral_contrast_mean', 0.0),
                'flatness_mean': spectral_data.get('spectral_flatness_mean', 0.0),
                'high_freq_ratio': spectral_data.get('spectral_centroid_mean', 0.0) / 4000.0  # Normalized
            }
            
        except Exception as e:
            logger.error(f"Spectral analysis hatası: {e}")
            return {}

    async def _analyze_voice_quality(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Ses kalitesi metrikleri (jitter, shimmer, HNR)"""
        try:
            # Simplified voice quality metrics
            # Production'da daha gelişmiş algoritma kullanılmalı
            
            # Jitter approximation (pitch period variation)
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            f0_contour = np.max(pitches, axis=0)
            valid_f0 = f0_contour[f0_contour > 0]
            
            if len(valid_f0) > 1:
                periods = 1.0 / valid_f0
                jitter = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0
            else:
                jitter = 0.0
            
            # Shimmer approximation (amplitude variation)
            rms = librosa.feature.rms(y=y)[0]
            shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
            
            # HNR approximation (Harmonics-to-Noise Ratio)
            # Simplified calculation
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            harmonic_strength = np.mean(magnitude)
            noise_estimate = np.std(magnitude)
            hnr = 20 * np.log10(harmonic_strength / noise_estimate) if noise_estimate > 0 else 0
            
            return {
                'jitter': float(np.clip(jitter, 0, 1)),
                'shimmer': float(np.clip(shimmer, 0, 1)),
                'hnr': float(np.clip(hnr, 0, 50)),
                'voice_stability': float(1.0 - (jitter + shimmer) / 2.0)
            }
            
        except Exception as e:
            logger.error(f"Voice quality analysis hatası: {e}")
            return {'jitter': 0.0, 'shimmer': 0.0, 'hnr': 20.0}

    async def _classify_age(self, acoustic_analysis: Dict[str, Any]) -> AgeCategory:
        """Akustik analiz sonuçlarından yaş kategorisi belirler"""
        try:
            # Heuristic rules based classification
            age_scores = {
                AgeCategory.CHILD: 0.0,
                AgeCategory.TEEN: 0.0,
                AgeCategory.YOUNG_ADULT: 0.0,
                AgeCategory.MIDDLE_AGED: 0.0,
                AgeCategory.SENIOR: 0.0
            }
            
            # F0 based scoring
            f0_mean = acoustic_analysis.get('f0', {}).get('mean', 150.0)
            
            if f0_mean >= self.age_thresholds['f0']['child_min']:
                age_scores[AgeCategory.CHILD] += 3.0
            elif f0_mean >= self.age_thresholds['f0']['teen_min']:
                age_scores[AgeCategory.TEEN] += 2.5
            elif f0_mean >= self.age_thresholds['f0']['young_min']:
                age_scores[AgeCategory.YOUNG_ADULT] += 2.0
            elif f0_mean >= self.age_thresholds['f0']['middle_min']:
                age_scores[AgeCategory.MIDDLE_AGED] += 1.5
            else:
                age_scores[AgeCategory.SENIOR] += 1.0
            
            # Formant based scoring
            formants = acoustic_analysis.get('formants', {})
            f1_mean = formants.get('f1_mean', 500.0)
            f2_mean = formants.get('f2_mean', 1500.0)
            
            if (f1_mean > self.age_thresholds['formant']['f1_child_min'] or 
                f2_mean > self.age_thresholds['formant']['f2_child_min']):
                age_scores[AgeCategory.CHILD] += 2.0
                age_scores[AgeCategory.TEEN] += 1.0
            else:
                age_scores[AgeCategory.YOUNG_ADULT] += 1.5
                age_scores[AgeCategory.MIDDLE_AGED] += 1.0
                age_scores[AgeCategory.SENIOR] += 0.5
            
            # Voice quality based scoring (aging indicators)
            quality = acoustic_analysis.get('quality', {})
            jitter = quality.get('jitter', 0.01)
            shimmer = quality.get('shimmer', 0.1)
            hnr = quality.get('hnr', 20.0)
            
            aging_score = jitter + shimmer - (hnr / 50.0)  # Higher = more aging
            
            if aging_score > 0.3:
                age_scores[AgeCategory.SENIOR] += 2.0
                age_scores[AgeCategory.MIDDLE_AGED] += 1.0
            elif aging_score > 0.1:
                age_scores[AgeCategory.MIDDLE_AGED] += 2.0
                age_scores[AgeCategory.YOUNG_ADULT] += 1.0
            else:
                age_scores[AgeCategory.YOUNG_ADULT] += 1.5
                age_scores[AgeCategory.TEEN] += 1.0
                age_scores[AgeCategory.CHILD] += 0.5
            
            # Spectral features
            spectral = acoustic_analysis.get('spectral', {})
            high_freq_ratio = spectral.get('high_freq_ratio', 0.5)
            
            if high_freq_ratio > 0.7:  # Higher frequencies = younger
                age_scores[AgeCategory.CHILD] += 1.0
                age_scores[AgeCategory.TEEN] += 0.5
            elif high_freq_ratio < 0.3:  # Lower frequencies = older
                age_scores[AgeCategory.SENIOR] += 1.0
                age_scores[AgeCategory.MIDDLE_AGED] += 0.5
            
            # ML classification (if available and trained)
            if self.classifier and hasattr(self.classifier, 'predict_proba'):
                try:
                    ml_scores = await self._get_ml_prediction(acoustic_analysis)
                    # Weight ML scores
                    for category, score in ml_scores.items():
                        if category in age_scores:
                            age_scores[category] += score * 0.5  # Reduced weight for untrained model
                except Exception as e:
                    logger.warning(f"ML prediction hatası: {e}")
            
            # En yüksek skorlu kategoriyi seç
            best_category = max(age_scores.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Age classification scores: {age_scores}")
            logger.info(f"Selected age category: {best_category.value}")
            
            return best_category
            
        except Exception as e:
            logger.error(f"Age classification hatası: {e}")
            return AgeCategory.YOUNG_ADULT  # Default fallback

    async def _get_ml_prediction(self, acoustic_analysis: Dict[str, Any]) -> Dict[AgeCategory, float]:
        """ML model prediction (eğer model mevcutsa)"""
        try:
            # Feature vector oluştur
            features = []
            
            # F0 features
            f0_data = acoustic_analysis.get('f0', {})
            features.extend([
                f0_data.get('mean', 150.0),
                f0_data.get('std', 20.0),
                f0_data.get('range', 100.0)
            ])
            
            # Formant features
            formant_data = acoustic_analysis.get('formants', {})
            features.extend([
                formant_data.get('f1_mean', 500.0),
                formant_data.get('f2_mean', 1500.0),
                formant_data.get('formant_dispersion', 1000.0)
            ])
            
            # Spectral features
            spectral_data = acoustic_analysis.get('spectral', {})
            features.extend([
                spectral_data.get('centroid_mean', 2000.0),
                spectral_data.get('rolloff_mean', 4000.0),
                spectral_data.get('high_freq_ratio', 0.5)
            ])
            
            # Quality features
            quality_data = acoustic_analysis.get('quality', {})
            features.extend([
                quality_data.get('jitter', 0.01),
                quality_data.get('shimmer', 0.1),
                quality_data.get('hnr', 20.0)
            ])
            
            # Normalize features
            feature_array = np.array(features).reshape(1, -1)
            if self.scaler:
                try:
                    feature_array = self.scaler.transform(feature_array)
                except:
                    pass  # Scaler not fitted, use raw features
            
            # Predict
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(feature_array)[0]
                
                # Map to age categories (assuming order: CHILD, TEEN, YOUNG_ADULT, MIDDLE_AGED, SENIOR)
                categories = list(AgeCategory)
                ml_scores = {}
                for i, category in enumerate(categories):
                    ml_scores[category] = float(probabilities[i]) if i < len(probabilities) else 0.0
                
                return ml_scores
            
        except Exception as e:
            logger.warning(f"ML prediction hatası: {e}")
        
        return {}

    async def _calculate_confidence(self, acoustic_analysis: Dict[str, Any]) -> float:
        """Multi-metric confidence hesaplama"""
        try:
            confidence_components = {}
            
            # F0 consistency
            f0_data = acoustic_analysis.get('f0', {})
            f0_consistency = f0_data.get('consistency', 0.5)
            confidence_components['f0_consistency'] = f0_consistency
            
            # Formant reliability
            formant_data = acoustic_analysis.get('formants', {})
            f1_mean = formant_data.get('f1_mean', 0)
            f2_mean = formant_data.get('f2_mean', 0)
            formant_reliability = 1.0 if (f1_mean > 0 and f2_mean > 0) else 0.3
            confidence_components['formant_reliability'] = formant_reliability
            
            # Spectral stability
            spectral_data = acoustic_analysis.get('spectral', {})
            centroid_std = spectral_data.get('centroid_std', 1000.0)
            spectral_stability = max(0.0, 1.0 - (centroid_std / 2000.0))  # Normalized
            confidence_components['spectral_stability'] = spectral_stability
            
            # Voice quality consistency
            quality_data = acoustic_analysis.get('quality', {})
            voice_stability = quality_data.get('voice_stability', 0.5)
            confidence_components['voice_quality'] = voice_stability
            
            # ML confidence (if available)
            ml_confidence = 0.6  # Default moderate confidence for heuristic approach
            confidence_components['ml_confidence'] = ml_confidence
            
            # Weighted average
            total_confidence = 0.0
            for component, weight in self.confidence_weights.items():
                component_confidence = confidence_components.get(component, 0.5)
                total_confidence += component_confidence * weight
            
            # Audio quality bonus/penalty
            additional_data = acoustic_analysis.get('additional', {})
            duration = additional_data.get('duration', 0)
            
            # Duration penalty for very short audio
            if duration < 1.0:
                total_confidence *= 0.7
            elif duration < 0.5:
                total_confidence *= 0.5
            
            # Clamp to [0, 1]
            final_confidence = np.clip(total_confidence, 0.0, 1.0)
            
            logger.info(f"Confidence components: {confidence_components}")
            logger.info(f"Final confidence: {final_confidence:.3f}")
            
            return float(final_confidence)
            
        except Exception as e:
            logger.error(f"Confidence calculation hatası: {e}")
            return 0.5

    def _get_age_range(self, category: AgeCategory) -> Tuple[int, int]:
        """Yaş kategorisinden sayısal yaş aralığı"""
        age_ranges = {
            AgeCategory.CHILD: (5, 12),
            AgeCategory.TEEN: (13, 19),
            AgeCategory.YOUNG_ADULT: (20, 35),
            AgeCategory.MIDDLE_AGED: (36, 55),
            AgeCategory.SENIOR: (56, 80)
        }
        return age_ranges.get(category, (20, 35))

    def _create_analysis_details(
        self, 
        acoustic_analysis: Dict[str, Any], 
        age_category: AgeCategory
    ) -> List[AnalysisDetail]:
        """Analiz detaylarını oluşturur"""
        details = []
        
        try:
            # F0 analysis detail
            f0_data = acoustic_analysis.get('f0', {})
            if f0_data:
                f0_mean = f0_data.get('mean', 0)
                details.append(AnalysisDetail(
                    feature="fundamental_frequency",
                    value=f0_mean,
                    description=f"Ortalama F0: {f0_mean:.1f} Hz",
                    confidence=f0_data.get('consistency', 0.5)
                ))
            
            # Formant analysis detail
            formant_data = acoustic_analysis.get('formants', {})
            if formant_data:
                f1_mean = formant_data.get('f1_mean', 0)
                f2_mean = formant_data.get('f2_mean', 0)
                details.append(AnalysisDetail(
                    feature="formant_frequencies",
                    value=f"{f1_mean:.0f}, {f2_mean:.0f} Hz",
                    description=f"F1: {f1_mean:.0f} Hz, F2: {f2_mean:.0f} Hz",
                    confidence=0.8 if f1_mean > 0 and f2_mean > 0 else 0.3
                ))
            
            # Voice quality detail
            quality_data = acoustic_analysis.get('quality', {})
            if quality_data:
                jitter = quality_data.get('jitter', 0)
                shimmer = quality_data.get('shimmer', 0)
                hnr = quality_data.get('hnr', 0)
                details.append(AnalysisDetail(
                    feature="voice_quality",
                    value=f"Jitter: {jitter:.3f}, Shimmer: {shimmer:.3f}, HNR: {hnr:.1f}",
                    description=f"Ses kalitesi metrikleri (yaşlanma göstergeleri)",
                    confidence=quality_data.get('voice_stability', 0.5)
                ))
            
            # Spectral characteristics detail
            spectral_data = acoustic_analysis.get('spectral', {})
            if spectral_data:
                centroid = spectral_data.get('centroid_mean', 0)
                high_freq_ratio = spectral_data.get('high_freq_ratio', 0)
                details.append(AnalysisDetail(
                    feature="spectral_characteristics",
                    value=f"Centroid: {centroid:.0f} Hz",
                    description=f"Spektral merkez: {centroid:.0f} Hz, Yüksek frekans oranı: {high_freq_ratio:.2f}",
                    confidence=0.7
                ))
            
            # Age category reasoning
            age_range = self._get_age_range(age_category)
            details.append(AnalysisDetail(
                feature="age_classification",
                value=age_category.value,
                description=f"Tahmini yaş aralığı: {age_range[0]}-{age_range[1]} yaş",
                confidence=0.8
            ))
            
        except Exception as e:
            logger.error(f"Analysis details oluşturma hatası: {e}")
            details.append(AnalysisDetail(
                feature="error",
                value="analysis_error",
                description=f"Detay analizi sırasında hata: {str(e)[:100]}",
                confidence=0.0
            ))
        
        return details

    def _create_error_result(self, error_message: str) -> AgeAnalysisResult:
        """Hata durumunda default result döner"""
        return AgeAnalysisResult(
            age_category=AgeCategory.YOUNG_ADULT,  # Safe default
            confidence=0.0,
            estimated_age_range=(20, 35),
            analysis_details=[
                AnalysisDetail(
                    feature="error",
                    value="analysis_failed",
                    description=f"Yaş analizi başarısız: {error_message}",
                    confidence=0.0
                )
            ],
            acoustic_features={
                'error': error_message,
                'fallback_used': True
            }
        )

    async def batch_analyze_age(
        self, 
        audio_files: List[str]
    ) -> Dict[str, AgeAnalysisResult]:
        """Birden fazla ses dosyasını paralel olarak analiz eder"""
        try:
            logger.info(f"Batch age analysis başlatılıyor: {len(audio_files)} dosya")
            
            # Concurrent analysis with semaphore
            semaphore = asyncio.Semaphore(self.settings.max_concurrent_analysis)
            
            async def analyze_single(audio_file: str) -> Tuple[str, AgeAnalysisResult]:
                async with semaphore:
                    result = await self.analyze_age(audio_file)
                    return audio_file, result
            
            # Execute all analyses concurrently
            tasks = [analyze_single(audio_file) for audio_file in audio_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            batch_results = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis hatası: {result}")
                    continue
                
                audio_file, analysis_result = result
                batch_results[audio_file] = analysis_result
            
            logger.info(f"Batch age analysis tamamlandı: {len(batch_results)}/{len(audio_files)} başarılı")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch age analysis hatası: {e}")
            return {}

    async def get_age_statistics(self, results: List[AgeAnalysisResult]) -> Dict[str, Any]:
        """Yaş analizi sonuçlarından istatistikler çıkarır"""
        try:
            if not results:
                return {}
            
            # Category distribution
            category_counts = {}
            confidence_scores = []
            age_ranges = []
            
            for result in results:
                # Category count
                category = result.age_category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Confidence scores
                confidence_scores.append(result.confidence)
                
                # Age ranges
                age_ranges.append(result.estimated_age_range)
            
            # Calculate statistics
            total_count = len(results)
            category_percentages = {
                category: (count / total_count) * 100 
                for category, count in category_counts.items()
            }
            
            # Confidence statistics
            confidence_stats = {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            }
            
            # Age range statistics
            min_ages = [age_range[0] for age_range in age_ranges]
            max_ages = [age_range[1] for age_range in age_ranges]
            
            age_stats = {
                'estimated_min_age_avg': float(np.mean(min_ages)),
                'estimated_max_age_avg': float(np.mean(max_ages)),
                'age_range_span_avg': float(np.mean([max_age - min_age for min_age, max_age in age_ranges]))
            }
            
            return {
                'total_analyses': total_count,
                'category_distribution': category_counts,
                'category_percentages': category_percentages,
                'confidence_statistics': confidence_stats,
                'age_statistics': age_stats,
                'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0],
                'average_confidence': confidence_stats['mean']
            }
            
        except Exception as e:
            logger.error(f"Age statistics hesaplama hatası: {e}")
            return {'error': str(e)}

    async def cleanup_resources(self) -> None:
        """Servisi temizler ve kaynakları serbest bırakır"""
        try:
            logger.info("Age service kaynakları temizleniyor...")
            
            # Model references'ları temizle
            self.classifier = None
            self.scaler = None
            self.feature_extractor = None
            
            # Model loaded flag'ini sıfırla
            self._model_loaded = False
            
            logger.info("Age service kaynakları temizlendi")
            
        except Exception as e:
            logger.error(f"Age service cleanup hatası: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """Servis bilgilerini döner"""
        return {
            'service_name': 'SimpleAgeService',
            'version': '1.0.0',
            'description': 'Advanced age classification using acoustic features',
            'supported_categories': [category.value for category in AgeCategory],
            'features': [
                'F0 (Fundamental Frequency) analysis',
                'Formant frequency extraction',
                'Spectral feature analysis', 
                'Voice quality metrics (jitter, shimmer, HNR)',
                'ML classification with heuristic fallback',
                'Multi-metric confidence scoring',
                'Batch processing support',
                'Caching with TTL',
                'Async processing'
            ],
            'model_loaded': self._model_loaded,
            'cache_enabled': True,
            'batch_processing': True,
            'age_thresholds': self.age_thresholds,
            'confidence_weights': self.confidence_weights
        }


# Singleton instance
_age_service_instance = None

async def get_age_service() -> SimpleAgeService:
    """Age service singleton instance'ını döner"""
    global _age_service_instance
    if _age_service_instance is None:
        _age_service_instance = SimpleAgeService()
    return _age_service_instance