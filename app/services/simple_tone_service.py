"""
Advanced Tone and Speaking Style Classification Service

Bu servis, prosodic patterns, energy dynamics ve speaking characteristics
kullanarak konuşma tonunu ve stilini tespit eder.

Features:
- Prosodic analysis: tempo, rhythm, stress patterns, intonation
- Energy dynamics: RMS energy, dynamic range, intensity variations
- Pitch dynamics: F0 contour, pitch range, intonation patterns
- Speaking rate: syllable rate, pause patterns, articulation speed
- ML + Heuristic tone classification
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
from scipy import signal
from scipy.stats import skew, kurtosis

from app.models.schemas import (
    ToneCategory, ToneAnalysisResult, AnalysisDetail,
    AudioFeatures, ProcessingError
)
from app.config import get_settings
from app.utils.cache import cache_manager
from app.utils.feature_extract import FeatureExtractor

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class SimpleToneService:
    """
    Advanced Tone Classification Service
    
    Konuşma tonunu tespit etmek için çoklu prosodic özellik kullanır:
    - Prosodic analysis: Tempo, rhythm, stress, intonation
    - Energy dynamics: Intensity patterns, dynamic range
    - Pitch dynamics: F0 contour, pitch variation patterns
    - Speaking characteristics: Rate, articulation, pause patterns
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_extractor = None
        self.scaler = None
        self.classifier = None
        self._model_loaded = False
        self._lock = asyncio.Lock()
        
        # Tone classification thresholds (research-based)
        self.tone_thresholds = {
            'energy': {
                'energetic_min': 0.7,      # High energy threshold
                'calm_max': 0.3,           # Low energy threshold
                'dynamic_range_min': 0.4    # Minimum dynamic range for energetic
            },
            'pitch': {
                'formal_f0_std_max': 20.0,     # Low F0 variation = formal
                'energetic_f0_range_min': 80.0, # High F0 range = energetic
                'authoritative_f0_min': 90.0,   # Lower F0 = authoritative
                'pitch_slope_threshold': 0.1    # Intonation pattern indicator
            },
            'prosodic': {
                'fast_speech_rate': 6.0,       # Syllables per second
                'slow_speech_rate': 3.0,
                'formal_pause_ratio': 0.15,    # Higher pause ratio = formal
                'energetic_tempo_min': 120.0   # BPM-like tempo metric
            },
            'spectral': {
                'formal_centroid_range': (1000, 2500),  # Formal speech spectral range
                'energetic_high_freq_ratio': 0.6,       # High freq content = energetic
                'calm_spectral_rolloff_max': 3000.0     # Lower rolloff = calmer
            }
        }
        
        # Confidence calculation weights
        self.confidence_weights = {
            'energy_consistency': 0.25,
            'pitch_stability': 0.25,
            'prosodic_clarity': 0.2,
            'spectral_coherence': 0.15,
            'ml_confidence': 0.15
        }

    async def _ensure_model_loaded(self) -> None:
        """ML modelini lazy loading ile yükler"""
        if self._model_loaded:
            return
            
        async with self._lock:
            if self._model_loaded:
                return
                
            try:
                logger.info("Tone classification modelini yükleniyor...")
                
                # Feature extractor'ı başlat
                if not self.feature_extractor:
                    self.feature_extractor = FeatureExtractor()
                
                # ML modelini yükle veya oluştur
                model_path = Path(self.settings.models_dir) / "tone_classifier.joblib"
                scaler_path = Path(self.settings.models_dir) / "tone_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    # Pre-trained model varsa yükle
                    self.classifier = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Pre-trained tone model yüklendi")
                else:
                    # Basit model oluştur (production'da gerçek data ile train edilecek)
                    self.classifier = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.scaler = StandardScaler()
                    logger.warning("Mock tone model oluşturuldu - production'da train edilmeli")
                
                self._model_loaded = True
                logger.info("Tone classification servisi hazır")
                
            except Exception as e:
                logger.error(f"Tone model yükleme hatası: {e}")
                # Fallback: sadece heuristic rules kullan
                self._model_loaded = True

    async def analyze_tone(
        self, 
        audio_file: str, 
        features: Optional[AudioFeatures] = None
    ) -> ToneAnalysisResult:
        """
        Ses dosyasından ton analizi yapar
        
        Args:
            audio_file: Analiz edilecek ses dosyası yolu
            features: Önceden çıkarılmış özellikler (optional)
            
        Returns:
            ToneAnalysisResult: Ton analizi sonucu
        """
        try:
            # Cache kontrolü
            cache_key = f"tone_analysis:{Path(audio_file).stem}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Tone analysis cache hit: {audio_file}")
                return ToneAnalysisResult(**cached_result)

            logger.info(f"Tone analysis başlatılıyor: {audio_file}")
            await self._ensure_model_loaded()
            
            # Feature extraction
            if not features:
                if not self.feature_extractor:
                    self.feature_extractor = FeatureExtractor()
                features = await self.feature_extractor.extract_features(audio_file)
            
            # Prosodic ve ton analizi yap
            tone_analysis = await self._analyze_tone_features(audio_file, features)
            
            # Tone classification
            tone_category = await self._classify_tone(tone_analysis)
            
            # Confidence hesaplama
            confidence = await self._calculate_confidence(tone_analysis)
            
            # Speaking characteristics
            speaking_chars = self._extract_speaking_characteristics(tone_analysis)
            
            # Analysis details
            details = self._create_analysis_details(tone_analysis, tone_category)
            
            result = ToneAnalysisResult(
                tone_category=tone_category,
                confidence=confidence,
                speaking_characteristics=speaking_chars,
                analysis_details=details,
                prosodic_features=tone_analysis
            )
            
            # Cache sonucu
            await cache_manager.set(
                cache_key, 
                result.dict(), 
                ttl=self.settings.cache_ttl_seconds
            )
            
            logger.info(f"Tone analysis tamamlandı: {tone_category.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Tone analysis hatası: {e}")
            return self._create_error_result(str(e))

    async def _analyze_tone_features(
        self, 
        audio_file: str, 
        features: AudioFeatures
    ) -> Dict[str, Any]:
        """Ton tespiti için prosodic özellikler analiz eder"""
        try:
            # Audio dosyasını yükle
            y, sr = librosa.load(audio_file, sr=None)
            
            analysis = {}
            
            # 1. Energy Dynamics Analysis
            energy_analysis = await self._analyze_energy_dynamics(y, sr)
            analysis['energy'] = energy_analysis
            
            # 2. Pitch Dynamics Analysis
            pitch_analysis = await self._analyze_pitch_dynamics(y, sr)
            analysis['pitch'] = pitch_analysis
            
            # 3. Prosodic Pattern Analysis
            prosodic_analysis = await self._analyze_prosodic_patterns(y, sr)
            analysis['prosodic'] = prosodic_analysis
            
            # 4. Speaking Rate Analysis
            speech_rate_analysis = await self._analyze_speaking_rate(y, sr)
            analysis['speech_rate'] = speech_rate_analysis
            
            # 5. Spectral Tone Features
            spectral_tone_analysis = await self._analyze_spectral_tone_features(features)
            analysis['spectral_tone'] = spectral_tone_analysis
            
            # 6. Rhythm and Timing Analysis
            rhythm_analysis = await self._analyze_rhythm_patterns(y, sr)
            analysis['rhythm'] = rhythm_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Tone feature analysis hatası: {e}")
            return {}

    async def _analyze_energy_dynamics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Enerji dinamikleri analizi (intensity, dynamic range)"""
        try:
            # RMS energy calculation
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Energy statistics
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            energy_max = float(np.max(rms))
            energy_min = float(np.min(rms))
            
            # Dynamic range
            dynamic_range = energy_max - energy_min
            dynamic_range_norm = dynamic_range / energy_max if energy_max > 0 else 0
            
            # Energy variability (coefficient of variation)
            energy_cv = energy_std / energy_mean if energy_mean > 0 else 0
            
            # Energy contour analysis
            energy_slope = self._calculate_trend_slope(rms)
            
            # Peak detection for energy bursts
            peaks, _ = signal.find_peaks(rms, height=energy_mean + energy_std)
            energy_peak_rate = len(peaks) / (len(rms) / sr * (512/sr)) if len(rms) > 0 else 0
            
            # Energy distribution analysis
            energy_skewness = float(skew(rms))
            energy_kurtosis = float(kurtosis(rms))
            
            return {
                'mean': energy_mean,
                'std': energy_std,
                'dynamic_range': float(dynamic_range),
                'dynamic_range_normalized': float(dynamic_range_norm),
                'coefficient_variation': float(energy_cv),
                'slope': float(energy_slope),
                'peak_rate': float(energy_peak_rate),
                'skewness': energy_skewness,
                'kurtosis': energy_kurtosis,
                'energy_consistency': float(1.0 - energy_cv)  # Higher = more consistent
            }
            
        except Exception as e:
            logger.error(f"Energy dynamics analysis hatası: {e}")
            return {'mean': 0.5, 'dynamic_range_normalized': 0.5}

    async def _analyze_pitch_dynamics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Pitch dinamikleri ve intonation pattern analizi"""
        try:
            # F0 extraction
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
            
            # Extract F0 contour
            f0_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_contour.append(pitch)
            
            if not f0_contour:
                return {'mean': 150.0, 'std': 20.0, 'range': 100.0}
            
            f0_array = np.array(f0_contour)
            
            # Basic F0 statistics
            f0_mean = float(np.mean(f0_array))
            f0_std = float(np.std(f0_array))
            f0_range = float(np.max(f0_array) - np.min(f0_array))
            
            # F0 dynamics
            f0_slope = self._calculate_trend_slope(f0_array)
            f0_cv = f0_std / f0_mean if f0_mean > 0 else 0
            
            # Intonation patterns
            f0_diff = np.diff(f0_array)
            rising_tendency = float(np.mean(f0_diff > 0))  # Percentage of rising segments
            falling_tendency = float(np.mean(f0_diff < 0))  # Percentage of falling segments
            
            # Pitch reset detection (for phrase boundaries)
            pitch_resets = self._detect_pitch_resets(f0_array)
            
            # F0 contour smoothness
            f0_smoothness = self._calculate_contour_smoothness(f0_array)
            
            # Pitch register analysis
            f0_median = float(np.median(f0_array))
            high_register_ratio = float(np.mean(f0_array > f0_median + f0_std))
            low_register_ratio = float(np.mean(f0_array < f0_median - f0_std))
            
            return {
                'mean': f0_mean,
                'std': f0_std,
                'range': f0_range,
                'slope': float(f0_slope),
                'coefficient_variation': float(f0_cv),
                'rising_tendency': rising_tendency,
                'falling_tendency': falling_tendency,
                'pitch_resets': float(pitch_resets),
                'smoothness': float(f0_smoothness),
                'high_register_ratio': high_register_ratio,
                'low_register_ratio': low_register_ratio,
                'intonation_complexity': float(f0_cv + abs(rising_tendency - falling_tendency))
            }
            
        except Exception as e:
            logger.error(f"Pitch dynamics analysis hatası: {e}")
            return {'mean': 150.0, 'std': 20.0, 'range': 100.0}

    async def _analyze_prosodic_patterns(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Prosodic pattern analizi (stress, rhythm, timing)"""
        try:
            # Frame-based analysis
            frame_length = 2048
            hop_length = 512
            
            # Energy-based stress detection
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Stress pattern detection
            stress_threshold = np.mean(rms) + 0.5 * np.std(rms)
            stressed_frames = rms > stress_threshold
            stress_ratio = float(np.mean(stressed_frames))
            
            # Rhythm regularity analysis
            rhythm_regularity = self._calculate_rhythm_regularity(rms)
            
            # Pause detection and analysis
            silence_threshold = 0.01 * np.max(rms)
            silent_frames = rms < silence_threshold
            
            # Calculate pause statistics
            pause_segments = self._find_continuous_segments(silent_frames)
            pause_durations = [(end - start) * (hop_length / sr) for start, end in pause_segments]
            
            if pause_durations:
                pause_ratio = float(sum(pause_durations) / (len(y) / sr))
                average_pause_duration = float(np.mean(pause_durations))
                pause_frequency = float(len(pause_durations) / (len(y) / sr))  # pauses per second
            else:
                pause_ratio = 0.0
                average_pause_duration = 0.0
                pause_frequency = 0.0
            
            # Speech tempo estimation
            speech_segments = self._find_continuous_segments(~silent_frames)
            if speech_segments:
                speech_duration = sum([(end - start) * (hop_length / sr) for start, end in speech_segments])
                speech_density = float(speech_duration / (len(y) / sr))
            else:
                speech_density = 0.5
            
            return {
                'stress_ratio': stress_ratio,
                'rhythm_regularity': float(rhythm_regularity),
                'pause_ratio': pause_ratio,
                'average_pause_duration': average_pause_duration,
                'pause_frequency': pause_frequency,
                'speech_density': speech_density,
                'prosodic_complexity': float(stress_ratio * rhythm_regularity * (1 - pause_ratio))
            }
            
        except Exception as e:
            logger.error(f"Prosodic patterns analysis hatası: {e}")
            return {'stress_ratio': 0.5, 'rhythm_regularity': 0.5}

    async def _analyze_speaking_rate(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Konuşma hızı ve artikülasyon analizi"""
        try:
            # Syllable rate estimation using zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
            
            # Simplified syllable detection
            # Bu production'da daha gelişmiş syllable detection ile değiştirilebilir
            zcr_peaks, _ = signal.find_peaks(zcr, height=np.mean(zcr))
            syllable_rate = float(len(zcr_peaks) / (len(y) / sr))
            
            # Speaking tempo categories
            if syllable_rate > self.tone_thresholds['prosodic']['fast_speech_rate']:
                tempo_category = 'fast'
            elif syllable_rate < self.tone_thresholds['prosodic']['slow_speech_rate']:
                tempo_category = 'slow'
            else:
                tempo_category = 'normal'
            
            # Articulation clarity (spectral flux based)
            stft = librosa.stft(y)
            spectral_flux = np.sum(np.diff(np.abs(stft), axis=1) ** 2, axis=0)
            articulation_clarity = float(np.mean(spectral_flux))
            
            # Speech rhythm consistency
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            if len(onset_frames) > 1:
                onset_intervals = np.diff(onset_frames) * (512 / sr)  # Convert to seconds
                rhythm_consistency = float(1.0 - (np.std(onset_intervals) / np.mean(onset_intervals)))
            else:
                rhythm_consistency = 0.5
            
            return {
                'syllable_rate': syllable_rate,
                'tempo_category': tempo_category,
                'articulation_clarity': articulation_clarity,
                'rhythm_consistency': max(0.0, rhythm_consistency),
                'speaking_fluency': float((syllable_rate / 5.0) * rhythm_consistency)  # Normalized fluency
            }
            
        except Exception as e:
            logger.error(f"Speaking rate analysis hatası: {e}")
            return {'syllable_rate': 4.0, 'tempo_category': 'normal'}

    async def _analyze_spectral_tone_features(self, features: AudioFeatures) -> Dict[str, float]:
        """Spektral özelliklerden ton karakteristikleri"""
        try:
            spectral_data = features.spectral_features
            
            # Spectral characteristics for tone
            centroid_mean = spectral_data.get('spectral_centroid_mean', 2000.0)
            rolloff_mean = spectral_data.get('spectral_rolloff_mean', 4000.0)
            bandwidth_mean = spectral_data.get('spectral_bandwidth_mean', 1500.0)
            
            # High frequency content ratio
            high_freq_ratio = centroid_mean / 4000.0  # Normalized to 0-1
            
            # Spectral shape indicators
            brightness = centroid_mean / rolloff_mean if rolloff_mean > 0 else 0.5
            spectral_spread = bandwidth_mean / centroid_mean if centroid_mean > 0 else 0.5
            
            # Tone-specific spectral features
            formal_indicator = 1.0 if (1000 <= centroid_mean <= 2500) else 0.0
            energetic_indicator = 1.0 if high_freq_ratio > 0.6 else 0.0
            calm_indicator = 1.0 if rolloff_mean < 3000.0 else 0.0
            
            return {
                'centroid_mean': centroid_mean,
                'rolloff_mean': rolloff_mean,
                'bandwidth_mean': bandwidth_mean,
                'high_freq_ratio': float(high_freq_ratio),
                'brightness': float(brightness),
                'spectral_spread': float(spectral_spread),
                'formal_indicator': formal_indicator,
                'energetic_indicator': energetic_indicator,
                'calm_indicator': calm_indicator
            }
            
        except Exception as e:
            logger.error(f"Spectral tone features analysis hatası: {e}")
            return {'centroid_mean': 2000.0, 'high_freq_ratio': 0.5}

    async def _analyze_rhythm_patterns(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Rhythm ve timing pattern analizi"""
        try:
            # Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) < 2:
                return {'rhythm_regularity': 0.5, 'tempo_estimate': 120.0}
            
            # Inter-onset intervals
            iois = np.diff(onset_times)
            
            # Rhythm regularity (lower CV = more regular)
            if len(iois) > 0 and np.mean(iois) > 0:
                rhythm_regularity = float(1.0 - (np.std(iois) / np.mean(iois)))
                rhythm_regularity = max(0.0, min(1.0, rhythm_regularity))
            else:
                rhythm_regularity = 0.5
            
            # Tempo estimation (BPM-like)
            if len(iois) > 0:
                average_ioi = np.mean(iois)
                tempo_estimate = float(60.0 / average_ioi) if average_ioi > 0 else 120.0
            else:
                tempo_estimate = 120.0
            
            # Rhythmic complexity
            if len(iois) > 1:
                rhythmic_complexity = float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else 0.5
            else:
                rhythmic_complexity = 0.5
            
            # Syncopation measure (simplified)
            syncopation = self._calculate_syncopation_measure(iois)
            
            return {
                'rhythm_regularity': rhythm_regularity,
                'tempo_estimate': tempo_estimate,
                'rhythmic_complexity': rhythmic_complexity,
                'syncopation': float(syncopation),
                'onset_density': float(len(onset_times) / (len(y) / sr))  # onsets per second
            }
            
        except Exception as e:
            logger.error(f"Rhythm patterns analysis hatası: {e}")
            return {'rhythm_regularity': 0.5, 'tempo_estimate': 120.0}

    def _calculate_trend_slope(self, data: np.ndarray) -> float:
        """Veri serisinin genel eğilimi (slope) hesaplar"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return float(slope)

    def _detect_pitch_resets(self, f0_contour: np.ndarray) -> int:
        """Pitch reset noktalarını tespit eder (phrase boundaries)"""
        if len(f0_contour) < 3:
            return 0
        
        # Large sudden changes in F0 (simplified approach)
        f0_diff = np.abs(np.diff(f0_contour))
        threshold = np.mean(f0_diff) + 2 * np.std(f0_diff)
        resets = np.sum(f0_diff > threshold)
        
        return int(resets)

    def _calculate_contour_smoothness(self, contour: np.ndarray) -> float:
        """Kontur düzgünlüğü hesaplar"""
        if len(contour) < 3:
            return 0.5
        
        # Second derivative for smoothness
        second_derivative = np.diff(contour, 2)
        smoothness = 1.0 / (1.0 + np.std(second_derivative))
        
        return float(np.clip(smoothness, 0.0, 1.0))

    def _calculate_rhythm_regularity(self, energy_contour: np.ndarray) -> float:
        """Rhythm düzenlilik ölçüsü"""
        if len(energy_contour) < 4:
            return 0.5
        
        # Autocorrelation based rhythm regularity
        autocorr = np.correlate(energy_contour, energy_contour, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
        
        if len(peaks) > 0:
            # Regularity based on peak consistency
            peak_intervals = np.diff(peaks)
            if len(peak_intervals) > 1:
                regularity = 1.0 - (np.std(peak_intervals) / np.mean(peak_intervals))
                return float(np.clip(regularity, 0.0, 1.0))
        
        return 0.5

    def _find_continuous_segments(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        """Binary array'de continuous segment'ler bulur"""
        segments = []
        start = None
        
        for i, value in enumerate(binary_array):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(binary_array)))
        
        return segments

    def _calculate_syncopation_measure(self, intervals: np.ndarray) -> float:
        """Syncopation ölçüsü (rhythm complexity)"""
        if len(intervals) < 3:
            return 0.0
        
        # Simplified syncopation: deviation from regular pattern
        expected_interval = np.mean(intervals)
        deviations = np.abs(intervals - expected_interval)
        syncopation = np.mean(deviations) / expected_interval if expected_interval > 0 else 0
        
        return float(np.clip(syncopation, 0.0, 1.0))

    async def _classify_tone(self, tone_analysis: Dict[str, Any]) -> ToneCategory:
        """Ton analiz sonuçlarından ton kategorisi belirler"""
        try:
            # Heuristic rules based classification
            tone_scores = {
                ToneCategory.FORMAL: 0.0,
                ToneCategory.CASUAL: 0.0,
                ToneCategory.ENERGETIC: 0.0,
                ToneCategory.CALM: 0.0,
                ToneCategory.AUTHORITATIVE: 0.0
            }
            
            # Energy-based scoring
            energy_data = tone_analysis.get('energy', {})
            energy_mean = energy_data.get('mean', 0.5)
            dynamic_range = energy_data.get('dynamic_range_normalized', 0.5)
            energy_consistency = energy_data.get('energy_consistency', 0.5)
            
            if energy_mean > self.tone_thresholds['energy']['energetic_min']:
                tone_scores[ToneCategory.ENERGETIC] += 3.0
            elif energy_mean < self.tone_thresholds['energy']['calm_max']:
                tone_scores[ToneCategory.CALM] += 2.5
                tone_scores[ToneCategory.FORMAL] += 1.0
            
            if dynamic_range > self.tone_thresholds['energy']['dynamic_range_min']:
                tone_scores[ToneCategory.ENERGETIC] += 2.0
                tone_scores[ToneCategory.CASUAL] += 1.0
            
            if energy_consistency > 0.7:
                tone_scores[ToneCategory.FORMAL] += 1.5
                tone_scores[ToneCategory.AUTHORITATIVE] += 1.0
            
            # Pitch-based scoring  
            pitch_data = tone_analysis.get('pitch', {})
            f0_std = pitch_data.get('std', 20.0)
            f0_range = pitch_data.get('range', 100.0)
            f0_mean = pitch_data.get('mean', 150.0)
            
            if f0_std < self.tone_thresholds['pitch']['formal_f0_std_max']:
                tone_scores[ToneCategory.FORMAL] += 2.0
                tone_scores[ToneCategory.AUTHORITATIVE] += 1.5
            
            if f0_range > self.tone_thresholds['pitch']['energetic_f0_range_min']:
                tone_scores[ToneCategory.ENERGETIC] += 2.5
                tone_scores[ToneCategory.CASUAL] += 1.0
            
            if f0_mean < self.tone_thresholds['pitch']['authoritative_f0_min']:
                tone_scores[ToneCategory.AUTHORITATIVE] += 2.0
                tone_scores[ToneCategory.FORMAL] += 1.0
            
            # Prosodic pattern scoring
            prosodic_data = tone_analysis.get('prosodic', {})
            pause_ratio = prosodic_data.get('pause_ratio', 0.1)
            stress_ratio = prosodic_data.get('stress_ratio', 0.5)
            speech_density = prosodic_data.get('speech_density', 0.8)
            
            if pause_ratio > self.tone_thresholds['prosodic']['formal_pause_ratio']:
                tone_scores[ToneCategory.FORMAL] += 2.0
                tone_scores[ToneCategory.CALM] += 1.5
            else:
                tone_scores[ToneCategory.ENERGETIC] += 1.0
                tone_scores[ToneCategory.CASUAL] += 1.5
            
            if stress_ratio > 0.6:
                tone_scores[ToneCategory.ENERGETIC] += 1.5
                tone_scores[ToneCategory.AUTHORITATIVE] += 1.0
            
            if speech_density > 0.85:
                tone_scores[ToneCategory.ENERGETIC] += 1.0
                tone_scores[ToneCategory.CASUAL] += 0.5
            elif speech_density < 0.6:
                tone_scores[ToneCategory.CALM] += 1.5
                tone_scores[ToneCategory.FORMAL] += 1.0
            
            # Speaking rate scoring
            speech_rate_data = tone_analysis.get('speech_rate', {})
            syllable_rate = speech_rate_data.get('syllable_rate', 4.0)
            tempo_category = speech_rate_data.get('tempo_category', 'normal')
            
            if tempo_category == 'fast':
                tone_scores[ToneCategory.ENERGETIC] += 2.0
                tone_scores[ToneCategory.CASUAL] += 1.0
            elif tempo_category == 'slow':
                tone_scores[ToneCategory.CALM] += 2.0
                tone_scores[ToneCategory.FORMAL] += 1.5
            
            # Spectral tone features
            spectral_tone_data = tone_analysis.get('spectral_tone', {})
            formal_indicator = spectral_tone_data.get('formal_indicator', 0.0)
            energetic_indicator = spectral_tone_data.get('energetic_indicator', 0.0)
            calm_indicator = spectral_tone_data.get('calm_indicator', 0.0)
            
            tone_scores[ToneCategory.FORMAL] += formal_indicator * 1.5
            tone_scores[ToneCategory.ENERGETIC] += energetic_indicator * 1.5
            tone_scores[ToneCategory.CALM] += calm_indicator * 1.5
            
            # Rhythm pattern scoring
            rhythm_data = tone_analysis.get('rhythm', {})
            rhythm_regularity = rhythm_data.get('rhythm_regularity', 0.5)
            tempo_estimate = rhythm_data.get('tempo_estimate', 120.0)
            
            if rhythm_regularity > 0.7:
                tone_scores[ToneCategory.FORMAL] += 1.5
                tone_scores[ToneCategory.AUTHORITATIVE] += 1.0
            else:
                tone_scores[ToneCategory.CASUAL] += 1.0
            
            if tempo_estimate > 140.0:
                tone_scores[ToneCategory.ENERGETIC] += 1.0
            elif tempo_estimate < 100.0:
                tone_scores[ToneCategory.CALM] += 1.0
            
            # ML classification (if available and trained)
            if self.classifier and hasattr(self.classifier, 'predict_proba'):
                try:
                    ml_scores = await self._get_ml_prediction(tone_analysis)
                    # Weight ML scores
                    for category, score in ml_scores.items():
                        if category in tone_scores:
                            tone_scores[category] += score * 0.3  # Reduced weight for untrained model
                except Exception as e:
                    logger.warning(f"ML prediction hatası: {e}")
            
            # Normalization and final selection
            max_score = max(tone_scores.values()) if tone_scores.values() else 1.0
            if max_score > 0:
                for category in tone_scores:
                    tone_scores[category] /= max_score
            
            # En yüksek skorlu kategoriyi seç
            best_category = max(tone_scores.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Tone classification scores: {tone_scores}")
            logger.info(f"Selected tone category: {best_category.value}")
            
            return best_category
            
        except Exception as e:
            logger.error(f"Tone classification hatası: {e}")
            return ToneCategory.CASUAL  # Safe default fallback

    async def _get_ml_prediction(self, tone_analysis: Dict[str, Any]) -> Dict[ToneCategory, float]:
        """ML model prediction (eğer model mevcutsa)"""
        try:
            # Feature vector oluştur
            features = []
            
            # Energy features
            energy_data = tone_analysis.get('energy', {})
            features.extend([
                energy_data.get('mean', 0.5),
                energy_data.get('dynamic_range_normalized', 0.5),
                energy_data.get('energy_consistency', 0.5)
            ])
            
            # Pitch features
            pitch_data = tone_analysis.get('pitch', {})
            features.extend([
                pitch_data.get('mean', 150.0) / 300.0,  # Normalized
                pitch_data.get('std', 20.0) / 100.0,    # Normalized
                pitch_data.get('range', 100.0) / 300.0  # Normalized
            ])
            
            # Prosodic features
            prosodic_data = tone_analysis.get('prosodic', {})
            features.extend([
                prosodic_data.get('pause_ratio', 0.1),
                prosodic_data.get('stress_ratio', 0.5),
                prosodic_data.get('speech_density', 0.8)
            ])
            
            # Speaking rate features
            speech_rate_data = tone_analysis.get('speech_rate', {})
            features.extend([
                speech_rate_data.get('syllable_rate', 4.0) / 10.0,  # Normalized
                speech_rate_data.get('articulation_clarity', 0.5)
            ])
            
            # Spectral tone features
            spectral_data = tone_analysis.get('spectral_tone', {})
            features.extend([
                spectral_data.get('high_freq_ratio', 0.5),
                spectral_data.get('brightness', 0.5)
            ])
            
            # Rhythm features
            rhythm_data = tone_analysis.get('rhythm', {})
            features.extend([
                rhythm_data.get('rhythm_regularity', 0.5),
                rhythm_data.get('tempo_estimate', 120.0) / 200.0  # Normalized
            ])
            
            # Normalize features
            feature_array = np.array(features).reshape(1, -1)
            if self.scaler and hasattr(self.scaler, 'transform'):
                try:
                    feature_array = self.scaler.transform(feature_array)
                except:
                    pass  # Scaler not fitted, use raw features
            
            # Predict
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(feature_array)[0]
                
                # Map to tone categories
                categories = list(ToneCategory)
                ml_scores = {}
                for i, category in enumerate(categories):
                    ml_scores[category] = float(probabilities[i]) if i < len(probabilities) else 0.0
                
                return ml_scores
            
        except Exception as e:
            logger.warning(f"ML prediction hatası: {e}")
        
        return {}

    async def _calculate_confidence(self, tone_analysis: Dict[str, Any]) -> float:
        """Multi-metric confidence hesaplama"""
        try:
            confidence_components = {}
            
            # Energy consistency
            energy_data = tone_analysis.get('energy', {})
            energy_consistency = energy_data.get('energy_consistency', 0.5)
            confidence_components['energy_consistency'] = energy_consistency
            
            # Pitch stability
            pitch_data = tone_analysis.get('pitch', {})
            pitch_smoothness = pitch_data.get('smoothness', 0.5)
            confidence_components['pitch_stability'] = pitch_smoothness
            
            # Prosodic clarity
            prosodic_data = tone_analysis.get('prosodic', {})
            prosodic_complexity = prosodic_data.get('prosodic_complexity', 0.5)
            confidence_components['prosodic_clarity'] = prosodic_complexity
            
            # Spectral coherence
            spectral_data = tone_analysis.get('spectral_tone', {})
            brightness = spectral_data.get('brightness', 0.5)
            spectral_coherence = min(1.0, brightness + 0.3)  # Adjusted coherence metric
            confidence_components['spectral_coherence'] = spectral_coherence
            
            # ML confidence (default moderate for heuristic approach)
            ml_confidence = 0.7  # Higher than age service due to clearer tone patterns
            confidence_components['ml_confidence'] = ml_confidence
            
            # Weighted average
            total_confidence = 0.0
            for component, weight in self.confidence_weights.items():
                component_confidence = confidence_components.get(component, 0.5)
                total_confidence += component_confidence * weight
            
            # Audio quality adjustments
            speech_rate_data = tone_analysis.get('speech_rate', {})
            rhythm_consistency = speech_rate_data.get('rhythm_consistency', 0.5)
            
            # Bonus for clear rhythm patterns
            if rhythm_consistency > 0.7:
                total_confidence += 0.1
            
            # Duration consideration
            energy_data = tone_analysis.get('energy', {})
            if 'sample_count' in energy_data:
                # Adjust for very short samples
                if energy_data['sample_count'] < 22050:  # < 1 second at 22kHz
                    total_confidence *= 0.8
            
            # Clamp to [0, 1]
            final_confidence = np.clip(total_confidence, 0.0, 1.0)
            
            logger.info(f"Tone confidence components: {confidence_components}")
            logger.info(f"Final tone confidence: {final_confidence:.3f}")
            
            return float(final_confidence)
            
        except Exception as e:
            logger.error(f"Tone confidence calculation hatası: {e}")
            return 0.6  # Default moderate confidence

    def _extract_speaking_characteristics(self, tone_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Konuşma karakteristiklerini çıkarır"""
        try:
            characteristics = {}
            
            # Speaking rate characteristics
            speech_rate_data = tone_analysis.get('speech_rate', {})
            characteristics['speaking_rate'] = speech_rate_data.get('syllable_rate', 4.0)
            characteristics['tempo_category'] = speech_rate_data.get('tempo_category', 'normal')
            characteristics['articulation_clarity'] = speech_rate_data.get('articulation_clarity', 0.5)
            
            # Energy characteristics
            energy_data = tone_analysis.get('energy', {})
            characteristics['energy_level'] = energy_data.get('mean', 0.5)
            characteristics['dynamic_range'] = energy_data.get('dynamic_range_normalized', 0.5)
            
            # Prosodic characteristics
            prosodic_data = tone_analysis.get('prosodic', {})
            characteristics['pause_pattern'] = prosodic_data.get('pause_ratio', 0.1)
            characteristics['stress_pattern'] = prosodic_data.get('stress_ratio', 0.5)
            
            # Pitch characteristics
            pitch_data = tone_analysis.get('pitch', {})
            characteristics['pitch_variation'] = pitch_data.get('coefficient_variation', 0.2)
            characteristics['intonation_complexity'] = pitch_data.get('intonation_complexity', 0.5)
            
            # Rhythm characteristics
            rhythm_data = tone_analysis.get('rhythm', {})
            characteristics['rhythm_regularity'] = rhythm_data.get('rhythm_regularity', 0.5)
            characteristics['tempo_estimate'] = rhythm_data.get('tempo_estimate', 120.0)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Speaking characteristics extraction hatası: {e}")
            return {'speaking_rate': 4.0, 'energy_level': 0.5}

    def _create_analysis_details(
        self, 
        tone_analysis: Dict[str, Any], 
        tone_category: ToneCategory
    ) -> List[AnalysisDetail]:
        """Analiz detaylarını oluşturur"""
        details = []
        
        try:
            # Energy analysis detail
            energy_data = tone_analysis.get('energy', {})
            if energy_data:
                energy_level = energy_data.get('mean', 0.5)
                dynamic_range = energy_data.get('dynamic_range_normalized', 0.5)
                details.append(AnalysisDetail(
                    feature="energy_dynamics",
                    value=f"Level: {energy_level:.2f}, Range: {dynamic_range:.2f}",
                    description=f"Enerji seviyesi: {energy_level:.2f}, Dinamik aralık: {dynamic_range:.2f}",
                    confidence=energy_data.get('energy_consistency', 0.7)
                ))
            
            # Pitch dynamics detail
            pitch_data = tone_analysis.get('pitch', {})
            if pitch_data:
                f0_mean = pitch_data.get('mean', 150.0)
                f0_range = pitch_data.get('range', 100.0)
                details.append(AnalysisDetail(
                    feature="pitch_dynamics",
                    value=f"F0: {f0_mean:.1f} Hz, Range: {f0_range:.1f} Hz",
                    description=f"Ortalama F0: {f0_mean:.1f} Hz, Pitch aralığı: {f0_range:.1f} Hz",
                    confidence=pitch_data.get('smoothness', 0.7)
                ))
            
            # Speaking rate detail
            speech_rate_data = tone_analysis.get('speech_rate', {})
            if speech_rate_data:
                syllable_rate = speech_rate_data.get('syllable_rate', 4.0)
                tempo_category = speech_rate_data.get('tempo_category', 'normal')
                details.append(AnalysisDetail(
                    feature="speaking_rate",
                    value=f"{syllable_rate:.1f} syllables/sec ({tempo_category})",
                    description=f"Konuşma hızı: {syllable_rate:.1f} hece/saniye ({tempo_category})",
                    confidence=speech_rate_data.get('rhythm_consistency', 0.7)
                ))
            
            # Prosodic pattern detail
            prosodic_data = tone_analysis.get('prosodic', {})
            if prosodic_data:
                pause_ratio = prosodic_data.get('pause_ratio', 0.1)
                stress_ratio = prosodic_data.get('stress_ratio', 0.5)
                details.append(AnalysisDetail(
                    feature="prosodic_patterns",
                    value=f"Pauses: {pause_ratio:.2f}, Stress: {stress_ratio:.2f}",
                    description=f"Duraklama oranı: {pause_ratio:.2f}, Vurgu oranı: {stress_ratio:.2f}",
                    confidence=prosodic_data.get('rhythm_regularity', 0.7)
                ))
            
            # Tone classification result
            details.append(AnalysisDetail(
                feature="tone_classification",
                value=tone_category.value,
                description=f"Tespit edilen ton kategorisi: {tone_category.value}",
                confidence=0.8
            ))
            
        except Exception as e:
            logger.error(f"Tone analysis details oluşturma hatası: {e}")
            details.append(AnalysisDetail(
                feature="error",
                value="analysis_error",
                description=f"Detay analizi sırasında hata: {str(e)[:100]}",
                confidence=0.0
            ))
        
        return details

    def _create_error_result(self, error_message: str) -> ToneAnalysisResult:
        """Hata durumunda default result döner"""
        return ToneAnalysisResult(
            tone_category=ToneCategory.CASUAL,  # Safe default
            confidence=0.0,
            speaking_characteristics={
                'speaking_rate': 4.0,
                'energy_level': 0.5,
                'error': error_message
            },
            analysis_details=[
                AnalysisDetail(
                    feature="error",
                    value="analysis_failed",
                    description=f"Ton analizi başarısız: {error_message}",
                    confidence=0.0
                )
            ],
            prosodic_features={
                'error': error_message,
                'fallback_used': True
            }
        )

    async def batch_analyze_tone(
        self, 
        audio_files: List[str]
    ) -> Dict[str, ToneAnalysisResult]:
        """Birden fazla ses dosyasını paralel olarak analiz eder"""
        try:
            logger.info(f"Batch tone analysis başlatılıyor: {len(audio_files)} dosya")
            
            # Concurrent analysis with semaphore
            semaphore = asyncio.Semaphore(self.settings.max_concurrent_analysis)
            
            async def analyze_single(audio_file: str) -> Tuple[str, ToneAnalysisResult]:
                async with semaphore:
                    result = await self.analyze_tone(audio_file)
                    return audio_file, result
            
            # Execute all analyses concurrently
            tasks = [analyze_single(audio_file) for audio_file in audio_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            batch_results = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch tone analysis hatası: {result}")
                    continue
                
                audio_file, analysis_result = result
                batch_results[audio_file] = analysis_result
            
            logger.info(f"Batch tone analysis tamamlandı: {len(batch_results)}/{len(audio_files)} başarılı")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch tone analysis hatası: {e}")
            return {}

    async def get_tone_statistics(self, results: List[ToneAnalysisResult]) -> Dict[str, Any]:
        """Ton analizi sonuçlarından istatistikler çıkarır"""
        try:
            if not results:
                return {}
            
            # Category distribution
            category_counts = {}
            confidence_scores = []
            speaking_rates = []
            energy_levels = []
            
            for result in results:
                # Category count
                category = result.tone_category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Confidence scores
                confidence_scores.append(result.confidence)
                
                # Speaking characteristics
                characteristics = result.speaking_characteristics
                if isinstance(characteristics, dict):
                    speaking_rates.append(characteristics.get('speaking_rate', 4.0))
                    energy_levels.append(characteristics.get('energy_level', 0.5))
            
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
            
            # Speaking characteristics statistics
            if speaking_rates:
                speaking_stats = {
                    'average_speaking_rate': float(np.mean(speaking_rates)),
                    'speaking_rate_std': float(np.std(speaking_rates)),
                    'fast_speakers_ratio': float(np.mean([rate > 5.5 for rate in speaking_rates]))
                }
            else:
                speaking_stats = {}
            
            if energy_levels:
                energy_stats = {
                    'average_energy_level': float(np.mean(energy_levels)),
                    'energy_variation': float(np.std(energy_levels)),
                    'high_energy_ratio': float(np.mean([level > 0.7 for level in energy_levels]))
                }
            else:
                energy_stats = {}
            
            return {
                'total_analyses': total_count,
                'category_distribution': category_counts,
                'category_percentages': category_percentages,
                'confidence_statistics': confidence_stats,
                'speaking_characteristics_stats': speaking_stats,
                'energy_statistics': energy_stats,
                'most_common_tone': max(category_counts.items(), key=lambda x: x[1])[0],
                'average_confidence': confidence_stats['mean']
            }
            
        except Exception as e:
            logger.error(f"Tone statistics hesaplama hatası: {e}")
            return {'error': str(e)}

    async def cleanup_resources(self) -> None:
        """Servisi temizler ve kaynakları serbest bırakır"""
        try:
            logger.info("Tone service kaynakları temizleniyor...")
            
            # Model references'ları temizle
            self.classifier = None
            self.scaler = None
            self.feature_extractor = None
            
            # Model loaded flag'ini sıfırla
            self._model_loaded = False
            
            logger.info("Tone service kaynakları temizlendi")
            
        except Exception as e:
            logger.error(f"Tone service cleanup hatası: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """Servis bilgilerini döner"""
        return {
            'service_name': 'SimpleToneService',
            'version': '1.0.0',
            'description': 'Advanced tone and speaking style classification',
            'supported_categories': [category.value for category in ToneCategory],
            'features': [
                'Prosodic pattern analysis (tempo, rhythm, stress)',
                'Energy dynamics analysis (RMS, dynamic range, intensity)',
                'Pitch dynamics analysis (F0 contour, intonation patterns)',
                'Speaking rate analysis (syllable rate, pause patterns)',
                'Spectral tone features (brightness, high-freq content)',
                'Rhythm and timing analysis (regularity, syncopation)',
                'ML + heuristic classification',
                'Multi-metric confidence scoring',
                'Speaking characteristics extraction',
                'Batch processing support',
                'Caching with TTL',
                'Async processing'
            ],
            'model_loaded': self._model_loaded,
            'cache_enabled': True,
            'batch_processing': True,
            'tone_thresholds': self.tone_thresholds,
            'confidence_weights': self.confidence_weights
        }


# Singleton instance
_tone_service_instance = None

async def get_tone_service() -> SimpleToneService:
    """Tone service singleton instance'ını döner"""
    global _tone_service_instance
    if _tone_service_instance is None:
        _tone_service_instance = SimpleToneService()
    return _tone_service_instance