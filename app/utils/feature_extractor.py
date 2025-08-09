"""
Ses özelliği çıkarma modülü - MFCC, spektral ve prosodik özellikler

Bu modül ses dosyalarından makine öğrenmesi modelleri için gerekli özellikleri çıkarır.
MFCC, spektral analiz ve prosodik özellikler dahil olmak üzere kapsamlı özellik seti sağlar.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import librosa
import librosa.feature
from scipy import signal
from scipy.stats import skew, kurtosis
import pandas as pd

from app.config import get_audio_config, Settings
from app.models.schemas import AudioData, FeatureVector, FeatureConfig
from app.utils.cache import cache_manager
from app.utils.audio_loader import audio_loader

logger = logging.getLogger(__name__)

class FeatureExtractionError(Exception):
    """Özellik çıkarma hatası için özel exception"""
    pass

class FeatureExtractor:
    """
    Ses özelliği çıkarma sınıfı
    
    Özellikler:
    - MFCC (Mel-Frequency Cepstral Coefficients) - ses karakteristikleri
    - Spektral özellikler (centroid, bandwidth, rolloff) - ton ve pitch analizi
    - Prosodik özellikler (rhythm, stress patterns) - konuşma dinamikleri
    - Zaman domain özellikleri - temel ses istatistikleri
    - Configurable parametreler - esnek ayarlar
    - Async processing - performanslı işleme
    """
    
    def __init__(self, config: Optional[Settings] = None):
        """FeatureExtractor'ı başlat"""
        self.config = config or get_audio_config()
        self.executor = ThreadPoolExecutor(max_workers=2)  # CPU yoğun işlemler için
        
        # Özellik çıkarma ayarları
        self.sample_rate = getattr(self.config, 'target_sample_rate', 16000)
        self.n_mfcc = getattr(self.config, 'n_mfcc', 13)
        self.n_fft = getattr(self.config, 'n_fft', 2048)
        self.hop_length = getattr(self.config, 'hop_length', 512)
        self.win_length = getattr(self.config, 'win_length', None)
        
        # Spektral analiz ayarları
        self.n_mels = getattr(self.config, 'n_mels', 128)
        self.fmin = getattr(self.config, 'fmin', 0)
        self.fmax = getattr(self.config, 'fmax', None)
        
        # Prosodik analiz ayarları
        self.frame_length = getattr(self.config, 'frame_length', 2048)
        self.hop_length_prosody = getattr(self.config, 'hop_length_prosody', 512)
        
        logger.info(f"FeatureExtractor başlatıldı - MFCC: {self.n_mfcc}, Sample Rate: {self.sample_rate}")
    
    async def extract_features(
        self, 
        audio_data: Union[AudioData, np.ndarray], 
        feature_types: Optional[List[str]] = None
    ) -> FeatureVector:
        """
        Ana özellik çıkarma metodu - tüm özellikleri çıkarır
        
        Args:
            audio_data: Ses verisi (AudioData objesi veya numpy array)
            feature_types: Çıkarılacak özellik türleri (None = hepsi)
            
        Returns:
            FeatureVector: Çıkarılan özellikler
            
        Raises:
            FeatureExtractionError: Özellik çıkarma hatası
        """
        try:
            # Audio data'yı hazırla
            if isinstance(audio_data, AudioData):
                audio_array = audio_data.audio_data
                sample_rate = audio_data.sample_rate
                file_hash = audio_data.file_hash
            else:
                audio_array = audio_data
                sample_rate = self.sample_rate
                file_hash = None
            
            # Cache kontrolü
            if file_hash:
                cache_key = f"features:{file_hash}:{hash(str(feature_types))}"
                cached_features = cache_manager.get(cache_key)
                if cached_features:
                    logger.debug("Özellikler cache'den yüklendi")
                    return cached_features
            
            # Varsayılan özellik türleri
            if feature_types is None:
                feature_types = ['mfcc', 'spectral', 'prosodic', 'temporal']
            
            logger.info(f"Özellik çıkarma başlatıldı: {feature_types}")
            
            # Özellikleri paralel olarak çıkar
            tasks = []
            
            if 'mfcc' in feature_types:
                tasks.append(self._extract_mfcc_features(audio_array, sample_rate))
            
            if 'spectral' in feature_types:
                tasks.append(self._extract_spectral_features(audio_array, sample_rate))
            
            if 'prosodic' in feature_types:
                tasks.append(self._extract_prosodic_features(audio_array, sample_rate))
            
            if 'temporal' in feature_types:
                tasks.append(self._extract_temporal_features(audio_array, sample_rate))
            
            # Tüm özellikleri bekle
            feature_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sonuçları birleştir
            combined_features = {}
            feature_names = ['mfcc', 'spectral', 'prosodic', 'temporal']
            
            for i, result in enumerate(feature_results):
                if isinstance(result, Exception):
                    logger.warning(f"{feature_names[i]} özellik çıkarma hatası: {result}")
                    continue
                
                if isinstance(result, dict):
                    combined_features.update(result)
            
            if not combined_features:
                raise FeatureExtractionError("Hiçbir özellik çıkarılamadı")
            
            # FeatureVector oluştur
            feature_vector = FeatureVector(
                features=combined_features,
                feature_types=feature_types,
                sample_rate=sample_rate,
                audio_duration=len(audio_array) / sample_rate,
                extraction_config={
                    'n_mfcc': self.n_mfcc,
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels
                }
            )
            
            # Cache'e kaydet
            if file_hash:
                cache_manager.set(cache_key, feature_vector, ttl=600)  # 10 dakika
            
            logger.info(f"Özellik çıkarma tamamlandı: {len(combined_features)} özellik")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Özellik çıkarma başarısız: {str(e)}")
            raise FeatureExtractionError(f"Özellik çıkarma hatası: {str(e)}") from e
    
    async def _extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini çıkar
        Ses karakteristikleri ve tonalite için önemli
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme oranı
            
        Returns:
            Dict: MFCC özellikleri
        """
        try:
            # MFCC'yi thread pool'da hesapla
            mfcc_features = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self._compute_mfcc_sync, 
                audio_data, 
                sample_rate
            )
            
            return mfcc_features
            
        except Exception as e:
            logger.error(f"MFCC özellik çıkarma hatası: {str(e)}")
            return {}
    
    def _compute_mfcc_sync(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """MFCC hesaplaması - senkron"""
        try:
            # MFCC katsayılarını çıkar
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            
            # Delta ve Delta-Delta (derivative) özellikler
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # İstatistiksel özellikler
            features = {}
            
            # MFCC ortalamaları
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfcc[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfcc[i]))
            
            # Delta MFCC
            for i in range(self.n_mfcc):
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
            
            # Delta-Delta MFCC
            for i in range(self.n_mfcc):
                features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
                features[f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))
            
            # Global MFCC özellikleri
            features['mfcc_mean_overall'] = float(np.mean(mfcc))
            features['mfcc_std_overall'] = float(np.std(mfcc))
            features['mfcc_skew_overall'] = float(skew(mfcc.flatten()))
            features['mfcc_kurtosis_overall'] = float(kurtosis(mfcc.flatten()))
            
            logger.debug(f"MFCC özellikleri hesaplandı: {len(features)} özellik")
            return features
            
        except Exception as e:
            logger.error(f"MFCC hesaplama hatası: {str(e)}")
            return {}
    
    async def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Spektral özellikler - ton, pitch ve frekans analizi
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme oranı
            
        Returns:
            Dict: Spektral özellikler
        """
        try:
            spectral_features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._compute_spectral_sync,
                audio_data,
                sample_rate
            )
            
            return spectral_features
            
        except Exception as e:
            logger.error(f"Spektral özellik çıkarma hatası: {str(e)}")
            return {}
    
    def _compute_spectral_sync(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Spektral özellik hesaplaması - senkron"""
        try:
            features = {}
            
            # Temel spektral özellikler
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio_data, hop_length=self.hop_length
            )[0]
            
            # Chroma özellikleri (pitch class profili)
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Mel-scale spektrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data, sr=sample_rate, n_mels=self.n_mels, 
                hop_length=self.hop_length, n_fft=self.n_fft
            )
            
            # İstatistiksel özellikler
            features.update({
                # Spektral Centroid (parlaklık göstergesi)
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_centroid_max': float(np.max(spectral_centroid)),
                'spectral_centroid_min': float(np.min(spectral_centroid)),
                
                # Spektral Bandwidth (ses genişliği)
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                
                # Spektral Rolloff (yüksek frekans içeriği)
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                
                # Zero Crossing Rate (ses/müzik ayrımı)
                'zcr_mean': float(np.mean(zero_crossing_rate)),
                'zcr_std': float(np.std(zero_crossing_rate)),
                'zcr_max': float(np.max(zero_crossing_rate)),
                'zcr_min': float(np.min(zero_crossing_rate)),
            })
            
            # Chroma özellikleri (12 pitch class)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
            
            # Mel spektrogram özellikleri
            mel_mean = np.mean(mel_spectrogram, axis=1)
            mel_std = np.std(mel_spectrogram, axis=1)
            
            # İlk 20 mel band için özellikler (boyutu kontrol et)
            n_mel_features = min(20, len(mel_mean))
            for i in range(n_mel_features):
                features[f'mel_{i}_mean'] = float(mel_mean[i])
                features[f'mel_{i}_std'] = float(mel_std[i])
            
            # Spektral kontrast
            try:
                spectral_contrast = librosa.feature.spectral_contrast(
                    y=audio_data, sr=sample_rate, hop_length=self.hop_length
                )
                for i in range(min(7, spectral_contrast.shape[0])):  # 7 band
                    features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))
            except Exception as e:
                logger.debug(f"Spektral kontrast hesaplanamadı: {e}")
            
            # Tonsal özellikler
            try:
                tonnetz = librosa.feature.tonnetz(
                    y=librosa.effects.harmonic(audio_data), sr=sample_rate
                )
                for i in range(6):  # 6 tonal koordinat
                    features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                    features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
            except Exception as e:
                logger.debug(f"Tonnetz hesaplanamadı: {e}")
            
            logger.debug(f"Spektral özellikler hesaplandı: {len(features)} özellik")
            return features
            
        except Exception as e:
            logger.error(f"Spektral hesaplama hatası: {str(e)}")
            return {}
    
    async def _extract_prosodic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Prosodik özellikler - ritim, vurgu, tempo
        Konuşma dinamikleri ve duygusal ton için önemli
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme oranı
            
        Returns:
            Dict: Prosodik özellikler
        """
        try:
            prosodic_features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._compute_prosodic_sync,
                audio_data,
                sample_rate
            )
            
            return prosodic_features
            
        except Exception as e:
            logger.error(f"Prosodik özellik çıkarma hatası: {str(e)}")
            return {}
    
    def _compute_prosodic_sync(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Prosodik özellik hesaplaması - senkron"""
        try:
            features = {}
            
            # Pitch (F0) tahmini
            try:
                # Yin algoritması ile pitch
                f0 = librosa.yin(
                    audio_data, 
                    fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                    fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                    sr=sample_rate,
                    hop_length=self.hop_length_prosody
                )
                
                # Voiced frames (pitch var olan)
                voiced_frames = f0 > 0
                voiced_f0 = f0[voiced_frames]
                
                if len(voiced_f0) > 0:
                    features.update({
                        'pitch_mean': float(np.mean(voiced_f0)),
                        'pitch_std': float(np.std(voiced_f0)),
                        'pitch_max': float(np.max(voiced_f0)),
                        'pitch_min': float(np.min(voiced_f0)),
                        'pitch_range': float(np.max(voiced_f0) - np.min(voiced_f0)),
                        'voiced_ratio': float(np.sum(voiced_frames) / len(f0))
                    })
                    
                    # Pitch variation (jitter)
                    if len(voiced_f0) > 1:
                        pitch_diff = np.diff(voiced_f0)
                        features['pitch_jitter'] = float(np.std(pitch_diff))
                        features['pitch_slope'] = float(np.polyfit(range(len(voiced_f0)), voiced_f0, 1)[0])
                else:
                    # Eğer pitch bulunamazsa varsayılan değerler
                    features.update({
                        'pitch_mean': 0.0,
                        'pitch_std': 0.0,
                        'pitch_max': 0.0,
                        'pitch_min': 0.0,
                        'pitch_range': 0.0,
                        'voiced_ratio': 0.0,
                        'pitch_jitter': 0.0,
                        'pitch_slope': 0.0
                    })
            
            except Exception as e:
                logger.debug(f"Pitch hesaplanamadı: {e}")
                # Varsayılan pitch özellikleri
                features.update({
                    'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_max': 0.0,
                    'pitch_min': 0.0, 'pitch_range': 0.0, 'voiced_ratio': 0.0,
                    'pitch_jitter': 0.0, 'pitch_slope': 0.0
                })
            
            # Tempo ve ritim
            try:
                # Beat tracking ile tempo
                tempo, beats = librosa.beat.beat_track(
                    y=audio_data, sr=sample_rate, hop_length=self.hop_length_prosody
                )
                
                features['tempo'] = float(tempo)
                
                # Beat strength
                if len(beats) > 1:
                    beat_times = librosa.frames_to_time(beats, sr=sample_rate)
                    beat_intervals = np.diff(beat_times)
                    features['tempo_stability'] = float(1.0 / (np.std(beat_intervals) + 1e-8))
                else:
                    features['tempo_stability'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Tempo hesaplanamadı: {e}")
                features['tempo'] = 0.0
                features['tempo_stability'] = 0.0
            
            # Enerji ve amplitüd değişimi
            # Frame-based energy
            frame_length = self.frame_length
            hop_length = self.hop_length_prosody
            
            frames = librosa.util.frame(audio_data, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            
            # RMS energy per frame
            rms_energy = np.sqrt(np.mean(frames**2, axis=0))
            
            features.update({
                'energy_mean': float(np.mean(rms_energy)),
                'energy_std': float(np.std(rms_energy)),
                'energy_max': float(np.max(rms_energy)),
                'energy_min': float(np.min(rms_energy)),
                'energy_range': float(np.max(rms_energy) - np.min(rms_energy))
            })
            
            # Enerji değişim oranı
            if len(rms_energy) > 1:
                energy_diff = np.diff(rms_energy)
                features['energy_variation'] = float(np.std(energy_diff))
                features['energy_slope'] = float(np.polyfit(range(len(rms_energy)), rms_energy, 1)[0])
            else:
                features['energy_variation'] = 0.0
                features['energy_slope'] = 0.0
            
            # Pause detection (sessizlik analizi)
            # Düşük enerji framelerini bul
            silence_threshold = np.percentile(rms_energy, 25)  # Alt %25
            silence_frames = rms_energy < silence_threshold
            
            features['silence_ratio'] = float(np.sum(silence_frames) / len(silence_frames))
            
            # Speaking rate (konuşma hızı tahmini)
            voiced_frames_energy = rms_energy > silence_threshold
            if np.sum(voiced_frames_energy) > 0:
                speaking_duration = np.sum(voiced_frames_energy) * hop_length / sample_rate
                features['speaking_rate'] = float(speaking_duration / (len(audio_data) / sample_rate))
            else:
                features['speaking_rate'] = 0.0
            
            logger.debug(f"Prosodik özellikler hesaplandı: {len(features)} özellik")
            return features
            
        except Exception as e:
            logger.error(f"Prosodik hesaplama hatası: {str(e)}")
            return {}
    
    async def _extract_temporal_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Zaman domain özellikleri - temel ses istatistikleri
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme oranı
            
        Returns:
            Dict: Temporal özellikler
        """
        try:
            temporal_features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._compute_temporal_sync,
                audio_data,
                sample_rate
            )
            
            return temporal_features
            
        except Exception as e:
            logger.error(f"Temporal özellik çıkarma hatası: {str(e)}")
            return {}
    
    def _compute_temporal_sync(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Temporal özellik hesaplaması - senkron"""
        try:
            features = {}
            
            # Temel istatistikler
            features.update({
                'audio_mean': float(np.mean(audio_data)),
                'audio_std': float(np.std(audio_data)),
                'audio_max': float(np.max(audio_data)),
                'audio_min': float(np.min(audio_data)),
                'audio_range': float(np.max(audio_data) - np.min(audio_data)),
                'audio_skewness': float(skew(audio_data)),
                'audio_kurtosis': float(kurtosis(audio_data))
            })
            
            # RMS ve energy
            rms = np.sqrt(np.mean(audio_data**2))
            features['rms_energy'] = float(rms)
            
            # Peak to RMS ratio (crest factor)
            if rms > 0:
                features['crest_factor'] = float(np.max(np.abs(audio_data)) / rms)
            else:
                features['crest_factor'] = 0.0
            
            # Auto-correlation analizi
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # İlk minimum (fundamental period tahmini)
            if len(autocorr) > 1:
                first_min_idx = np.argmin(autocorr[1:100]) + 1  # İlk 100 sample'da ara
                features['autocorr_first_min'] = float(first_min_idx / sample_rate)
                
                # Autocorrelation peak
                if len(autocorr) > 50:
                    features['autocorr_peak'] = float(np.max(autocorr[20:100]))
                else:
                    features['autocorr_peak'] = 0.0
            else:
                features['autocorr_first_min'] = 0.0
                features['autocorr_peak'] = 0.0
            
            # Zero-crossing rate (detaylı)
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            features['zero_crossing_count'] = float(len(zero_crossings))
            features['zero_crossing_rate_detailed'] = float(len(zero_crossings) / len(audio_data))
            
            # Envelope tracking
            analytic_signal = signal.hilbert(audio_data)
            amplitude_envelope = np.abs(analytic_signal)
            
            features.update({
                'envelope_mean': float(np.mean(amplitude_envelope)),
                'envelope_std': float(np.std(amplitude_envelope)),
                'envelope_max': float(np.max(amplitude_envelope)),
            })
            
            # Attack time (ses başlangıcına ulaşma süresi)
            # İlk %90 peak'e ulaşma süresi
            peak_val = np.max(amplitude_envelope)
            if peak_val > 0:
                threshold_90 = peak_val * 0.9
                attack_idx = np.where(amplitude_envelope >= threshold_90)[0]
                if len(attack_idx) > 0:
                    features['attack_time'] = float(attack_idx[0] / sample_rate)
                else:
                    features['attack_time'] = 0.0
            else:
                features['attack_time'] = 0.0
            
            # Decay time (ses sonlanma süresi)
            # Son %10'a düşme süresi
            if peak_val > 0:
                threshold_10 = peak_val * 0.1
                # Ters çevir ve decay bul
                reversed_envelope = amplitude_envelope[::-1]
                decay_idx = np.where(reversed_envelope >= threshold_10)[0]
                if len(decay_idx) > 0:
                    features['decay_time'] = float(decay_idx[0] / sample_rate)
                else:
                    features['decay_time'] = 0.0
            else:
                features['decay_time'] = 0.0
            
            # Sesli/sessiz segment analizi
            # Frame-based analysis
            frame_size = 1024
            hop_size = 512
            
            frames = librosa.util.frame(audio_data, frame_length=frame_size, 
                                      hop_length=hop_size, axis=0)
            frame_energies = np.mean(frames**2, axis=0)
            
            # Adaptive threshold (Otsu benzeri)
            sorted_energies = np.sort(frame_energies)
            if len(sorted_energies) > 10:
                # Median değeri threshold olarak kullan
                energy_threshold = np.median(sorted_energies)
                voiced_frames = frame_energies > energy_threshold
                
                features['voiced_frame_ratio'] = float(np.sum(voiced_frames) / len(voiced_frames))
                
                # Segment istatistikleri
                if np.sum(voiced_frames) > 0:
                    voiced_segments = []
                    current_segment = 0
                    
                    for i, is_voiced in enumerate(voiced_frames):
                        if is_voiced:
                            current_segment += 1
                        else:
                            if current_segment > 0:
                                voiced_segments.append(current_segment)
                                current_segment = 0
                    
                    if current_segment > 0:
                        voiced_segments.append(current_segment)
                    
                    if voiced_segments:
                        features['avg_voiced_segment_length'] = float(np.mean(voiced_segments) * hop_size / sample_rate)
                        features['max_voiced_segment_length'] = float(np.max(voiced_segments) * hop_size / sample_rate)
                        features['num_voiced_segments'] = float(len(voiced_segments))
                    else:
                        features['avg_voiced_segment_length'] = 0.0
                        features['max_voiced_segment_length'] = 0.0
                        features['num_voiced_segments'] = 0.0
                else:
                    features['voiced_frame_ratio'] = 0.0
                    features['avg_voiced_segment_length'] = 0.0
                    features['max_voiced_segment_length'] = 0.0
                    features['num_voiced_segments'] = 0.0
            else:
                # Çok az frame varsa varsayılan değerler
                features['voiced_frame_ratio'] = 1.0
                features['avg_voiced_segment_length'] = len(audio_data) / sample_rate
                features['max_voiced_segment_length'] = len(audio_data) / sample_rate
                features['num_voiced_segments'] = 1.0
            
            logger.debug(f"Temporal özellikler hesaplandı: {len(features)} özellik")
            return features
            
        except Exception as e:
            logger.error(f"Temporal hesaplama hatası: {str(e)}")
            return {}
    
    async def extract_custom_features(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        feature_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Özel özellik çıkarma - kullanıcı tanımlı parametrelerle
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme oranı
            feature_config: Özel özellik konfigürasyonu
            
        Returns:
            Dict: Özel özellikler
        """
        try:
            custom_features = {}
            
            # Windowing parametreleri
            window_size = feature_config.get('window_size', 2048)
            hop_length = feature_config.get('hop_length', 512)
            
            # Özel MFCC parametreleri
            if 'custom_mfcc' in feature_config:
                mfcc_config = feature_config['custom_mfcc']
                n_mfcc = mfcc_config.get('n_mfcc', 13)
                
                mfcc = librosa.feature.mfcc(
                    y=audio_data, sr=sample_rate, n_mfcc=n_mfcc,
                    n_fft=window_size, hop_length=hop_length
                )
                
                # Sadece ortalama ve standart sapma
                for i in range(n_mfcc):
                    custom_features[f'custom_mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                    custom_features[f'custom_mfcc_{i}_std'] = float(np.std(mfcc[i]))
            
            # Özel frekans bantları
            if 'frequency_bands' in feature_config:
                bands = feature_config['frequency_bands']
                
                # STFT hesapla
                stft = librosa.stft(audio_data, n_fft=window_size, hop_length=hop_length)
                magnitude = np.abs(stft)
                
                # Her band için enerji
                freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=window_size)
                
                for band_name, (low_freq, high_freq) in bands.items():
                    # Frekans indekslerini bul
                    low_idx = np.argmin(np.abs(freqs - low_freq))
                    high_idx = np.argmin(np.abs(freqs - high_freq))
                    
                    # Band enerjisi
                    band_energy = np.mean(magnitude[low_idx:high_idx+1], axis=0)
                    
                    custom_features[f'band_{band_name}_mean'] = float(np.mean(band_energy))
                    custom_features[f'band_{band_name}_std'] = float(np.std(band_energy))
                    custom_features[f'band_{band_name}_max'] = float(np.max(band_energy))
            
            # Özel istatistikler
            if 'custom_stats' in feature_config:
                stats_config = feature_config['custom_stats']
                
                if 'percentiles' in stats_config:
                    percentiles = stats_config['percentiles']
                    for p in percentiles:
                        custom_features[f'percentile_{p}'] = float(np.percentile(audio_data, p))
                
                if 'moments' in stats_config:
                    moments = stats_config['moments']
                    if 'skewness' in moments:
                        custom_features['custom_skewness'] = float(skew(audio_data))
                    if 'kurtosis' in moments:
                        custom_features['custom_kurtosis'] = float(kurtosis(audio_data))
            
            logger.info(f"Özel özellikler çıkarıldı: {len(custom_features)} özellik")
            return custom_features
            
        except Exception as e:
            logger.error(f"Özel özellik çıkarma hatası: {str(e)}")
            return {}
    
    def get_feature_summary(self, feature_vector: FeatureVector) -> Dict[str, Any]:
        """
        Özellik vektörü özeti
        
        Args:
            feature_vector: FeatureVector objesi
            
        Returns:
            Dict: Özellik özeti
        """
        try:
            features = feature_vector.features
            
            # Kategori bazında sayımlar
            categories = {
                'mfcc': len([k for k in features.keys() if 'mfcc' in k]),
                'spectral': len([k for k in features.keys() if any(x in k for x in ['spectral', 'chroma', 'mel', 'zcr', 'tonnetz'])]),
                'prosodic': len([k for k in features.keys() if any(x in k for x in ['pitch', 'tempo', 'energy', 'silence'])]),
                'temporal': len([k for k in features.keys() if any(x in k for x in ['audio_', 'rms', 'crest', 'autocorr', 'envelope'])])
            }
            
            # Değer aralıkları
            values = list(features.values())
            
            summary = {
                'total_features': len(features),
                'feature_categories': categories,
                'value_statistics': {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                },
                'extraction_config': feature_vector.extraction_config,
                'audio_duration': feature_vector.audio_duration,
                'sample_rate': feature_vector.sample_rate
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Özellik özeti oluşturma hatası: {str(e)}")
            return {}
    
    async def batch_extract_features(
        self, 
        audio_files: List[str],
        feature_types: Optional[List[str]] = None
    ) -> Dict[str, FeatureVector]:
        """
        Toplu özellik çıkarma - birden fazla dosya
        
        Args:
            audio_files: Ses dosyası yolları listesi
            feature_types: Çıkarılacak özellik türleri
            
        Returns:
            Dict: Dosya adı -> FeatureVector mapping
        """
        try:
            results = {}
            
            logger.info(f"Toplu özellik çıkarma başlatıldı: {len(audio_files)} dosya")
            
            # Her dosya için ayrı ayrı işle (paralel yapılabilir)
            for file_path in audio_files:
                try:
                    file_name = Path(file_path).name
                    
                    # Audio'yu yükle
                    audio_data, sample_rate = audio_loader.load_audio(file_path)
                    
                    # AudioData objesi oluştur
                    audio_obj = AudioData(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        metadata=None,
                        file_hash=None
                    )
                    
                    # Özellikleri çıkar
                    feature_vector = await self.extract_features(audio_obj, feature_types)
                    results[file_name] = feature_vector
                    
                    logger.info(f"Özellikler çıkarıldı: {file_name}")
                    
                except Exception as e:
                    logger.error(f"Dosya işleme hatası {file_path}: {str(e)}")
                    results[Path(file_path).name] = None
            
            successful_count = len([v for v in results.values() if v is not None])
            logger.info(f"Toplu özellik çıkarma tamamlandı: {successful_count}/{len(audio_files)} başarılı")
            
            return results
            
        except Exception as e:
            logger.error(f"Toplu özellik çıkarma hatası: {str(e)}")
            return {}
    
    def save_features_to_csv(
        self, 
        features_dict: Dict[str, FeatureVector], 
        output_path: str
    ) -> str:
        """
        Özellikleri CSV dosyasına kaydet
        
        Args:
            features_dict: Dosya adı -> FeatureVector mapping
            output_path: Çıktı CSV dosya yolu
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        try:
            # DataFrame oluştur
            rows = []
            
            for file_name, feature_vector in features_dict.items():
                if feature_vector is None:
                    continue
                
                row = {'file_name': file_name}
                row.update(feature_vector.features)
                row['audio_duration'] = feature_vector.audio_duration
                row['sample_rate'] = feature_vector.sample_rate
                
                rows.append(row)
            
            if not rows:
                raise FeatureExtractionError("Kaydedilecek özellik bulunamadı")
            
            df = pd.DataFrame(rows)
            
            # CSV'ye kaydet
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False)
            
            logger.info(f"Özellikler CSV'ye kaydedildi: {output_path} ({len(df)} satır, {len(df.columns)} sütun)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"CSV kaydetme hatası: {str(e)}")
            raise FeatureExtractionError(f"CSV kaydetme hatası: {str(e)}") from e
    
    def __del__(self):
        """Thread pool executor'ı temizle"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# ================================
# SINGLETON VE CONVENIENCE FUNCTIONS
# ================================

# Global instance
_feature_extractor_instance: Optional[FeatureExtractor] = None

def get_feature_extractor(config: Optional[Settings] = None) -> FeatureExtractor:
    """
    FeatureExtractor singleton instance'ını al
    
    Args:
        config: Opsiyonel konfigürasyon override'ı
        
    Returns:
        FeatureExtractor: Singleton instance
    """
    global _feature_extractor_instance
    
    if _feature_extractor_instance is None:
        _feature_extractor_instance = FeatureExtractor(config)
    
    return _feature_extractor_instance

# ================================
# CONVENIENCE FUNCTIONS
# ================================

async def extract_audio_features(
    file_path: str, 
    feature_types: Optional[List[str]] = None
) -> FeatureVector:
    """
    Convenience function - ses dosyasından özellik çıkarma
    
    Args:
        file_path: Ses dosyası yolu
        feature_types: Çıkarılacak özellik türleri
        
    Returns:
        FeatureVector: Çıkarılan özellikler
    """
    try:
        # Audio'yu yükle
        audio_data, sample_rate = audio_loader.load_audio(file_path)
        
        # AudioData objesi oluştur
        audio_obj = AudioData(
            audio_data=audio_data,
            sample_rate=sample_rate,
            metadata=None,
            file_hash=None
        )
        
        # FeatureExtractor instance'ını al
        extractor = get_feature_extractor()
        
        # Özellikleri çıkar
        return await extractor.extract_features(audio_obj, feature_types)
        
    except Exception as e:
        logger.error(f"Özellik çıkarma convenience function hatası: {str(e)}")
        raise FeatureExtractionError(f"Özellik çıkarma hatası: {str(e)}") from e

async def extract_features_from_audio_data(
    audio_data: np.ndarray,
    sample_rate: int,
    feature_types: Optional[List[str]] = None
) -> FeatureVector:
    """
    Convenience function - numpy array'den özellik çıkarma
    
    Args:
        audio_data: Ses verisi numpy array
        sample_rate: Örnekleme oranı
        feature_types: Çıkarılacak özellik türleri
        
    Returns:
        FeatureVector: Çıkarılan özellikler
    """
    try:
        # AudioData objesi oluştur
        audio_obj = AudioData(
            audio_data=audio_data,
            sample_rate=sample_rate,
            metadata=None,
            file_hash=None
        )
        
        # FeatureExtractor instance'ını al
        extractor = get_feature_extractor()
        
        # Özellikleri çıkar
        return await extractor.extract_features(audio_obj, feature_types)
        
    except Exception as e:
        logger.error(f"Array'den özellik çıkarma hatası: {str(e)}")
        raise FeatureExtractionError(f"Özellik çıkarma hatası: {str(e)}") from e

def get_available_feature_types() -> List[str]:
    """
    Mevcut özellik türlerini listele
    
    Returns:
        List[str]: Özellik türleri listesi
    """
    return ['mfcc', 'spectral', 'prosodic', 'temporal']

def get_feature_descriptions() -> Dict[str, str]:
    """
    Özellik türlerinin açıklamalarını al
    
    Returns:
        Dict[str, str]: Özellik türü -> açıklama mapping
    """
    return {
        'mfcc': 'MFCC (Mel-Frequency Cepstral Coefficients) - Ses karakteristikleri ve tonalite',
        'spectral': 'Spektral özellikler - Ton, pitch ve frekans analizi',
        'prosodic': 'Prosodik özellikler - Ritim, vurgu, tempo ve konuşma dinamikleri',
        'temporal': 'Zaman domain özellikleri - Temel ses istatistikleri ve amplitüd analizi'
    }

# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    async def main():
        """Test ve örnek kullanım"""
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            
            try:
                print(f"FeatureExtractor test ediliyor: {test_file}")
                
                # Mevcut özellik türlerini göster
                print(f"Mevcut özellik türleri: {get_available_feature_types()}")
                
                # Tüm özellikleri çıkar
                features = await extract_audio_features(test_file)
                
                print(f"\nÇıkarılan özellikler:")
                print(f"Toplam özellik sayısı: {len(features.features)}")
                print(f"Audio süresi: {features.audio_duration:.2f}s")
                print(f"Sample rate: {features.sample_rate}Hz")
                
                # Özellik özeti
                extractor = get_feature_extractor()
                summary = extractor.get_feature_summary(features)
                print(f"\nÖzellik özeti: {summary['feature_categories']}")
                
                # İlk 10 özelliği göster
                feature_items = list(features.features.items())[:10]
                print(f"\nİlk 10 özellik:")
                for name, value in feature_items:
                    print(f"  {name}: {value:.4f}")
                
            except Exception as e:
                print(f"Hata: {e}")
        else:
            print("Kullanım: python feature_extract.py <audio_file>")
            print("FeatureExtractor modülü başarıyla yüklendi!")
    
    # Async main'i çalıştır
    if __name__ == "__main__":
        asyncio.run(main())