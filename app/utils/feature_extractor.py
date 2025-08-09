# app/utils/feature_extract.py
# Ses özellik çıkarma sistemi - ML modelleri için feature extraction

import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

from app.config import settings

# Uyarıları bastır
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class FeatureExtractionError(Exception):
    """Özellik çıkarma hatası"""
    pass

class AudioFeatureExtractor:
    """
    Ses dosyalarından makine öğrenmesi için özellik çıkarma sınıfı
    MFCC, spectral, prosodic ve diğer audio feature'ları çıkarır
    """
    
    def __init__(self):
        """FeatureExtractor'ı başlat"""
        self.sample_rate = settings.default_sample_rate
        self.extract_mfcc = settings.extract_mfcc
        self.extract_spectral = settings.extract_spectral
        self.extract_prosodic = settings.extract_prosodic
        
        # MFCC parametreleri
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        logger.info("AudioFeatureExtractor initialized")
        logger.info(f"Extract MFCC: {self.extract_mfcc}, Spectral: {self.extract_spectral}, Prosodic: {self.extract_prosodic}")
    
    def extract_all_features(
        self, 
        audio_data: np.ndarray, 
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tüm audio özelliklerini çıkar
        
        Args:
            audio_data: Audio sinyal verisi (1D numpy array)
            sample_rate: Sample rate (opsiyonel)
            
        Returns:
            Dict: Tüm çıkarılan özellikler
        """
        try:
            sr = sample_rate or self.sample_rate
            features = {}
            
            logger.debug(f"Extracting features from audio: {audio_data.shape}, {sr}Hz")
            
            # Temel özellikler
            features.update(self._extract_basic_features(audio_data, sr))
            
            # MFCC özellikleri
            if self.extract_mfcc:
                features.update(self._extract_mfcc_features(audio_data, sr))
            
            # Spektral özellikler
            if self.extract_spectral:
                features.update(self._extract_spectral_features(audio_data, sr))
            
            # Prosodic özellikler (pitch, rhythm, vb.)
            if self.extract_prosodic:
                features.update(self._extract_prosodic_features(audio_data, sr))
            
            # Ek özellikler
            features.update(self._extract_additional_features(audio_data, sr))
            
            logger.info(f"Feature extraction completed: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise FeatureExtractionError(f"Feature extraction error: {e}")
    
    def _extract_basic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Temel audio özellikleri"""
        try:
            features = {}
            
            # Duration
            features['duration'] = len(audio_data) / sr
            
            # RMS Energy (Root Mean Square)
            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_max'] = float(np.max(rms))
            features['rms_min'] = float(np.min(rms))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # Amplitude statistics
            features['amplitude_mean'] = float(np.mean(np.abs(audio_data)))
            features['amplitude_std'] = float(np.std(audio_data))
            features['amplitude_max'] = float(np.max(np.abs(audio_data)))
            features['amplitude_range'] = float(np.ptp(audio_data))  # peak-to-peak
            
            # Silence ratio (düşük enerji yüzdesi)
            silence_threshold = 0.01
            silence_frames = np.sum(rms < silence_threshold)
            features['silence_ratio'] = float(silence_frames / len(rms))
            
            logger.debug("Basic features extracted")
            return features
            
        except Exception as e:
            logger.warning(f"Basic feature extraction failed: {e}")
            return {}
    
    def _extract_mfcc_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """MFCC (Mel-Frequency Cepstral Coefficients) özellikleri"""
        try:
            features = {}
            
            # MFCC çıkarma
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Her MFCC coefficient için istatistikler
            for i in range(self.n_mfcc):
                mfcc_coef = mfcc[i]
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc_coef))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfcc_coef))
                features[f'mfcc_{i+1}_max'] = float(np.max(mfcc_coef))
                features[f'mfcc_{i+1}_min'] = float(np.min(mfcc_coef))
            
            # MFCC delta (1. türev) ve delta-delta (2. türev)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Delta istatistikleri
            features['mfcc_delta_mean'] = float(np.mean(mfcc_delta))
            features['mfcc_delta_std'] = float(np.std(mfcc_delta))
            features['mfcc_delta2_mean'] = float(np.mean(mfcc_delta2))
            features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))
            
            logger.debug(f"MFCC features extracted: {self.n_mfcc} coefficients")
            return features
            
        except Exception as e:
            logger.warning(f"MFCC feature extraction failed: {e}")
            return {}
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Spektral özellikler (frequency domain)"""
        try:
            features = {}
            
            # Spectral Centroid (spektral ağırlık merkezi)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            features['spectral_centroid_max'] = float(np.max(spectral_centroids))
            features['spectral_centroid_min'] = float(np.min(spectral_centroids))
            
            # Spectral Bandwidth (spektral bant genişliği)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=self.hop_length)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Spectral Rolloff (spectral roll-off frequency)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=self.hop_length)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral Contrast (spektral kontrast)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, hop_length=self.hop_length)
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            features['spectral_contrast_std'] = float(np.std(spectral_contrast))
            
            # Spectral Flatness (spectral flatness measure)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, hop_length=self.hop_length)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            # Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=self.hop_length)
            for i in range(12):  # 12 semitone
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
            features['chroma_stdev'] = float(np.mean(np.std(chroma, axis=1)))
            
            # Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr)
            for i in range(6):  # 6 tonnetz dimensions
                features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
            
            logger.debug("Spectral features extracted")
            return features
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return {}
    
    def _extract_prosodic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Prosodic özellikler (pitch, rhythm, tempo)"""
        try:
            features = {}
            
            # Fundamental Frequency (F0) - Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, hop_length=self.hop_length)
            
            # En güçlü pitch değerlerini al
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Sadece geçerli pitch değerleri
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                
                # Pitch statistics
                features['pitch_median'] = float(np.median(pitch_values))
                features['pitch_q25'] = float(np.percentile(pitch_values, 25))
                features['pitch_q75'] = float(np.percentile(pitch_values, 75))
                features['pitch_iqr'] = features['pitch_q75'] - features['pitch_q25']
                
                # Voice stability (pitch variation)
                features['pitch_stability'] = float(1 / (1 + features['pitch_std'] / features['pitch_mean']))
            else:
                # Sessizlik veya pitch tespit edilemedi
                for key in ['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min', 'pitch_range', 
                           'pitch_median', 'pitch_q25', 'pitch_q75', 'pitch_iqr', 'pitch_stability']:
                    features[key] = 0.0
            
            # Tempo (beat tracking)
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=self.hop_length)
                features['tempo'] = float(tempo)
                
                # Rhythmic regularity
                if len(beat_frames) > 1:
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                    beat_intervals = np.diff(beat_times)
                    features['rhythm_regularity'] = float(1 / (1 + np.std(beat_intervals)))
                    features['beat_strength'] = float(np.mean(beat_intervals))
                else:
                    features['rhythm_regularity'] = 0.0
                    features['beat_strength'] = 0.0
                    
            except Exception:
                features['tempo'] = 0.0
                features['rhythm_regularity'] = 0.0
                features['beat_strength'] = 0.0
            
            # Speech rate estimation (syllable-like units per second)
            # Envelope-based approach
            envelope = librosa.onset.onset_strength(y=audio_data, sr=sr)
            peaks, _ = find_peaks(envelope, distance=int(sr/8))  # Min 125ms apart
            speech_rate = len(peaks) / (len(audio_data) / sr)
            features['speech_rate'] = float(speech_rate)
            
            # Energy contour features
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length_energy = int(0.01 * sr)  # 10ms hop
            
            energy_frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length_energy):
                frame = audio_data[i:i+frame_length]
                energy = np.sum(frame**2)
                energy_frames.append(energy)
            
            if energy_frames:
                energy_frames = np.array(energy_frames)
                features['energy_mean'] = float(np.mean(energy_frames))
                features['energy_std'] = float(np.std(energy_frames))
                features['energy_max'] = float(np.max(energy_frames))
                features['energy_range'] = float(np.ptp(energy_frames))
                
                # Energy dynamics
                energy_diff = np.diff(energy_frames)
                features['energy_dynamics'] = float(np.std(energy_diff))
            
            logger.debug("Prosodic features extracted")
            return features
            
        except Exception as e:
            logger.warning(f"Prosodic feature extraction failed: {e}")
            return {}
    
    def _extract_additional_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Ek özellikler (voice quality, etc.)"""
        try:
            features = {}
            
            # Mel-scale features
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
            )
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            features['mel_mean'] = float(np.mean(mel_db))
            features['mel_std'] = float(np.std(mel_db))
            features['mel_max'] = float(np.max(mel_db))
            features['mel_min'] = float(np.min(mel_db))
            
            # Harmonic-Percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Harmonic features
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                features['harmonic_ratio'] = float(harmonic_energy / total_energy)
                features['percussive_ratio'] = float(percussive_energy / total_energy)
            else:
                features['harmonic_ratio'] = 0.0
                features['percussive_ratio'] = 0.0
            
            # Voice quality indicators
            # Jitter approximation (pitch perturbation)
            rms_frames = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            if len(rms_frames) > 1:
                rms_variation = np.std(np.diff(rms_frames))
                features['voice_quality_jitter'] = float(rms_variation)
            else:
                features['voice_quality_jitter'] = 0.0
            
            # Shimmer approximation (amplitude perturbation)  
            amplitude_frames = np.abs(audio_data[::self.hop_length])
            if len(amplitude_frames) > 1:
                amplitude_variation = np.std(np.diff(amplitude_frames))
                features['voice_quality_shimmer'] = float(amplitude_variation)
            else:
                features['voice_quality_shimmer'] = 0.0
            
            # Formant estimation (simplified)
            # Use LPC (Linear Predictive Coding) approximation
            try:
                # Pre-emphasis
                pre_emphasis = 0.97
                emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
                
                # Window the signal
                windowed = emphasized_audio * np.hamming(len(emphasized_audio))
                
                # Rough formant estimation using spectral peaks
                fft = np.fft.rfft(windowed)
                magnitude = np.abs(fft)
                freqs = np.fft.rfftfreq(len(windowed), 1/sr)
                
                # Find peaks in spectrum (potential formants)
                peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)
                if len(peaks) >= 2:
                    # Approximate first two formants
                    peak_freqs = freqs[peaks]
                    sorted_peaks = np.sort(peak_freqs)
                    features['formant_f1'] = float(sorted_peaks[0]) if len(sorted_peaks) > 0 else 0.0
                    features['formant_f2'] = float(sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0
                else:
                    features['formant_f1'] = 0.0
                    features['formant_f2'] = 0.0
                    
            except Exception:
                features['formant_f1'] = 0.0
                features['formant_f2'] = 0.0
            
            logger.debug("Additional features extracted")
            return features
            
        except Exception as e:
            logger.warning(f"Additional feature extraction failed: {e}")
            return {}
    
    def extract_gender_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Cinsiyet analizi için özelleşmiş özellikler"""
        try:
            features = {}
            
            # Fundamental frequency (critical for gender)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['f0_mean'] = float(np.mean(pitch_values))
                features['f0_std'] = float(np.std(pitch_values))
                features['f0_median'] = float(np.median(pitch_values))
            else:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
                features['f0_median'] = 0.0
            
            # Spectral centroid (voice brightness)
            spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spec_cent))
            
            # Formant features (critical for gender classification)
            # Simplified formant estimation
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/sr)
            
            # Find spectral peaks
            peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)
            if len(peaks) >= 2:
                peak_freqs = freqs[peaks]
                sorted_peaks = np.sort(peak_freqs)[:4]  # İlk 4 formant
                for i, freq in enumerate(sorted_peaks):
                    features[f'formant_{i+1}'] = float(freq)
            
            # MFCC (first few coefficients are good for gender)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=5)
            for i in range(5):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
            
            return features
            
        except Exception as e:
            logger.warning(f"Gender feature extraction failed: {e}")
            return {}
    
    def extract_age_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Yaş analizi için özelleşmiş özellikler"""
        try:
            features = {}
            
            # Voice quality features (important for age)
            rms = librosa.feature.rms(y=audio_data)[0]
            features['voice_stability'] = float(1 / (1 + np.std(rms) / np.mean(rms)))
            
            # Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            
            features['spec_centroid_mean'] = float(np.mean(spec_centroid))
            features['spec_bandwidth_mean'] = float(np.mean(spec_bandwidth))
            features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
            
            # Jitter and shimmer (voice quality degradation with age)
            frame_length = int(0.025 * sr)
            frames = []
            for i in range(0, len(audio_data) - frame_length, frame_length//2):
                frame = audio_data[i:i+frame_length]
                frames.append(np.sqrt(np.mean(frame**2)))
            
            if len(frames) > 1:
                features['amplitude_variation'] = float(np.std(frames) / np.mean(frames))
            else:
                features['amplitude_variation'] = 0.0
            
            # Speech rate (changes with age)
            envelope = librosa.onset.onset_strength(y=audio_data, sr=sr)
            peaks, _ = find_peaks(envelope, distance=int(sr/8))
            features['speech_rate'] = float(len(peaks) / (len(audio_data) / sr))
            
            return features
            
        except Exception as e:
            logger.warning(f"Age feature extraction failed: {e}")
            return {}
    
    def extract_emotion_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Duygu analizi için özelleşmiş özellikler"""
        try:
            features = {}
            
            # Energy-based features
            rms = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_max'] = float(np.max(rms))
            
            # Pitch features (emotional prosody)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                features['pitch_variation'] = float(np.std(pitch_values))
            
            # Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['brightness'] = float(np.mean(spec_centroid))
            
            # Tempo and rhythm
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
                features['tempo'] = float(tempo)
            except:
                features['tempo'] = 0.0
            
            # MFCC for emotional coloring
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=8)
            for i in range(8):
                features[f'emotional_mfcc_{i+1}'] = float(np.mean(mfcc[i]))
            
            return features
            
        except Exception as e:
            logger.warning(f"Emotion feature extraction failed: {e}")
            return {}

# ================================
# CONVENIENCE FUNCTIONS
# ================================

# Global feature extractor instance
feature_extractor = AudioFeatureExtractor()

def extract_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Convenience function - tüm özellikleri çıkar
    
    Args:
        audio_data: Audio verisi
        sample_rate: Sample rate
        
    Returns:
        Dict: Çıkarılan özellikler
    """
    return feature_extractor.extract_all_features(audio_data, sample_rate)

def extract_specialized_features(
    audio_data: np.ndarray, 
    sample_rate: int, 
    feature_type: str
) -> Dict[str, float]:
    """
    Özelleşmiş özellik çıkarma
    
    Args:
        audio_data: Audio verisi
        sample_rate: Sample rate
        feature_type: "gender", "age", "emotion"
        
    Returns:
        Dict: Özelleşmiş özellikler
    """
    if feature_type == "gender":
        return feature_extractor.extract_gender_features(audio_data, sample_rate)
    elif feature_type == "age":
        return feature_extractor.extract_age_features(audio_data, sample_rate)
    elif feature_type == "emotion":
        return feature_extractor.extract_emotion_features(audio_data, sample_rate)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

# ================================
# EXAMPLE USAGE & TESTING
# ================================

if __name__ == "__main__":
    # Test ve örnek kullanım
    import sys
    
    if len(sys.argv) > 1:
        from app.utils.audio_loader import load_audio_file
        
        try:
            audio_file = sys.argv[1]
            print(f"Testing FeatureExtractor with: {audio_file}")
            
            # Audio yükle
            audio_data, sr = load_audio_file(audio_file)
            print(f"Loaded audio: {audio_data.shape}, {sr}Hz")
            
            # Tüm özellikleri çıkar
            all_features = extract_features(audio_data, sr)
            print(f"Extracted {len(all_features)} features")
            
            # Özelleşmiş özellikler
            gender_features = extract_specialized_features(audio_data, sr, "gender")
            age_features = extract_specialized_features(audio_data, sr, "age")
            emotion_features = extract_specialized_features(audio_data, sr, "emotion")
            
            print(f"Gender features: {len(gender_features)}")
            print(f"Age features: {len(age_features)}")
            print(f"Emotion features: {len(emotion_features)}")
            
            # Örnek özellik değerleri
            print("\n=== Sample Features ===")
            if 'f0_mean' in gender_features:
                print(f"F0 Mean: {gender_features['f0_mean']:.2f} Hz")
            if 'spectral_centroid_mean' in all_features:
                print(f"Spectral Centroid: {all_features['spectral_centroid_mean']:.2f} Hz")
            if 'mfcc_1_mean' in all_features:
                print(f"MFCC 1: {all_features['mfcc_1_mean']:.4f}")
            if 'energy_mean' in emotion_features:
                print(f"Energy Mean: {emotion_features['energy_mean']:.4f}")
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python feature_extract.py <audio_file>")
        print("FeatureExtractor module loaded successfully!")
        print("Available functions:")
        print("- extract_features(audio_data, sample_rate)")
        print("- extract_specialized_features(audio_data, sample_rate, 'gender'/'age'/'emotion')")