# app/services/simple_gender_service.py
# Ses özelliklerinden cinsiyet tespiti servisi

import numpy as np
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

from app.models.schemas import Gender
from app.config import settings

logger = logging.getLogger(__name__)

class SimpleGenderServiceError(Exception):
    """Cinsiyet analizi hatası"""
    pass

class SimpleGenderService:
    """
    Ses özelliklerinden cinsiyet tespiti servisi
    Feature-based machine learning yaklaşımı kullanır
    """
    
    def __init__(self):
        """SimpleGenderService'i başlat"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_loaded = False
        
        # Model parametreleri
        self.model_path = settings.gender_model_path or os.path.join(settings.models_path, "gender_model.pkl")
        self.scaler_path = os.path.join(settings.models_path, "gender_scaler.pkl")
        self.confidence_threshold = settings.gender_confidence_threshold
        
        # Feature importance weights (cinsiyet tespiti için önemli özellikler)
        self.important_features = {
            'f0_mean': 0.25,           # Fundamental frequency (en önemli)
            'f0_median': 0.20,
            'spectral_centroid_mean': 0.15,  # Voice brightness
            'formant_1': 0.12,         # First formant
            'formant_2': 0.10,         # Second formant
            'pitch_mean': 0.08,        # Pitch statistics
            'harmonic_ratio': 0.05,    # Voice quality
            'mfcc_1_mean': 0.05        # MFCC coefficients
        }
        
        # Gender boundaries (heuristic thresholds)
        self.gender_thresholds = {
            'f0_high': 180,      # Hz - typically female
            'f0_low': 120,       # Hz - typically male
            'f0_uncertain': 150, # Hz - uncertain range
            'spectral_centroid_high': 3500,  # Hz
            'spectral_centroid_low': 2500    # Hz
        }
        
        # Stats
        self.classification_count = 0
        self.accuracy_estimate = 0.0
        
        logger.info("SimpleGenderService initialized")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    async def initialize_model(self):
        """ML modelini yükle veya oluştur"""
        if self.model_loaded:
            return
        
        try:
            # Önceden eğitilmiş model varsa yükle
            if await self._load_existing_model():
                logger.info(f"Gender model training completed: {accuracy:.3f} accuracy")
            return training_result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "accuracy": 0.0
            }
    
    async def _save_model(self):
        """Eğitilmiş modeli kaydet"""
        try:
            # Model dizinini oluştur
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Model ve scaler'ı kaydet
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Feature names'i kaydet
            if self.feature_names:
                feature_names_path = self.model_path.replace('.pkl', '_features.txt')
                with open(feature_names_path, 'w') as f:
                    for feature_name in self.feature_names:
                        f.write(f"{feature_name}\n")
            
            logger.info(f"Gender model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    # ================================
    # SERVICE MANAGEMENT
    # ================================
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Servis istatistikleri"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "classification_count": self.classification_count,
            "accuracy_estimate": self.accuracy_estimate,
            "important_features": list(self.important_features.keys()),
            "gender_thresholds": self.gender_thresholds,
            "feature_names_count": len(self.feature_names) if self.feature_names else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Servis sağlık kontrolü"""
        try:
            if not self.model_loaded:
                await self.initialize_model()
            
            # Test classification
            test_features = {
                'f0_mean': 150.0,
                'spectral_centroid_mean': 3000.0,
                'mfcc_1_mean': -10.5
            }
            
            test_result = await self._heuristic_classification_from_features(test_features)
            
            return {
                "status": "healthy",
                "model_ready": self.model_loaded,
                "classification_count": self.classification_count,
                "test_classification": test_result['gender'].value,
                "accuracy_estimate": self.accuracy_estimate
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
            logger.info("Cleaning up SimpleGenderService...")
            
            # Model referanslarını temizle
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.model_loaded = False
            
            logger.info("SimpleGenderService cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# ================================
# MOCK CLASSES (TEMPORARY)
# ================================

class MockGenderModel:
    """Geçici mock model sınıfı"""
    
    def __init__(self):
        self.classes_ = ['male', 'female', 'unknown']
    
    def predict(self, X):
        """Mock prediction"""
        predictions = []
        for features in X:
            # F0 bazlı basit karar
            f0_mean = features[0] if len(features) > 0 else 150
            
            if f0_mean > 180:
                predictions.append('female')
            elif f0_mean < 120:
                predictions.append('male')
            else:
                predictions.append('unknown')
                
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Mock probability prediction"""
        probas = []
        for features in X:
            f0_mean = features[0] if len(features) > 0 else 150
            
            if f0_mean > 180:
                # Female
                probas.append([0.15, 0.80, 0.05])
            elif f0_mean < 120:
                # Male  
                probas.append([0.80, 0.15, 0.05])
            else:
                # Unknown
                probas.append([0.35, 0.35, 0.30])
        
        return np.array(probas)

class MockScaler:
    """Geçici mock scaler sınıfı"""
    
    def transform(self, X):
        """Mock scaling (no actual scaling)"""
        return np.array(X)
    
    def fit_transform(self, X):
        """Mock fit and transform"""
        return np.array(X)

# ================================
# GLOBAL SERVICE INSTANCE
# ================================

# Singleton pattern
_gender_service_instance = None

def get_gender_service() -> SimpleGenderService:
    """Global SimpleGenderService instance'ı al"""
    global _gender_service_instance
    if _gender_service_instance is None:
        _gender_service_instance = SimpleGenderService()
    return _gender_service_instance

# ================================
# CONVENIENCE FUNCTIONS
# ================================

async def classify_gender_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function - features'dan cinsiyet tespiti"""
    service = get_gender_service()
    return await service.analyze(None, None, features)

async def classify_gender_from_audio(
    audio_data: np.ndarray, 
    sample_rate: int
) -> Dict[str, Any]:
    """Convenience function - audio'dan cinsiyet tespiti"""
    service = get_gender_service()
    return await service.analyze(audio_data, sample_rate)

# ================================
# EXAMPLE USAGE & TESTING
# ================================

if __name__ == "__main__":
    import sys
    
    async def test_gender_service():
        try:
            print("Testing SimpleGenderService...")
            
            # Servis oluştur
            service = get_gender_service()
            
            # Health check
            health = await service.health_check()
            print(f"Health check: {health}")
            
            # Test features
            test_cases = [
                {
                    "name": "High pitch (female)",
                    "features": {
                        'f0_mean': 220.0,
                        'f0_median': 215.0,
                        'spectral_centroid_mean': 3800.0,
                        'formant_1': 550.0,
                        'formant_2': 1500.0,
                        'mfcc_1_mean': -8.2,
                        'harmonic_ratio': 0.7
                    }
                },
                {
                    "name": "Low pitch (male)",
                    "features": {
                        'f0_mean': 105.0,
                        'f0_median': 108.0,
                        'spectral_centroid_mean': 2200.0,
                        'formant_1': 380.0,
                        'formant_2': 1100.0,
                        'mfcc_1_mean': -12.5,
                        'harmonic_ratio': 0.6
                    }
                },
                {
                    "name": "Medium pitch (uncertain)",
                    "features": {
                        'f0_mean': 155.0,
                        'f0_median': 150.0,
                        'spectral_centroid_mean': 2800.0,
                        'formant_1': 450.0,
                        'formant_2': 1300.0,
                        'mfcc_1_mean': -10.1,
                        'harmonic_ratio': 0.65
                    }
                }
            ]
            
            # Test her case
            print(f"\n=== Gender Classification Tests ===")
            for test_case in test_cases:
                print(f"\n--- {test_case['name']} ---")
                result = await service.analyze(None, None, test_case['features'])
                
                print(f"Gender: {result['gender'].value}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Method: {result['method']}")
                print(f"Probabilities: {result['probabilities']}")
                
                if 'features_used' in result:
                    print(f"Key features: {result['features_used']}")
            
            # Batch test
            print(f"\n=== Batch Classification Test ===")
            batch_features = [case['features'] for case in test_cases]
            batch_names = [case['name'] for case in test_cases]
            
            batch_results = await service.batch_analyze(batch_features, batch_names)
            for result in batch_results:
                print(f"{result['file_name']}: {result['gender'].value} ({result['confidence']:.3f})")
            
            # Stats
            stats = service.get_service_stats()
            print(f"\n=== Service Stats ===")
            print(f"Classifications performed: {stats['classification_count']}")
            print(f"Model loaded: {stats['model_loaded']}")
            print(f"Accuracy estimate: {stats['accuracy_estimate']:.3f}")
            print(f"Important features: {stats['important_features']}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Async test çalıştır
    import asyncio
    asyncio.run(test_gender_service())info("Existing gender model loaded successfully")
            else:
                # Model yoksa default/heuristic model oluştur
                await self._create_default_model()
                logger.info("Default heuristic gender model created")
            
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to initialize gender model: {e}")
            raise SimpleGenderServiceError(f"Model initialization failed: {e}")
    
    async def analyze(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Audio'dan cinsiyet tespiti yap
        
        Args:
            audio_data: Audio sinyal verisi
            sample_rate: Sample rate
            features: Önceden çıkarılmış features
            
        Returns:
            Dict: Cinsiyet tespiti sonuçları
        """
        try:
            # Model yükle
            await self.initialize_model()
            
            # Features'ı hazırla
            if features is None:
                # Bu durumda feature extraction yapılmalı
                # Gerçek implementasyonda feature_extractor kullanılacak
                logger.warning("No features provided, using basic heuristics")
                return await self._heuristic_classification(audio_data, sample_rate)
            
            # ML tabanlı classification
            result = await self._ml_classification(features)
            
            # İstatistikleri güncelle
            self.classification_count += 1
            
            logger.info(f"Gender classification completed: {result['gender']}")
            return result
            
        except Exception as e:
            logger.error(f"Gender analysis failed: {e}")
            raise SimpleGenderServiceError(f"Gender analysis failed: {e}")
    
    async def batch_analyze(
        self, 
        feature_list: List[Dict[str, Any]],
        file_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Toplu cinsiyet analizi
        
        Args:
            feature_list: Feature dictionary'leri listesi
            file_names: Dosya isimleri (opsiyonel)
            
        Returns:
            List[Dict]: Her örneklem için analiz sonuçları
        """
        try:
            await self.initialize_model()
            
            results = []
            
            for i, features in enumerate(feature_list):
                try:
                    # Her sample için analiz yap
                    result = await self._ml_classification(features)
                    
                    analysis_result = {
                        "index": i,
                        "file_name": file_names[i] if file_names and i < len(file_names) else f"sample_{i}",
                        "success": True,
                        **result
                    }
                    
                    results.append(analysis_result)
                    
                except Exception as e:
                    logger.warning(f"Batch analysis failed for sample {i}: {e}")
                    results.append({
                        "index": i,
                        "file_name": file_names[i] if file_names and i < len(file_names) else f"sample_{i}",
                        "success": False,
                        "gender": Gender.UNKNOWN,
                        "confidence": 0.0,
                        "error": str(e)
                    })
            
            successful = sum(1 for r in results if r.get("success"))
            logger.info(f"Batch gender analysis: {successful}/{len(feature_list)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch gender analysis failed: {e}")
            raise SimpleGenderServiceError(f"Batch analysis failed: {e}")
    
    # ================================
    # PRIVATE ANALYSIS METHODS
    # ================================
    
    async def _ml_classification(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """ML model ile cinsiyet classification"""
        try:
            # Feature vektorü hazırla
            feature_vector = self._prepare_feature_vector(features)
            
            if self.model and self.scaler and len(feature_vector) > 0:
                # ML prediction
                scaled_features = self.scaler.transform([feature_vector])
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
                
                # Sonuçları işle
                gender_classes = self.model.classes_
                gender_probs = dict(zip(gender_classes, probabilities))
                
                # En yüksek probability
                max_prob = max(probabilities)
                predicted_gender = Gender(prediction) if prediction in [g.value for g in Gender] else Gender.UNKNOWN
                
                return {
                    "gender": predicted_gender,
                    "confidence": float(max_prob),
                    "probabilities": {
                        "male": float(gender_probs.get('male', 0.0)),
                        "female": float(gender_probs.get('female', 0.0)),
                        "unknown": float(gender_probs.get('unknown', 0.0))
                    },
                    "method": "ml_model",
                    "feature_count": len(feature_vector)
                }
            else:
                # Fallback to heuristic
                logger.warning("ML model not available, using heuristic classification")
                return await self._heuristic_classification_from_features(features)
                
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to heuristic")
            return await self._heuristic_classification_from_features(features)
    
    async def _heuristic_classification_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Feature tabanlı heuristic classification"""
        try:
            # Fundamental frequency analysis
            f0_mean = features.get('f0_mean', 0)
            f0_median = features.get('f0_median', 0)
            pitch_mean = features.get('pitch_mean', 0)
            
            # En güvenilir pitch değerini seç
            primary_pitch = f0_mean if f0_mean > 0 else (f0_median if f0_median > 0 else pitch_mean)
            
            # Spectral features
            spectral_centroid = features.get('spectral_centroid_mean', 0)
            
            # Formant features
            formant_1 = features.get('formant_1', 0)
            formant_2 = features.get('formant_2', 0)
            
            # MFCC features
            mfcc_1 = features.get('mfcc_1_mean', 0)
            
            # Scoring system
            female_score = 0.0
            male_score = 0.0
            
            # Pitch-based scoring (en önemli faktör)
            if primary_pitch > 0:
                if primary_pitch >= self.gender_thresholds['f0_high']:
                    female_score += 0.4
                elif primary_pitch <= self.gender_thresholds['f0_low']:
                    male_score += 0.4
                elif primary_pitch > self.gender_thresholds['f0_uncertain']:
                    female_score += 0.2
                else:
                    male_score += 0.2
            
            # Spectral centroid (voice brightness)
            if spectral_centroid > 0:
                if spectral_centroid >= self.gender_thresholds['spectral_centroid_high']:
                    female_score += 0.15
                elif spectral_centroid <= self.gender_thresholds['spectral_centroid_low']:
                    male_score += 0.15
                else:
                    # Intermediate range
                    if spectral_centroid > 3000:
                        female_score += 0.05
                    else:
                        male_score += 0.05
            
            # Formant analysis (if available)
            if formant_1 > 0 and formant_2 > 0:
                # Female voices typically have higher formants
                if formant_1 > 500 and formant_2 > 1400:
                    female_score += 0.1
                elif formant_1 < 400 and formant_2 < 1200:
                    male_score += 0.1
            
            # MFCC-based feature (voice timbre)
            if mfcc_1 != 0:
                # MFCC1 genellikle erkek seslerde daha düşük
                if mfcc_1 > 0:
                    female_score += 0.05
                else:
                    male_score += 0.05
            
            # Sonuç hesaplama
            if female_score > male_score and female_score > 0.3:
                gender = Gender.FEMALE
                confidence = min(female_score, 0.95)
            elif male_score > female_score and male_score > 0.3:
                gender = Gender.MALE
                confidence = min(male_score, 0.95)
            else:
                gender = Gender.UNKNOWN
                confidence = max(female_score, male_score) * 0.5  # Düşük confidence
            
            # Probability distribution
            total_score = female_score + male_score + 0.1  # Unknown için base
            probabilities = {
                "female": float(female_score / total_score),
                "male": float(male_score / total_score),
                "unknown": float(0.1 / total_score)
            }
            
            return {
                "gender": gender,
                "confidence": float(confidence),
                "probabilities": probabilities,
                "method": "heuristic",
                "features_used": {
                    "primary_pitch": float(primary_pitch),
                    "spectral_centroid": float(spectral_centroid),
                    "formant_1": float(formant_1),
                    "formant_2": float(formant_2)
                },
                "scores": {
                    "female_score": float(female_score),
                    "male_score": float(male_score)
                }
            }
            
        except Exception as e:
            logger.error(f"Heuristic classification failed: {e}")
            return {
                "gender": Gender.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {"male": 0.33, "female": 0.33, "unknown": 0.34},
                "method": "fallback",
                "error": str(e)
            }
    
    async def _heuristic_classification(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Audio'dan direkt heuristic classification (feature extraction gerekli)"""
        try:
            # Basit pitch estimation
            import librosa
            
            # Fundamental frequency estimation
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                
                # Basit classification
                if avg_pitch >= 180:
                    gender = Gender.FEMALE
                    confidence = 0.75
                elif avg_pitch <= 120:
                    gender = Gender.MALE
                    confidence = 0.75
                else:
                    gender = Gender.UNKNOWN
                    confidence = 0.45
            else:
                gender = Gender.UNKNOWN
                confidence = 0.2
            
            return {
                "gender": gender,
                "confidence": confidence,
                "probabilities": {
                    "male": 0.8 if gender == Gender.MALE else (0.1 if gender == Gender.FEMALE else 0.45),
                    "female": 0.8 if gender == Gender.FEMALE else (0.1 if gender == Gender.MALE else 0.45),
                    "unknown": 0.1 if gender != Gender.UNKNOWN else 0.1
                },
                "method": "basic_pitch_analysis",
                "avg_pitch": float(avg_pitch) if pitch_values else 0.0
            }
            
        except Exception as e:
            logger.error(f"Basic heuristic classification failed: {e}")
            return {
                "gender": Gender.UNKNOWN,
                "confidence": 0.0,
                "probabilities": {"male": 0.33, "female": 0.33, "unknown": 0.34},
                "method": "error_fallback",
                "error": str(e)
            }
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Features'dan ML model için feature vector hazırla"""
        try:
            feature_vector = []
            
            # Önceden tanımlı önemli features
            if self.feature_names:
                # Eğitilmiş modelin feature'ları varsa onları kullan
                for feature_name in self.feature_names:
                    value = features.get(feature_name, 0.0)
                    feature_vector.append(float(value))
            else:
                # Default feature set
                important_features = [
                    'f0_mean', 'f0_median', 'pitch_mean', 'pitch_std',
                    'spectral_centroid_mean', 'spectral_bandwidth_mean',
                    'formant_1', 'formant_2', 
                    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
                    'harmonic_ratio', 'rms_mean', 'zcr_mean'
                ]
                
                for feature_name in important_features:
                    value = features.get(feature_name, 0.0)
                    feature_vector.append(float(value))
            
            return feature_vector
            
        except Exception as e:
            logger.warning(f"Feature vector preparation failed: {e}")
            return []
    
    # ================================
    # MODEL MANAGEMENT
    # ================================
    
    async def _load_existing_model(self) -> bool:
        """Önceden eğitilmiş modeli yükle"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Model ve scaler'ı yükle
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Feature names'i yükle (varsa)
                feature_names_path = self.model_path.replace('.pkl', '_features.txt')
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        self.feature_names = [line.strip() for line in f.readlines()]
                
                logger.info(f"Loaded existing gender model from {self.model_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            return False
    
    async def _create_default_model(self):
        """Default/heuristic tabanlı basit model oluştur"""
        try:
            # Basit mock model (gerçek data ile değiştirilecek)
            self.model = MockGenderModel()
            self.scaler = MockScaler()
            
            # Feature names
            self.feature_names = [
                'f0_mean', 'f0_median', 'pitch_mean', 'pitch_std',
                'spectral_centroid_mean', 'spectral_bandwidth_mean',
                'formant_1', 'formant_2', 
                'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
                'harmonic_ratio', 'rms_mean', 'zcr_mean'
            ]
            
            logger.info("Created default mock gender model")
            
        except Exception as e:
            logger.error(f"Failed to create default model: {e}")
            raise SimpleGenderServiceError(f"Default model creation failed: {e}")
    
    async def train_model(
        self, 
        training_features: List[Dict[str, Any]], 
        training_labels: List[str],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Yeni model eğitimi (eğitim verisi varsa)
        
        Args:
            training_features: Eğitim feature'ları
            training_labels: Eğitim labels (male, female, unknown)
            save_model: Modeli kaydet
            
        Returns:
            Dict: Eğitim sonuçları
        """
        try:
            # Feature matrix hazırla
            feature_vectors = []
            for features in training_features:
                vector = self._prepare_feature_vector(features)
                if len(vector) > 0:
                    feature_vectors.append(vector)
            
            if len(feature_vectors) == 0:
                raise ValueError("No valid feature vectors prepared")
            
            X = np.array(feature_vectors)
            y = np.array(training_labels)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaler fit
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Model training
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluation
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracy_estimate = accuracy
            
            # Save model
            if save_model:
                await self._save_model()
            
            training_result = {
                "success": True,
                "accuracy": float(accuracy),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
                "model_type": "RandomForestClassifier",
                "classes": list(self.model.classes_)
            }
            
            logger.