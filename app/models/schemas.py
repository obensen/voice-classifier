# app/models/schemas.py
# API request/response modelleri ve enum sınıfları

from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any, Optional, List

# ===== ENUM SINIFLAR =====

class Gender(str, Enum):
    """Cinsiyet enum sınıfı"""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class AgeGroup(str, Enum):
    """Yaş grubu enum sınıfı"""
    CHILD = "child"       # 0-12 yaş
    YOUNG = "young"       # 13-17 yaş  
    TEEN = "teen"         # 18-25 yaş
    ADULT = "adult"       # 26-60 yaş
    SENIOR = "senior"     # 60+ yaş
    UNKNOWN = "unknown"

class Language(str, Enum):
    """Dil enum sınıfı"""
    TURKISH = "tr"
    ENGLISH = "en" 
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    UNKNOWN = "unknown"

class Tone(str, Enum):
    """Konuşma tonu enum sınıfı"""
    FORMAL = "formal"           # Resmi
    INFORMAL = "informal"       # Gayri resmi
    FRIENDLY = "friendly"       # Dostane
    PROFESSIONAL = "professional" # Profesyonel
    CASUAL = "casual"           # Rahat
    EXCITED = "excited"         # Heyecanlı
    ANGRY = "angry"             # Kızgın
    SAD = "sad"                 # Üzgün
    CALM = "calm"               # Sakin
    NEUTRAL = "neutral"         # Nötr
    UNKNOWN = "unknown"

class Emotion(str, Enum):
    """Duygu durumu enum sınıfı"""
    HAPPY = "happy"             # Mutlu
    SAD = "sad"                 # Üzgün
    ANGRY = "angry"             # Kızgın
    NEUTRAL = "neutral"         # Nötr
    EXCITED = "excited"         # Heyecanlı
    CALM = "calm"               # Sakin
    HAPPINESS = "happiness"     # Mutluluk
    SADNESS = "sadness"         # Üzüntü
    FEAR = "fear"               # Korku
    DISGUST = "disgust"         # Tiksinti
    SURPRISE = "surprise"       # Şaşkınlık
    UNKNOWN = "unknown"

# ===== PYDANTIC MODELLER =====

class HealthResponse(BaseModel):
    """API sağlık kontrolü response modeli"""
    status: str = Field(..., description="API durumu")
    message: str = Field(..., description="Durum mesajı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "message": "API is healthy!"
            }
        }

class AnalysisResult(BaseModel):
    """Ses analizi sonucu modeli"""
    gender: Gender = Field(..., description="Tespit edilen cinsiyet")
    age_group: AgeGroup = Field(..., description="Tespit edilen yaş grubu")
    language: Language = Field(..., description="Tespit edilen dil")
    tone: Tone = Field(..., description="Tespit edilen konuşma tonu")
    emotion: Emotion = Field(..., description="Tespit edilen duygu durumu")
    confidence: Dict[str, float] = Field(..., description="Her analiz için güven skorları")
    features: Optional[Dict[str, Any]] = Field(None, description="Çıkarılan ses özellikleri")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "male",
                "age_group": "adult",
                "language": "tr",
                "tone": "neutral",
                "emotion": "neutral",
                "confidence": {
                    "gender": 0.85,
                    "age_group": 0.72,
                    "language": 0.91,
                    "tone": 0.68,
                    "emotion": 0.73
                },
                "features": {
                    "f0": 150.5,
                    "rms_energy": 0.03,
                    "tempo": 120
                }
            }
        }

class AnalysisResponse(BaseModel):
    """API analiz response modeli"""
    success: bool = Field(..., description="İşlem başarı durumu")
    message: str = Field(..., description="İşlem sonuç mesajı")
    result: Optional[AnalysisResult] = Field(None, description="Analiz sonuçları")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Ek bilgiler")
    error: Optional[str] = Field(None, description="Hata mesajı (varsa)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Audio analyzed successfully",
                "result": {
                    "gender": "female",
                    "age_group": "adult",
                    "language": "en",
                    "tone": "professional",
                    "emotion": "neutral",
                    "confidence": {
                        "gender": 0.89,
                        "age_group": 0.76,
                        "language": 0.94,
                        "tone": 0.82,
                        "emotion": 0.71
                    }
                },
                "metadata": {
                    "duration": 3.5,
                    "sample_rate": 16000,
                    "file_size": 112000
                }
            }
        }

# ===== TTS ÖZEL MODELLER =====

class VoiceProfile(BaseModel):
    """TTS ses profili modeli"""
    gender: Gender
    age_group: AgeGroup
    language: Language
    tone: Tone
    emotion: Emotion
    confidence_scores: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "female",
                "age_group": "adult", 
                "language": "tr",
                "tone": "calm",
                "emotion": "neutral",
                "confidence_scores": {
                    "gender": 0.87,
                    "age_group": 0.72,
                    "language": 0.96,
                    "tone": 0.84,
                    "emotion": 0.69
                }
            }
        }

class CategoryMatch(BaseModel):
    """Kategori eşleştirme sonucu modeli"""
    category: str = Field(..., description="Video kategorisi adı")
    description: str = Field(..., description="Kategori açıklaması")
    compatibility_score: float = Field(..., ge=0.0, le=1.0, description="Uyumluluk skoru")
    explanation: str = Field(..., description="Eşleştirme açıklaması")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Haber",
                "description": "Resmi, güvenilir ve tarafsız bilgi sunumu",
                "compatibility_score": 0.87,
                "explanation": "Erkek ses ve nötr ton haber kategorisi için ideal..."
            }
        }

class TTSAnalysisResponse(BaseModel):
    """TTS analizi response modeli"""
    success: bool = Field(..., description="İşlem başarı durumu")
    message: str = Field(..., description="İşlem sonuç mesajı")
    voice_profile: Optional[VoiceProfile] = Field(None, description="Ses profili")
    category_matches: Optional[List[CategoryMatch]] = Field(None, description="Kategori eşleştirmeleri")
    best_match: Optional[CategoryMatch] = Field(None, description="En iyi eşleştirme")
    recommendations: Optional[List[str]] = Field(None, description="Öneriler")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Ek bilgiler")
    error: Optional[str] = Field(None, description="Hata mesajı (varsa)")

# ===== BATCH PROCESSING MODELLER =====

class BatchAnalysisResult(BaseModel):
    """Toplu analiz sonucu modeli"""
    file_name: str = Field(..., description="Dosya adı")
    analysis_result: Optional[AnalysisResult] = Field(None, description="Analiz sonucu")
    error: Optional[str] = Field(None, description="Hata mesajı (varsa)")

class BatchAnalysisResponse(BaseModel):
    """Toplu analiz response modeli"""
    success: bool = Field(..., description="İşlem başarı durumu")
    message: str = Field(..., description="İşlem sonuç mesajı")
    results: List[BatchAnalysisResult] = Field(..., description="Tüm dosyaların analiz sonuçları")
    summary: Optional[Dict[str, Any]] = Field(None, description="Özet bilgiler")
