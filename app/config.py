# app/config.py
# Yapılandırma ayarları ve environment variables yönetimi

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """
    Uygulama yapılandırma ayarları
    Environment variables ve default değerler
    """
    
    # ================================
    # API SETTINGS
    # ================================
    
    app_name: str = Field(default="Voice Classifier API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="TTS seslerini analiz ederek video kategorileriyle eşleştiren akıllı sistem",
        env="APP_DESCRIPTION"
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=7000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=True, env="RELOAD")
    
    # API limits
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB", description="Max file size in MB")
    max_files_per_batch: int = Field(default=10, env="MAX_FILES_PER_BATCH")
    request_timeout_seconds: int = Field(default=300, env="REQUEST_TIMEOUT_SECONDS")
    
    # ================================
    # SECURITY SETTINGS  
    # ================================
    
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"], 
        env="CORS_ORIGINS",
        description="Comma-separated list of allowed origins"
    )
    
    # ================================
    # PATHS SETTINGS
    # ================================
    
    # Base paths
    base_dir: Path = Field(default=Path(__file__).parent.parent)
    app_dir: Path = Field(default=Path(__file__).parent)
    
    # Data directories  
    temp_audio_dir: str = Field(default="temp_audio", env="TEMP_AUDIO_DIR")
    models_dir: str = Field(default="models", env="MODELS_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    static_dir: str = Field(default="static", env="STATIC_DIR")
    
    # ================================
    # CACHE SETTINGS
    # ================================
    
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS", description="5 minutes default")
    cache_max_entries: int = Field(default=1000, env="CACHE_MAX_ENTRIES")
    cache_cleanup_interval: int = Field(default=3600, env="CACHE_CLEANUP_INTERVAL", description="1 hour")
    
    # ================================
    # LOGGING SETTINGS
    # ================================
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    log_file_max_mb: int = Field(default=10, env="LOG_FILE_MAX_MB")
    log_file_backup_count: int = Field(default=5, env="LOG_FILE_BACKUP_COUNT")
    
    # ================================
    # ML MODEL SETTINGS
    # ================================
    
    # Model loading settings
    model_lazy_loading: bool = Field(default=True, env="MODEL_LAZY_LOADING")
    model_cache_enabled: bool = Field(default=True, env="MODEL_CACHE_ENABLED")
    model_device: str = Field(default="cpu", env="MODEL_DEVICE", description="cpu, cuda, mps")
    
    # Whisper settings (Language Detection)
    whisper_model_size: str = Field(default="base", env="WHISPER_MODEL_SIZE", description="tiny, base, small, medium, large")
    whisper_model_path: Optional[str] = Field(default=None, env="WHISPER_MODEL_PATH")
    
    # Gender detection settings  
    gender_model_enabled: bool = Field(default=True, env="GENDER_MODEL_ENABLED")
    gender_model_path: Optional[str] = Field(default=None, env="GENDER_MODEL_PATH")
    gender_confidence_threshold: float = Field(default=0.7, env="GENDER_CONFIDENCE_THRESHOLD")
    
    # Age detection settings
    age_model_enabled: bool = Field(default=True, env="AGE_MODEL_ENABLED")
    age_model_path: Optional[str] = Field(default=None, env="AGE_MODEL_PATH")
    age_confidence_threshold: float = Field(default=0.6, env="AGE_CONFIDENCE_THRESHOLD")
    
    # Tone analysis settings
    tone_model_enabled: bool = Field(default=True, env="TONE_MODEL_ENABLED")
    tone_model_path: Optional[str] = Field(default=None, env="TONE_MODEL_PATH")
    tone_confidence_threshold: float = Field(default=0.65, env="TONE_CONFIDENCE_THRESHOLD")
    
    # Emotion analysis settings
    emotion_model_enabled: bool = Field(default=True, env="EMOTION_MODEL_ENABLED")
    emotion_model_path: Optional[str] = Field(default=None, env="EMOTION_MODEL_PATH")
    emotion_confidence_threshold: float = Field(default=0.6, env="EMOTION_CONFIDENCE_THRESHOLD")
    
    # ================================
    # AUDIO PROCESSING SETTINGS
    # ================================
    
    # Audio format settings
    supported_audio_formats: List[str] = Field(
        default=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
        env="SUPPORTED_AUDIO_FORMATS"
    )
    default_sample_rate: int = Field(default=16000, env="DEFAULT_SAMPLE_RATE")
    max_audio_duration_seconds: int = Field(default=300, env="MAX_AUDIO_DURATION", description="5 minutes max")
    
    # Feature extraction settings
    feature_extraction_enabled: bool = Field(default=True, env="FEATURE_EXTRACTION_ENABLED")
    extract_mfcc: bool = Field(default=True, env="EXTRACT_MFCC")
    extract_spectral: bool = Field(default=True, env="EXTRACT_SPECTRAL") 
    extract_prosodic: bool = Field(default=True, env="EXTRACT_PROSODIC")
    
    # Audio preprocessing
    normalize_audio: bool = Field(default=True, env="NORMALIZE_AUDIO")
    remove_silence: bool = Field(default=False, env="REMOVE_SILENCE")
    noise_reduction: bool = Field(default=False, env="NOISE_REDUCTION")
    
    # ================================
    # TTS CATEGORY SETTINGS
    # ================================
    
    # Category matching
    category_matching_enabled: bool = Field(default=True, env="CATEGORY_MATCHING_ENABLED")
    min_compatibility_score: float = Field(default=0.5, env="MIN_COMPATIBILITY_SCORE")
    max_categories_returned: int = Field(default=5, env="MAX_CATEGORIES_RETURNED")
    
    # Predefined categories
    video_categories: List[str] = Field(
        default=[
            "Haber", "Belgesel", "Eğitim", "Reklam", "Çocuk İçerikleri",
            "Teknik Eğitim", "Müzik", "Spor", "Yemek", "Seyahat", "Podcast"
        ],
        env="VIDEO_CATEGORIES"
    )
    
    # ================================
    # PERFORMANCE SETTINGS
    # ================================
    
    # Parallel processing
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    enable_parallel_analysis: bool = Field(default=True, env="ENABLE_PARALLEL_ANALYSIS")
    
    # Memory management
    max_memory_usage_mb: int = Field(default=2048, env="MAX_MEMORY_USAGE_MB", description="2GB default")
    gc_enabled: bool = Field(default=True, env="GC_ENABLED", description="Garbage collection")
    
    # ================================
    # DATABASE SETTINGS (Future)
    # ================================
    
    database_enabled: bool = Field(default=False, env="DATABASE_ENABLED")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    
    # ================================
    # MONITORING & METRICS
    # ================================
    
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    prometheus_enabled: bool = Field(default=False, env="PROMETHEUS_ENABLED")
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL", description="seconds")
    
    # ================================
    # DEVELOPMENT SETTINGS
    # ================================
    
    development_mode: bool = Field(default=False, env="DEVELOPMENT_MODE")
    mock_responses: bool = Field(default=True, env="MOCK_RESPONSES", description="Use mock responses for development")
    profiling_enabled: bool = Field(default=False, env="PROFILING_ENABLED")
    
    # ================================
    # COMPUTED PROPERTIES
    # ================================
    
    @property
    def temp_audio_path(self) -> Path:
        """Temp audio dizin path'i"""
        return self.base_dir / self.temp_audio_dir
    
    @property  
    def models_path(self) -> Path:
        """Models dizin path'i"""
        return self.base_dir / self.models_dir
    
    @property
    def logs_path(self) -> Path:
        """Logs dizin path'i"""
        return self.base_dir / self.logs_dir
        
    @property
    def max_file_size_bytes(self) -> int:
        """Max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    # ================================
    # METHODS
    # ================================
    
    def create_directories(self):
        """Gerekli dizinleri oluştur"""
        directories = [
            self.temp_audio_path,
            self.models_path, 
            self.logs_path,
            self.base_dir / self.static_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> dict:
        """Model konfigürasyonlarını toplu olarak döndür"""
        return {
            "device": self.model_device,
            "lazy_loading": self.model_lazy_loading,
            "cache_enabled": self.model_cache_enabled,
            "whisper": {
                "model_size": self.whisper_model_size,
                "model_path": self.whisper_model_path
            },
            "gender": {
                "enabled": self.gender_model_enabled,
                "model_path": self.gender_model_path,
                "threshold": self.gender_confidence_threshold
            },
            "age": {
                "enabled": self.age_model_enabled,
                "model_path": self.age_model_path,
                "threshold": self.age_confidence_threshold
            },
            "tone": {
                "enabled": self.tone_model_enabled,
                "model_path": self.tone_model_path,
                "threshold": self.tone_confidence_threshold
            },
            "emotion": {
                "enabled": self.emotion_model_enabled,
                "model_path": self.emotion_model_path,
                "threshold": self.emotion_confidence_threshold
            }
        }
    
    def get_audio_config(self) -> dict:
        """Audio processing konfigürasyonları"""
        return {
            "supported_formats": self.supported_audio_formats,
            "sample_rate": self.default_sample_rate,
            "max_duration": self.max_audio_duration_seconds,
            "max_file_size": self.max_file_size_bytes,
            "preprocessing": {
                "normalize": self.normalize_audio,
                "remove_silence": self.remove_silence,
                "noise_reduction": self.noise_reduction
            },
            "features": {
                "enabled": self.feature_extraction_enabled,
                "mfcc": self.extract_mfcc,
                "spectral": self.extract_spectral,
                "prosodic": self.extract_prosodic
            }
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# ================================
# GLOBAL SETTINGS INSTANCE
# ================================

@lru_cache()
def get_settings() -> Settings:
    """
    Settings instance'ını cache'leyerek döndür
    Singleton pattern ile tek instance kullan
    """
    return Settings()

# Global settings instance
settings = get_settings()

# ================================
# ENVIRONMENT HELPERS
# ================================

def is_development() -> bool:
    """Development environment kontrolü"""
    return settings.development_mode or settings.debug

def is_production() -> bool:
    """Production environment kontrolü"""  
    return not is_development()

def get_cors_origins() -> List[str]:
    """CORS origins listesi"""
    if isinstance(settings.cors_origins, str):
        return [origin.strip() for origin in settings.cors_origins.split(",")]
    return settings.cors_origins

# ================================
# STARTUP INITIALIZATION
# ================================

def initialize_app():
    """Uygulama başlatma işlemleri"""
    # Dizinleri oluştur
    settings.create_directories()
    
    # Environment kontrolü
    if is_development():
        print("🔧 Development mode active")
    else:
        print("🚀 Production mode active")
    
    # Model konfigürasyonlarını logla
    model_config = settings.get_model_config()
    print(f"📊 Models on device: {model_config['device']}")
    print(f"⚡ Lazy loading: {model_config['lazy_loading']}")
    
    # Audio konfigürasyonlarını logla
    audio_config = settings.get_audio_config()
    print(f"🎵 Supported formats: {', '.join(audio_config['supported_formats'])}")
    print(f"📏 Max file size: {settings.max_file_size_mb}MB")
    
    return settings