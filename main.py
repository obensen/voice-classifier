# main.py
# Voice Classifier API - FastAPI Ana Uygulama

import os
import tempfile
import asyncio
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

# Local imports
from app.models.schemas import (
    HealthResponse, AnalysisResponse, TTSAnalysisResponse, 
    BatchAnalysisResponse, Gender, AgeGroup, Language, Tone, Emotion
)
from app.utils.cache import cache, metrics

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Gerekli dizinleri oluştur
os.makedirs('logs', exist_ok=True)
os.makedirs('temp_audio', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global servis değişkenleri (lazy loading için)
_analysis_service = None
_tts_service = None

async def get_analysis_service():
    """Analysis service'i lazy loading ile al"""
    global _analysis_service
    if _analysis_service is None:
        # Bu satır services implementasyonu tamamlandığında açılacak
        # from app.services.analysis_service import AnalysisService
        # _analysis_service = AnalysisService()
        logger.info("Analysis service would be initialized here")
        _analysis_service = "placeholder"  # Geçici
    return _analysis_service

async def get_tts_service():
    """TTS service'i lazy loading ile al"""
    global _tts_service  
    if _tts_service is None:
        # Bu satır services implementasyonu tamamlandığında açılacak
        # from app.services.tts_analyzer_service import TTSAnalyzerService
        # _tts_service = TTSAnalyzerService()
        logger.info("TTS service would be initialized here")
        _tts_service = "placeholder"  # Geçici
    return _tts_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlatma ve kapanma olayları"""
    logger.info("🚀 Voice Classifier API başlatılıyor...")
    
    # Startup
    try:
        # Cache temizliği
        cache.clear()
        logger.info("Cache temizlendi")
        
        # Servisleri ön-yükleme (opsiyonel)
        # await get_analysis_service()
        # await get_tts_service()
        
        logger.info("✅ Voice Classifier API başarıyla başlatıldı")
        
    except Exception as e:
        logger.error(f"❌ Başlatma hatası: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Voice Classifier API kapatılıyor...")
    cache.clear()
    logger.info("Cache temizlendi, uygulama kapatıldı")

# FastAPI uygulaması
app = FastAPI(
    title="Voice Classifier API",
    description="TTS seslerini analiz ederek video kategorileriyle eşleştiren akıllı sistem",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prod'da specific origins kullan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# HEALTH CHECK ENDPOINTS
# ================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Ana endpoint - API durumu"""
    return HealthResponse(
        status="ok", 
        message="Voice Classifier API is running! Visit /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    try:
        # Cache durumu kontrolü
        cache_stats = cache.get_stats()
        
        return HealthResponse(
            status="healthy",
            message=f"API is healthy! Cache entries: {cache_stats['total_entries']}"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}"
        )

# ================================
# UTILITY FUNCTIONS
# ================================

async def save_upload_file(upload_file: UploadFile) -> str:
    """Upload edilen dosyayı geçici dizine kaydet"""
    try:
        # Geçici dosya oluştur
        suffix = Path(upload_file.filename).suffix if upload_file.filename else '.tmp'
        with tempfile.NamedTemporaryFile(
            delete=False, 
            dir='temp_audio', 
            suffix=suffix
        ) as temp_file:
            # Dosya içeriğini kopyala
            content = await upload_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            logger.info(f"File saved: {temp_file.name}, size: {len(content)} bytes")
            return temp_file.name
            
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(status_code=400, detail=f"File save failed: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Geçici dosyayı temizle"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Temp file cleaned: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

# ================================
# MAIN ANALYSIS ENDPOINTS
# ================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Tam ses analizi - tüm özellikleri analiz eder
    """
    temp_file_path = None
    try:
        # Dosya validasyonu
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Cache kontrolü
        cache_key = f"{file.filename}_{file.size}" if hasattr(file, 'size') else file.filename
        cached_result = cache.get(cache_key, "full_analysis")
        
        if cached_result:
            metrics.hit()
            logger.info(f"Cache hit for full analysis: {file.filename}")
            return cached_result
        
        metrics.miss()
        
        # Dosyayı kaydet
        temp_file_path = await save_upload_file(file)
        
        # Analiz servisi al
        analysis_service = await get_analysis_service()
        
        # TODO: Gerçek analiz implementasyonu
        # result = await analysis_service.analyze_all(temp_file_path)
        
        # Geçici mock response
        mock_result = AnalysisResponse(
            success=True,
            message="Audio analyzed successfully (mock response)",
            result={
                "gender": Gender.MALE,
                "age_group": AgeGroup.ADULT,
                "language": Language.TURKISH,
                "tone": Tone.NEUTRAL,
                "emotion": Emotion.NEUTRAL,
                "confidence": {
                    "gender": 0.85,
                    "age_group": 0.72,
                    "language": 0.91,
                    "tone": 0.68,
                    "emotion": 0.73
                }
            },
            metadata={
                "filename": file.filename,
                "processing_time": "mock"
            }
        )
        
        # Cache'e kaydet
        cache.set(cache_key, "full_analysis", mock_result, ttl=300)
        
        logger.info(f"Full analysis completed: {file.filename}")
        return mock_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

# ================================
# INDIVIDUAL ANALYSIS ENDPOINTS
# ================================

@app.post("/analyze/gender", response_model=AnalysisResponse)
async def analyze_gender(file: UploadFile = File(...)):
    """Sadece cinsiyet analizi"""
    temp_file_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        cache_key = f"{file.filename}_{getattr(file, 'size', 'unknown')}"
        cached_result = cache.get(cache_key, "gender")
        
        if cached_result:
            metrics.hit()
            return cached_result
        
        metrics.miss()
        temp_file_path = await save_upload_file(file)
        
        # TODO: Gerçek gender analysis
        mock_result = AnalysisResponse(
            success=True,
            message="Gender analysis completed (mock)",
            result={
                "gender": Gender.FEMALE,
                "confidence": {"gender": 0.89}
            }
        )
        
        cache.set(cache_key, "gender", mock_result, ttl=300)
        return mock_result
        
    except Exception as e:
        logger.error(f"Gender analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.post("/analyze/age", response_model=AnalysisResponse)
async def analyze_age(file: UploadFile = File(...)):
    """Sadece yaş analizi"""
    # Similar implementation as gender
    return AnalysisResponse(
        success=True, 
        message="Age analysis endpoint (mock)",
        result={"age_group": AgeGroup.ADULT, "confidence": {"age_group": 0.76}}
    )

@app.post("/analyze/language", response_model=AnalysisResponse)
async def analyze_language(file: UploadFile = File(...)):
    """Sadece dil analizi"""
    return AnalysisResponse(
        success=True,
        message="Language analysis endpoint (mock)", 
        result={"language": Language.TURKISH, "confidence": {"language": 0.94}}
    )

@app.post("/analyze/tone", response_model=AnalysisResponse)
async def analyze_tone(file: UploadFile = File(...)):
    """Sadece ton analizi"""
    return AnalysisResponse(
        success=True,
        message="Tone analysis endpoint (mock)",
        result={"tone": Tone.PROFESSIONAL, "confidence": {"tone": 0.82}}
    )

@app.post("/analyze/emotion", response_model=AnalysisResponse)
async def analyze_emotion(file: UploadFile = File(...)):
    """Sadece duygu analizi"""
    return AnalysisResponse(
        success=True,
        message="Emotion analysis endpoint (mock)",
        result={"emotion": Emotion.NEUTRAL, "confidence": {"emotion": 0.71}}
    )

# ================================
# TTS SPECIFIC ENDPOINTS
# ================================

@app.post("/tts/analyze", response_model=TTSAnalysisResponse)
async def tts_analyze(file: UploadFile = File(...)):
    """
    TTS ses analizi ve kategori eşleştirmesi
    """
    temp_file_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        temp_file_path = await save_upload_file(file)
        
        # TODO: Gerçek TTS analizi
        tts_service = await get_tts_service()
        
        # Mock response
        mock_response = TTSAnalysisResponse(
            success=True,
            message="TTS analysis completed (mock)",
            voice_profile={
                "gender": Gender.MALE,
                "age_group": AgeGroup.ADULT,
                "language": Language.TURKISH,
                "tone": Tone.PROFESSIONAL,
                "emotion": Emotion.NEUTRAL,
                "confidence_scores": {
                    "gender": 0.87,
                    "age_group": 0.72,
                    "language": 0.96,
                    "tone": 0.84,
                    "emotion": 0.69
                }
            },
            category_matches=[
                {
                    "category": "Haber",
                    "description": "Resmi, güvenilir ve tarafsız bilgi sunumu",
                    "compatibility_score": 0.87,
                    "explanation": "Erkek ses ve profesyonel ton haber kategorisi için ideal"
                }
            ],
            best_match={
                "category": "Haber",
                "description": "Resmi, güvenilir ve tarafsız bilgi sunumu", 
                "compatibility_score": 0.87,
                "explanation": "En yüksek uyumluluk skoru"
            }
        )
        
        return mock_response
        
    except Exception as e:
        logger.error(f"TTS analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.post("/tts/batch-analyze", response_model=BatchAnalysisResponse)
async def tts_batch_analyze(files: List[UploadFile] = File(...)):
    """
    Çoklu TTS dosyası analizi
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    temp_file_paths = []
    
    try:
        # Tüm dosyaları kaydet
        for file in files:
            if file.filename:
                temp_path = await save_upload_file(file)
                temp_file_paths.append(temp_path)
                
                # Mock analysis result
                results.append({
                    "file_name": file.filename,
                    "analysis_result": {
                        "gender": Gender.FEMALE,
                        "age_group": AgeGroup.YOUNG,
                        "language": Language.TURKISH,
                        "tone": Tone.FRIENDLY,
                        "emotion": Emotion.HAPPY,
                        "confidence": {
                            "gender": 0.91,
                            "age_group": 0.78,
                            "language": 0.95,
                            "tone": 0.83,
                            "emotion": 0.77
                        }
                    }
                })
        
        return BatchAnalysisResponse(
            success=True,
            message=f"Batch analysis completed for {len(files)} files (mock)",
            results=results,
            summary={
                "total_files": len(files),
                "successful_analyses": len(results),
                "failed_analyses": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup all temp files
        for temp_path in temp_file_paths:
            cleanup_temp_file(temp_path)

@app.post("/tts/find-best")
async def find_best_tts(
    files: List[UploadFile] = File(...),
    category: str = Form(...)
):
    """
    Verilen kategori için en uygun TTS sesini bul
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Mock implementation
    return {
        "success": True,
        "message": f"Best TTS voice found for category: {category} (mock)",
        "best_match": {
            "filename": files[0].filename,
            "compatibility_score": 0.92,
            "category": category,
            "explanation": "Mock selection - highest compatibility score"
        },
        "all_results": [
            {
                "filename": file.filename,
                "compatibility_score": 0.92 - i * 0.1,
                "rank": i + 1
            }
            for i, file in enumerate(files[:5])  # İlk 5 dosya
        ]
    }

# ================================
# CACHE & METRICS ENDPOINTS
# ================================

@app.get("/cache/stats")
async def get_cache_stats():
    """Cache istatistiklerini getir"""
    cache_stats = cache.get_stats()
    metrics_data = {
        "hits": metrics.hits,
        "misses": metrics.misses,
        "hit_rate": metrics.get_hit_rate()
    }
    
    return {
        "cache": cache_stats,
        "metrics": metrics_data
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Cache'i temizle"""
    cache.clear()
    return {"success": True, "message": "Cache cleared successfully"}

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error": f"HTTP {exc.status_code}"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": "An unexpected error occurred"
        }
    )

# ================================
# DEVELOPMENT SERVER
# ================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7000,
        reload=True,
        log_level="info",
        reload_dirs=["app"]
    )
