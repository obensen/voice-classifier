# app/utils/cache.py
# Performans optimizasyonu için basit in-memory cache sistemi

from typing import Any, Optional
import time

class SimpleCache:
    """
    Basit in-memory önbellekleme sistemi
    Analysis sonuçlarını geçici olarak saklar
    """
    
    def __init__(self, default_ttl: int = 300):  # 5 dakika TTL
        """
        Cache'i başlatır
        
        Args:
            default_ttl: Varsayılan cache süresi (saniye)
        """
        self._cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str, subkey: str = None) -> Optional[Any]:
        """
        Önbellekten değer al
        
        Args:
            key: Ana cache anahtarı (genellikle dosya yolu)
            subkey: Alt anahtar (opsiyonel, analiz tipi için)
            
        Returns:
            Cached değer veya None
        """
        cache_key = f"{key}:{subkey}" if subkey else key
        
        if cache_key in self._cache:
            data, timestamp, ttl = self._cache[cache_key]
            
            # TTL kontrolü
            if time.time() - timestamp < ttl:
                return data
            else:
                # Süresi dolmuş, sil
                del self._cache[cache_key]
        
        return None
    
    def set(self, key: str, subkey: str, value: Any, ttl: Optional[int] = None):
        """
        Önbelleğe değer kaydet
        
        Args:
            key: Ana cache anahtarı
            subkey: Alt anahtar
            value: Saklanacak değer
            ttl: Cache süresi (opsiyonel)
        """
        cache_key = f"{key}:{subkey}"
        used_ttl = ttl if ttl is not None else self.default_ttl
        
        self._cache[cache_key] = (value, time.time(), used_ttl)
    
    def clear(self):
        """Tüm önbelleği temizle"""
        self._cache.clear()
    
    def cleanup_expired(self):
        """Süresi dolmuş cache girdilerini temizle"""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (data, timestamp, ttl) in self._cache.items():
            if current_time - timestamp >= ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> dict:
        """Cache istatistiklerini döndür"""
        return {
            "total_entries": len(self._cache),
            "cache_keys": list(self._cache.keys())
        }

# Global cache instance - Singleton pattern
cache = SimpleCache()

# Cache performans metrikleri için
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
    
    def hit(self):
        self.hits += 1
    
    def miss(self):
        self.misses += 1
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# Global metrics instance
metrics = CacheMetrics()