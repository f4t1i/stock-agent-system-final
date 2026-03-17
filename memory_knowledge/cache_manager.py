import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, db_path="/home/deepall/deepall_implementation/cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            response TEXT,
            embedding BLOB,
            created_at TIMESTAMP,
            last_used TIMESTAMP,
            use_count INTEGER DEFAULT 1
        )
        """)
        self.conn.commit()
        
    def get(self, query):
        query_hash = self._hash_query(query)
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT response, created_at FROM response_cache WHERE query_hash = ?", 
            (query_hash,)
        )
        result = cursor.fetchone()
        
        if result:
            response, created_at = result
            # Update stats
            cursor.execute(
                "UPDATE response_cache SET last_used = ?, use_count = use_count + 1 WHERE query_hash = ?",
                (datetime.now().isoformat(), query_hash)
            )
            self.conn.commit()
            logger.info(f"Cache hit for query hash {query_hash[:8]}")
            return json.loads(response)
        logger.info(f"Cache miss for query hash {query_hash[:8]}")
        return None
        
    def store(self, query, response, ttl_days=30):
        query_hash = self._hash_query(query)
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT OR REPLACE INTO response_cache (query_hash, query, response, created_at, last_used) VALUES (?, ?, ?, ?, ?)",
            (query_hash, query, json.dumps(response), now, now)
        )
        self.conn.commit()
        logger.info(f"Stored response for query hash {query_hash[:8]}")
        
    def _hash_query(self, query):
        return hashlib.md5(query.encode('utf-8')).hexdigest()
        
    def cleanup_old_entries(self, days=30):
        # Remove entries older than X days
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute(
            "DELETE FROM response_cache WHERE last_used < ?", 
            (cutoff,)
        )
        deleted = cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleaned up {deleted} old cache entries")
        return deleted
        
    def get_stats(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(use_count) FROM response_cache")
        count, total_use = cursor.fetchone()
        return {
            "total_entries": count,
            "total_uses": total_use or 0
        }

if __name__ == "__main__":
    # Simple test
    cache = CacheManager()
    test_query = "What is the capital of France?"
    test_response = {"answer": "Paris", "confidence": 0.99}
    
    # Store
    cache.store(test_query, test_response)
    
    # Retrieve
    cached = cache.get(test_query)
    print("Cached result:", cached)
    
    # Stats
    print("Cache stats:", cache.get_stats())
