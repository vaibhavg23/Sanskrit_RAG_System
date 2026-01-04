"""
Performance logging utilities
"""

import json
import os
from datetime import datetime
from typing import Dict, List

class PerformanceLogger:
    def __init__(self, log_file: str = None):
        if log_file is None:
            from config import LOG_DIR
            self.log_file = os.path.join(LOG_DIR, "performance_log.json")
        else:
            self.log_file = log_file
        
        self.logs = []
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Load existing logs if file exists"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
            except:
                self.logs = []
    
    def log_query(self, query: str, latency: float, num_docs: int, success: bool = True):
        """Log a query and its performance metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'latency_seconds': round(latency, 3),
            'num_docs_retrieved': num_docs,
            'success': success
        }
        self.logs.append(log_entry)
        self._save_logs()
    
    def _save_logs(self):
        """Save logs to file"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.logs:
            return {
                'total_queries': 0,
                'average_latency': 0,
                'success_rate': 0
            }
        
        successful = [log for log in self.logs if log.get('success', True)]
        latencies = [log['latency_seconds'] for log in successful]
        
        return {
            'total_queries': len(self.logs),
            'successful_queries': len(successful),
            'average_latency': sum(latencies) / len(latencies) if latencies else 0,
            'min_latency': min(latencies) if latencies else 0,
            'max_latency': max(latencies) if latencies else 0,
            'success_rate': len(successful) / len(self.logs) * 100 if self.logs else 0
        }
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        print("\n📊 Performance Statistics:")
        print("="*50)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Successful Queries: {stats['successful_queries']}")
        print(f"Average Latency: {stats['average_latency']:.2f}s")
        print(f"Min Latency: {stats['min_latency']:.2f}s")
        print(f"Max Latency: {stats['max_latency']:.2f}s")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print("="*50)


# Test logger
if __name__ == "__main__":
    logger = PerformanceLogger()
    
    # Log some test queries
    logger.log_query("Test query 1", 2.5, 3)
    logger.log_query("Test query 2", 3.1, 3)
    
    # Print statistics
    logger.print_statistics()