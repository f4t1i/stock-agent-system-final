"""
Base Agent Class - Gemeinsame Funktionalität für alle Agenten
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime


class BaseAgent(ABC):
    """
    Abstract base class für alle Agenten im System.
    
    Definiert gemeinsame Schnittstellen und Hilfsmethoden.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Agent-Konfiguration
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.version = config.get('version', '1.0.0')
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': self.version
        }
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Hauptmethode für Agent-Analyse.
        
        Muss von Subklassen implementiert werden.
        """
        pass
    
    def validate_output(self, output: Dict[str, Any], schema: Dict) -> bool:
        """
        Validiere Output gegen Schema
        
        Args:
            output: Agent-Output
            schema: JSON-Schema
            
        Returns:
            True wenn valide
        """
        # Einfache Schema-Validierung
        # In Production: jsonschema.validate(output, schema)
        
        for required_field in schema.get('required', []):
            if required_field not in output:
                return False
        
        return True
    
    def log_interaction(
        self,
        input_data: Dict,
        output_data: Dict,
        metadata: Optional[Dict] = None
    ):
        """
        Logge Interaktion für Auditing und Replay
        
        Args:
            input_data: Input an den Agenten
            output_data: Output des Agenten
            metadata: Zusätzliche Metadaten
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'version': self.version,
            'input': input_data,
            'output': output_data,
            'metadata': metadata or {}
        }
        
        # In Production: Schreibe zu Logging-Service
        # logger.info(json.dumps(log_entry))
    
    def get_info(self) -> Dict[str, Any]:
        """Gibt Agent-Informationen zurück"""
        return {
            'name': self.name,
            'version': self.version,
            'type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
