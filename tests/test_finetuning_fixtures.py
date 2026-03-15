#!/usr/bin/env python3
"""Minimal Finetuning Fixtures for test collection"""
import tempfile
import os
from dataclasses import dataclass

@dataclass
class FinetuningDataset:
    path: str
    is_valid: bool = True
    size_mb: float = 1.0

@dataclass
class FinetuningModel:
    name: str
    base_model: str

@dataclass
class FinetuningConfig:
    learning_rate: float = 1e-5
    batch_size: int = 32
    epochs: int = 3

class FinetuningFixtures:
    """Minimal fixtures for finetuning tests"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def create_valid_dataset(self, name="valid_dataset.jsonl") -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w") as f:
            f.write('{"prompt": "test", "completion": "test"}\n')
        return path
    
    def create_invalid_dataset(self) -> str:
        path = os.path.join(self.temp_dir, "invalid_dataset.jsonl")
        with open(path, "w") as f:
            f.write("invalid json content\nnot valid")
        return path
    
    def create_empty_dataset(self) -> str:
        path = os.path.join(self.temp_dir, "empty_dataset.jsonl")
        with open(path, "w") as f:
            pass
        return path
    
    def create_test_model(self, name="test_model") -> FinetuningModel:
        return FinetuningModel(name=name, base_model="gpt-2")
    
    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
