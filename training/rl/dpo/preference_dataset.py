#!/usr/bin/env python3
"""Preference Dataset Preparation - Task 10.1"""
from dataclasses import dataclass
from typing import List
from loguru import logger

@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str

class PreferenceDataset:
    def __init__(self):
        self.pairs = []
        logger.info("PreferenceDataset initialized")
    
    def add_pair(self, prompt, chosen, rejected):
        self.pairs.append(PreferencePair(prompt, chosen, rejected))
    
    def get_batch(self, batch_size=8):
        import random
        return random.sample(self.pairs, min(batch_size, len(self.pairs)))

if __name__ == "__main__":
    print("✓ Preference Dataset ready")
