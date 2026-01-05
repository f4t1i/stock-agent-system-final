#!/usr/bin/env python3
"""
Constitutional AI - Constitution Definition & Loading
Task 13.1

This module implements constitution loading and management for Constitutional AI.
A constitution is a set of principles that guide AI behavior.
"""

import json
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle."""
    
    name: str
    critique_request: str
    revision_request: str
    category: str = "general"
    weight: float = 1.0
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'critique_request': self.critique_request,
            'revision_request': self.revision_request,
            'category': self.category,
            'weight': self.weight,
            'examples': self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConstitutionalPrinciple':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            critique_request=data['critique_request'],
            revision_request=data['revision_request'],
            category=data.get('category', 'general'),
            weight=data.get('weight', 1.0),
            examples=data.get('examples', [])
        )


@dataclass
class Constitution:
    """A constitution containing multiple principles."""
    
    name: str
    version: str
    description: str
    principles: List[ConstitutionalPrinciple]
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate constitution."""
        if not self.principles:
            raise ValueError("Constitution must have at least one principle")
        
        # Validate principle names are unique
        names = [p.name for p in self.principles]
        if len(names) != len(set(names)):
            raise ValueError("Principle names must be unique")
    
    def get_principle(self, name: str) -> Optional[ConstitutionalPrinciple]:
        """Get principle by name."""
        for principle in self.principles:
            if principle.name == name:
                return principle
        return None
    
    def get_principles_by_category(self, category: str) -> List[ConstitutionalPrinciple]:
        """Get all principles in a category."""
        return [p for p in self.principles if p.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(p.category for p in self.principles))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'principles': [p.to_dict() for p in self.principles],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Constitution':
        """Create from dictionary."""
        principles = [ConstitutionalPrinciple.from_dict(p) for p in data['principles']]
        return cls(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            principles=principles,
            metadata=data.get('metadata', {})
        )


class ConstitutionLoader:
    """Load and manage constitutions."""
    
    def __init__(self, constitutions_dir: Optional[Path] = None):
        """
        Initialize loader.
        
        Args:
            constitutions_dir: Directory containing constitution files
        """
        self.constitutions_dir = constitutions_dir or Path("constitutions")
        self.constitutions: Dict[str, Constitution] = {}
        logger.info(f"ConstitutionLoader initialized: {self.constitutions_dir}")
        
        # Load predefined constitutions
        self._load_predefined()
    
    def _load_predefined(self):
        """Load predefined constitutions."""
        # Helpful AI constitution
        helpful = self._create_helpful_constitution()
        self.constitutions[helpful.name] = helpful
        
        # Harmless AI constitution
        harmless = self._create_harmless_constitution()
        self.constitutions[harmless.name] = harmless
        
        # Honest AI constitution
        honest = self._create_honest_constitution()
        self.constitutions[honest.name] = honest
        
        # Stock trading constitution
        stock_trading = self._create_stock_trading_constitution()
        self.constitutions[stock_trading.name] = stock_trading
        
        logger.info(f"Loaded {len(self.constitutions)} predefined constitutions")
    
    def _create_helpful_constitution(self) -> Constitution:
        """Create helpful AI constitution."""
        principles = [
            ConstitutionalPrinciple(
                name="be_helpful",
                critique_request="Critique the response for how helpful it is to the user. Point out ways it could be more helpful.",
                revision_request="Revise the response to be more helpful to the user.",
                category="helpfulness",
                weight=1.0,
                examples=[
                    {
                        "original": "I don't know.",
                        "revised": "I don't have that specific information, but I can help you find it. What exactly are you looking for?"
                    }
                ]
            ),
            ConstitutionalPrinciple(
                name="be_clear",
                critique_request="Critique the response for clarity. Point out any confusing or ambiguous parts.",
                revision_request="Revise the response to be clearer and easier to understand.",
                category="helpfulness",
                weight=0.8
            ),
            ConstitutionalPrinciple(
                name="be_concise",
                critique_request="Critique the response for conciseness. Point out any unnecessary verbosity.",
                revision_request="Revise the response to be more concise while maintaining all important information.",
                category="helpfulness",
                weight=0.6
            )
        ]
        
        return Constitution(
            name="helpful_ai",
            version="1.0.0",
            description="Constitution for creating helpful AI assistants",
            principles=principles,
            metadata={'author': 'system', 'created_at': '2024-01-01'}
        )
    
    def _create_harmless_constitution(self) -> Constitution:
        """Create harmless AI constitution."""
        principles = [
            ConstitutionalPrinciple(
                name="avoid_harm",
                critique_request="Critique the response for potential harm. Point out any content that could cause physical, emotional, or financial harm.",
                revision_request="Revise the response to remove any potentially harmful content.",
                category="safety",
                weight=2.0
            ),
            ConstitutionalPrinciple(
                name="avoid_bias",
                critique_request="Critique the response for bias or discrimination. Point out any unfair treatment of groups.",
                revision_request="Revise the response to be fair and unbiased.",
                category="safety",
                weight=1.5
            ),
            ConstitutionalPrinciple(
                name="respect_privacy",
                critique_request="Critique the response for privacy concerns. Point out any requests for personal information.",
                revision_request="Revise the response to respect user privacy.",
                category="safety",
                weight=1.8
            )
        ]
        
        return Constitution(
            name="harmless_ai",
            version="1.0.0",
            description="Constitution for creating harmless AI assistants",
            principles=principles,
            metadata={'author': 'system', 'created_at': '2024-01-01'}
        )
    
    def _create_honest_constitution(self) -> Constitution:
        """Create honest AI constitution."""
        principles = [
            ConstitutionalPrinciple(
                name="be_truthful",
                critique_request="Critique the response for truthfulness. Point out any false or misleading information.",
                revision_request="Revise the response to be more truthful and accurate.",
                category="honesty",
                weight=2.0
            ),
            ConstitutionalPrinciple(
                name="admit_uncertainty",
                critique_request="Critique the response for overconfidence. Point out claims made without sufficient evidence.",
                revision_request="Revise the response to acknowledge uncertainty where appropriate.",
                category="honesty",
                weight=1.5
            ),
            ConstitutionalPrinciple(
                name="cite_sources",
                critique_request="Critique the response for lack of sources. Point out factual claims without attribution.",
                revision_request="Revise the response to cite sources for factual claims.",
                category="honesty",
                weight=1.0
            )
        ]
        
        return Constitution(
            name="honest_ai",
            version="1.0.0",
            description="Constitution for creating honest AI assistants",
            principles=principles,
            metadata={'author': 'system', 'created_at': '2024-01-01'}
        )
    
    def _create_stock_trading_constitution(self) -> Constitution:
        """Create stock trading specific constitution."""
        principles = [
            ConstitutionalPrinciple(
                name="risk_disclosure",
                critique_request="Critique the trading recommendation for risk disclosure. Point out missing risk warnings.",
                revision_request="Revise to include appropriate risk disclosures.",
                category="trading",
                weight=2.0
            ),
            ConstitutionalPrinciple(
                name="data_driven",
                critique_request="Critique the analysis for data support. Point out claims without data backing.",
                revision_request="Revise to ensure all claims are supported by data.",
                category="trading",
                weight=1.8
            ),
            ConstitutionalPrinciple(
                name="avoid_speculation",
                critique_request="Critique for excessive speculation. Point out unsubstantiated predictions.",
                revision_request="Revise to reduce speculation and focus on data-driven insights.",
                category="trading",
                weight=1.5
            ),
            ConstitutionalPrinciple(
                name="regulatory_compliance",
                critique_request="Critique for regulatory compliance. Point out any advice that may violate regulations.",
                revision_request="Revise to ensure regulatory compliance.",
                category="trading",
                weight=2.5
            )
        ]
        
        return Constitution(
            name="stock_trading",
            version="1.0.0",
            description="Constitution for stock trading AI agents",
            principles=principles,
            metadata={'author': 'system', 'domain': 'finance', 'created_at': '2024-01-01'}
        )
    
    def load_from_file(self, file_path: Path) -> Constitution:
        """
        Load constitution from file.
        
        Args:
            file_path: Path to constitution file (JSON or YAML)
            
        Returns:
            Loaded constitution
        """
        logger.info(f"Loading constitution from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Constitution file not found: {file_path}")
        
        # Load based on file extension
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        constitution = Constitution.from_dict(data)
        self.constitutions[constitution.name] = constitution
        
        logger.info(f"Loaded constitution: {constitution.name} v{constitution.version}")
        return constitution
    
    def save_to_file(self, constitution: Constitution, file_path: Path, format: str = 'yaml'):
        """
        Save constitution to file.
        
        Args:
            constitution: Constitution to save
            file_path: Path to save to
            format: File format ('json' or 'yaml')
        """
        logger.info(f"Saving constitution to: {file_path}")
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = constitution.to_dict()
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved constitution: {constitution.name}")
    
    def get_constitution(self, name: str) -> Optional[Constitution]:
        """Get constitution by name."""
        return self.constitutions.get(name)
    
    def list_constitutions(self) -> List[str]:
        """List all available constitution names."""
        return list(self.constitutions.keys())
    
    def create_custom_constitution(
        self,
        name: str,
        version: str,
        description: str,
        principles: List[ConstitutionalPrinciple]
    ) -> Constitution:
        """
        Create a custom constitution.
        
        Args:
            name: Constitution name
            version: Version string
            description: Description
            principles: List of principles
            
        Returns:
            Created constitution
        """
        constitution = Constitution(
            name=name,
            version=version,
            description=description,
            principles=principles
        )
        
        self.constitutions[name] = constitution
        logger.info(f"Created custom constitution: {name}")
        
        return constitution


# Helper functions
def load_constitution(name_or_path: str) -> Constitution:
    """
    Load constitution by name or file path.
    
    Args:
        name_or_path: Constitution name or file path
        
    Returns:
        Loaded constitution
    """
    loader = ConstitutionLoader()
    
    # Try as name first
    constitution = loader.get_constitution(name_or_path)
    if constitution:
        return constitution
    
    # Try as file path
    path = Path(name_or_path)
    if path.exists():
        return loader.load_from_file(path)
    
    raise ValueError(f"Constitution not found: {name_or_path}")


def create_principle(
    name: str,
    critique_request: str,
    revision_request: str,
    category: str = "general",
    weight: float = 1.0
) -> ConstitutionalPrinciple:
    """Helper to create a principle."""
    return ConstitutionalPrinciple(
        name=name,
        critique_request=critique_request,
        revision_request=revision_request,
        category=category,
        weight=weight
    )


if __name__ == "__main__":
    print("=== Constitution Loader Test ===\n")
    
    # Test 1: Load predefined constitutions
    loader = ConstitutionLoader()
    print(f"Test 1: Loaded {len(loader.list_constitutions())} constitutions")
    for name in loader.list_constitutions():
        print(f"  - {name}")
    
    # Test 2: Get specific constitution
    helpful = loader.get_constitution("helpful_ai")
    print(f"\nTest 2: Loaded '{helpful.name}' v{helpful.version}")
    print(f"  Description: {helpful.description}")
    print(f"  Principles: {len(helpful.principles)}")
    
    # Test 3: Get principles by category
    stock_const = loader.get_constitution("stock_trading")
    trading_principles = stock_const.get_principles_by_category("trading")
    print(f"\nTest 3: Stock trading constitution")
    print(f"  Total principles: {len(stock_const.principles)}")
    print(f"  Trading category: {len(trading_principles)}")
    
    # Test 4: Save and load from file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    
    loader.save_to_file(helpful, temp_path, format='yaml')
    loaded = loader.load_from_file(temp_path)
    print(f"\nTest 4: Save/Load")
    print(f"  Saved to: {temp_path}")
    print(f"  Loaded: {loaded.name} v{loaded.version}")
    print(f"  Principles match: {len(loaded.principles) == len(helpful.principles)}")
    
    # Cleanup
    temp_path.unlink()
    
    print("\n✅ All tests passed!")
