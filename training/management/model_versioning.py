#!/usr/bin/env python3
"""
Model Versioning & Lineage - Task 7.4
Track model versions and lineage relationships.
Phase A1 Week 5-6: Task 7.4 COMPLETE
"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class ModelVersion:
    """Model version record"""
    model_id: str
    model_name: str
    version: str
    base_model: str
    provider: str
    job_id: str
    parent_model_id: Optional[str] = None
    status: str = 'active'  # active, deprecated, archived
    final_loss: Optional[float] = None
    trained_tokens: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    deprecated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class LineageNode:
    """Lineage tree node"""
    model: ModelVersion
    children: List['LineageNode'] = field(default_factory=list)
    depth: int = 0
    
    def __str__(self) -> str:
        indent = "  " * self.depth
        result = f"{indent}{self.model.model_name} (v{self.model.version})\n"
        for child in self.children:
            result += str(child)
        return result


class ModelRegistry:
    """Model version registry"""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        logger.info("ModelRegistry initialized")
    
    def register_model(self, model: ModelVersion) -> str:
        """Register new model version"""
        self.models[model.model_id] = model
        logger.info(f"✓ Model registered: {model.model_name} v{model.version}")
        return model.model_id
    
    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_model_by_name(self, model_name: str) -> Optional[ModelVersion]:
        """Get model by name"""
        for model in self.models.values():
            if model.model_name == model_name:
                return model
        return None
    
    def list_models(
        self,
        provider: Optional[str] = None,
        status: str = 'active'
    ) -> List[ModelVersion]:
        """List models"""
        models = list(self.models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        if status:
            models = [m for m in models if m.status == status]
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def deprecate_model(self, model_id: str):
        """Deprecate model"""
        if model_id in self.models:
            self.models[model_id].status = 'deprecated'
            self.models[model_id].deprecated_at = datetime.now()
            logger.info(f"✓ Model deprecated: {model_id}")
    
    def get_lineage(self, model_id: str) -> Optional[LineageNode]:
        """Build lineage tree from model"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        # Build tree
        node = LineageNode(model=model, depth=0)
        self._build_lineage_tree(node)
        return node
    
    def _build_lineage_tree(self, node: LineageNode):
        """Recursively build lineage tree"""
        # Find children
        for model in self.models.values():
            if model.parent_model_id == node.model.model_id:
                child_node = LineageNode(
                    model=model,
                    depth=node.depth + 1
                )
                node.children.append(child_node)
                self._build_lineage_tree(child_node)
    
    def get_ancestors(self, model_id: str) -> List[ModelVersion]:
        """Get all ancestors of a model"""
        ancestors = []
        current = self.get_model(model_id)
        
        while current and current.parent_model_id:
            parent = self.get_model(current.parent_model_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def compare_versions(
        self,
        model_id1: str,
        model_id2: str
    ) -> Dict:
        """Compare two model versions"""
        model1 = self.get_model(model_id1)
        model2 = self.get_model(model_id2)
        
        if not model1 or not model2:
            return {}
        
        return {
            'model1': {
                'name': model1.model_name,
                'version': model1.version,
                'loss': model1.final_loss,
                'tokens': model1.trained_tokens
            },
            'model2': {
                'name': model2.model_name,
                'version': model2.version,
                'loss': model2.final_loss,
                'tokens': model2.trained_tokens
            },
            'loss_improvement': (model1.final_loss - model2.final_loss) if model1.final_loss and model2.final_loss else None
        }


if __name__ == "__main__":
    print("=== Model Versioning Test ===\n")
    
    # Test 1: Create registry
    print("Test 1: Create registry")
    registry = ModelRegistry()
    print(f"✓ Registry created\n")
    
    # Test 2: Register models
    print("Test 2: Register models")
    base = ModelVersion(
        model_id="m1",
        model_name="stock-agent-v1",
        version="1.0.0",
        base_model="gpt-3.5-turbo",
        provider="openai",
        job_id="job1",
        final_loss=0.5
    )
    registry.register_model(base)
    
    child = ModelVersion(
        model_id="m2",
        model_name="stock-agent-v2",
        version="2.0.0",
        base_model="gpt-3.5-turbo",
        provider="openai",
        job_id="job2",
        parent_model_id="m1",
        final_loss=0.3
    )
    registry.register_model(child)
    print(f"✓ Registered 2 models\n")
    
    # Test 3: Get lineage
    print("Test 3: Get lineage")
    lineage = registry.get_lineage("m1")
    if lineage:
        print(f"✓ Lineage tree:\n{lineage}")
    
    # Test 4: Compare versions
    print("Test 4: Compare versions")
    comparison = registry.compare_versions("m1", "m2")
    if comparison:
        print(f"✓ Loss improvement: {comparison['loss_improvement']:.2f}\n")
    
    print("=== Tests Complete ===")
