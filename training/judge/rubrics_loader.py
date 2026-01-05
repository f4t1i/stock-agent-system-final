#!/usr/bin/env python3
"""
Rubrics Loading System - Task 3.2

Load and manage evaluation rubrics from files and templates.

Features:
- Load rubrics from YAML/JSON files
- Predefined rubric templates for different agent types
- Rubric validation
- Rubric registry/catalog
- Custom rubric creation
- Rubric versioning

Phase A1 Week 3-4: Task 3.2 COMPLETE
"""

import os
import json
import yaml
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from loguru import logger

from training.judge.llm_judge import JudgeRubric


@dataclass
class RubricTemplate:
    """
    Rubric template metadata
    """
    name: str
    agent_type: str
    version: str
    description: str
    rubrics: List[JudgeRubric]
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'agent_type': self.agent_type,
            'version': self.version,
            'description': self.description,
            'rubrics': [r.to_dict() for r in self.rubrics],
            'created_at': self.created_at,
            'created_by': self.created_by,
            'tags': self.tags
        }


class RubricsLoader:
    """
    Load and manage evaluation rubrics
    """
    
    def __init__(self, rubrics_dir: Optional[str] = None):
        """
        Initialize rubrics loader
        
        Args:
            rubrics_dir: Directory containing rubric files (optional)
        """
        self.rubrics_dir = rubrics_dir
        self.templates: Dict[str, RubricTemplate] = {}
        
        # Load predefined templates
        self._load_predefined_templates()
        
        # Load from directory if provided
        if rubrics_dir and os.path.exists(rubrics_dir):
            self.load_from_directory(rubrics_dir)
        
        logger.info(f"RubricsLoader initialized with {len(self.templates)} templates")
    
    def _load_predefined_templates(self):
        """Load predefined rubric templates"""
        
        # Technical Analysis Template
        technical_rubrics = [
            JudgeRubric(
                criterion="technical_accuracy",
                description="Accuracy of technical indicators and analysis",
                weight=3.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="data_interpretation",
                description="Correct interpretation of price data and patterns",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="reasoning_quality",
                description="Clarity and logic of technical reasoning",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="completeness",
                description="Coverage of relevant technical aspects",
                weight=1.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="actionability",
                description="Clarity of trading signals and recommendations",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            )
        ]
        
        self.templates["technical_v1"] = RubricTemplate(
            name="technical_v1",
            agent_type="technical",
            version="1.0.0",
            description="Technical analysis evaluation rubric",
            rubrics=technical_rubrics,
            tags=["technical", "trading", "analysis"]
        )
        
        # News Analysis Template
        news_rubrics = [
            JudgeRubric(
                criterion="relevance",
                description="Relevance of news to stock/market",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="sentiment_accuracy",
                description="Accuracy of sentiment analysis",
                weight=3.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="impact_assessment",
                description="Quality of market impact assessment",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="context_understanding",
                description="Understanding of broader market context",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="clarity",
                description="Clarity of explanation and conclusions",
                weight=1.5,
                min_score=0.0,
                max_score=1.0
            )
        ]
        
        self.templates["news_v1"] = RubricTemplate(
            name="news_v1",
            agent_type="news",
            version="1.0.0",
            description="News analysis evaluation rubric",
            rubrics=news_rubrics,
            tags=["news", "sentiment", "analysis"]
        )
        
        # Fundamental Analysis Template
        fundamental_rubrics = [
            JudgeRubric(
                criterion="financial_accuracy",
                description="Accuracy of financial metrics and ratios",
                weight=3.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="valuation_quality",
                description="Quality of valuation analysis",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="business_understanding",
                description="Understanding of business model and industry",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="risk_assessment",
                description="Identification and assessment of risks",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="reasoning_depth",
                description="Depth and thoroughness of analysis",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            )
        ]
        
        self.templates["fundamental_v1"] = RubricTemplate(
            name="fundamental_v1",
            agent_type="fundamental",
            version="1.0.0",
            description="Fundamental analysis evaluation rubric",
            rubrics=fundamental_rubrics,
            tags=["fundamental", "valuation", "analysis"]
        )
        
        # General Quality Template
        general_rubrics = [
            JudgeRubric(
                criterion="accuracy",
                description="Overall accuracy of analysis",
                weight=2.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="relevance",
                description="Relevance to the query",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="completeness",
                description="Completeness of response",
                weight=1.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="clarity",
                description="Clarity of explanation",
                weight=1.5,
                min_score=0.0,
                max_score=1.0
            ),
            JudgeRubric(
                criterion="reasoning_quality",
                description="Quality of reasoning",
                weight=2.0,
                min_score=0.0,
                max_score=1.0
            )
        ]
        
        self.templates["general_v1"] = RubricTemplate(
            name="general_v1",
            agent_type="general",
            version="1.0.0",
            description="General quality evaluation rubric",
            rubrics=general_rubrics,
            tags=["general", "quality"]
        )
        
        logger.info(f"Loaded {len(self.templates)} predefined templates")
    
    def load_from_file(self, file_path: str) -> RubricTemplate:
        """
        Load rubric template from file
        
        Args:
            file_path: Path to rubric file (YAML or JSON)
        
        Returns:
            RubricTemplate object
        """
        logger.info(f"Loading rubric from file: {file_path}")
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Rubric file not found: {file_path}")
        
        # Load file
        with open(file_path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate and parse
        template = self._parse_rubric_data(data)
        
        # Add to registry
        self.templates[template.name] = template
        
        logger.info(f"Loaded template: {template.name} ({len(template.rubrics)} rubrics)")
        
        return template
    
    def load_from_directory(self, directory: str):
        """
        Load all rubric templates from directory
        
        Args:
            directory: Directory path
        """
        logger.info(f"Loading rubrics from directory: {directory}")
        
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return
        
        # Find all YAML and JSON files
        files = list(dir_path.glob("*.yaml")) + \
                list(dir_path.glob("*.yml")) + \
                list(dir_path.glob("*.json"))
        
        loaded_count = 0
        
        for file_path in files:
            try:
                self.load_from_file(str(file_path))
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} templates from directory")
    
    def _parse_rubric_data(self, data: Dict) -> RubricTemplate:
        """
        Parse rubric data from dict
        
        Args:
            data: Rubric data dict
        
        Returns:
            RubricTemplate object
        """
        # Validate required fields
        required_fields = ['name', 'agent_type', 'version', 'rubrics']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Parse rubrics
        rubrics = []
        for rubric_data in data['rubrics']:
            rubric = JudgeRubric(
                criterion=rubric_data['criterion'],
                description=rubric_data['description'],
                weight=rubric_data.get('weight', 1.0),
                min_score=rubric_data.get('min_score', 0.0),
                max_score=rubric_data.get('max_score', 1.0)
            )
            rubrics.append(rubric)
        
        # Create template
        template = RubricTemplate(
            name=data['name'],
            agent_type=data['agent_type'],
            version=data['version'],
            description=data.get('description', ''),
            rubrics=rubrics,
            created_at=data.get('created_at'),
            created_by=data.get('created_by'),
            tags=data.get('tags', [])
        )
        
        return template
    
    def get_template(self, name: str) -> Optional[RubricTemplate]:
        """
        Get rubric template by name
        
        Args:
            name: Template name
        
        Returns:
            RubricTemplate object or None
        """
        return self.templates.get(name)
    
    def get_template_for_agent(
        self,
        agent_type: str,
        version: Optional[str] = None
    ) -> Optional[RubricTemplate]:
        """
        Get rubric template for agent type
        
        Args:
            agent_type: Agent type
            version: Template version (optional, uses latest)
        
        Returns:
            RubricTemplate object or None
        """
        # Find templates for agent type
        matching = [
            t for t in self.templates.values()
            if t.agent_type == agent_type
        ]
        
        if not matching:
            return None
        
        # Filter by version if specified
        if version:
            matching = [t for t in matching if t.version == version]
        
        if not matching:
            return None
        
        # Return latest version (sort by version string)
        matching.sort(key=lambda t: t.version, reverse=True)
        
        return matching[0]
    
    def list_templates(
        self,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[RubricTemplate]:
        """
        List rubric templates
        
        Args:
            agent_type: Filter by agent type (optional)
            tags: Filter by tags (optional)
        
        Returns:
            List of RubricTemplate objects
        """
        templates = list(self.templates.values())
        
        # Filter by agent type
        if agent_type:
            templates = [t for t in templates if t.agent_type == agent_type]
        
        # Filter by tags
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates
    
    def save_template(
        self,
        template: RubricTemplate,
        file_path: str,
        format: str = "yaml"
    ):
        """
        Save rubric template to file
        
        Args:
            template: RubricTemplate object
            file_path: Output file path
            format: Output format (yaml or json)
        """
        logger.info(f"Saving template to file: {file_path}")
        
        data = template.to_dict()
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif format == "json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Template saved: {file_path}")
    
    def create_custom_template(
        self,
        name: str,
        agent_type: str,
        version: str,
        description: str,
        rubrics: List[JudgeRubric],
        tags: Optional[List[str]] = None
    ) -> RubricTemplate:
        """
        Create custom rubric template
        
        Args:
            name: Template name
            agent_type: Agent type
            version: Version string
            description: Description
            rubrics: List of rubrics
            tags: Tags (optional)
        
        Returns:
            RubricTemplate object
        """
        logger.info(f"Creating custom template: {name}")
        
        template = RubricTemplate(
            name=name,
            agent_type=agent_type,
            version=version,
            description=description,
            rubrics=rubrics,
            tags=tags or []
        )
        
        # Add to registry
        self.templates[name] = template
        
        logger.info(f"Custom template created: {name} ({len(rubrics)} rubrics)")
        
        return template
    
    def validate_template(self, template: RubricTemplate) -> bool:
        """
        Validate rubric template
        
        Args:
            template: RubricTemplate object
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required fields
        if not template.name:
            raise ValueError("Template name is required")
        
        if not template.agent_type:
            raise ValueError("Agent type is required")
        
        if not template.version:
            raise ValueError("Version is required")
        
        if not template.rubrics:
            raise ValueError("At least one rubric is required")
        
        # Validate rubrics
        for rubric in template.rubrics:
            if not rubric.criterion:
                raise ValueError("Rubric criterion is required")
            
            if not rubric.description:
                raise ValueError("Rubric description is required")
            
            if rubric.min_score < 0.0:
                raise ValueError("Min score must be >= 0.0")
            
            if rubric.max_score <= rubric.min_score:
                raise ValueError("Max score must be > min score")
            
            if rubric.weight <= 0.0:
                raise ValueError("Weight must be > 0.0")
        
        return True


# ============================================================================
# Helper Functions
# ============================================================================

def load_rubrics(
    file_path: Optional[str] = None,
    template_name: Optional[str] = None,
    agent_type: Optional[str] = None
) -> List[JudgeRubric]:
    """
    Load rubrics (convenience function)
    
    Args:
        file_path: Path to rubric file (optional)
        template_name: Template name (optional)
        agent_type: Agent type (optional)
    
    Returns:
        List of JudgeRubric objects
    """
    loader = RubricsLoader()
    
    if file_path:
        template = loader.load_from_file(file_path)
        return template.rubrics
    
    if template_name:
        template = loader.get_template(template_name)
        if template:
            return template.rubrics
    
    if agent_type:
        template = loader.get_template_for_agent(agent_type)
        if template:
            return template.rubrics
    
    raise ValueError("Must provide file_path, template_name, or agent_type")


def create_rubric_template(
    name: str,
    agent_type: str,
    rubrics: List[JudgeRubric],
    version: str = "1.0.0",
    description: str = "",
    save_path: Optional[str] = None
) -> RubricTemplate:
    """
    Create and optionally save rubric template (convenience function)
    
    Args:
        name: Template name
        agent_type: Agent type
        rubrics: List of rubrics
        version: Version string
        description: Description
        save_path: Path to save template (optional)
    
    Returns:
        RubricTemplate object
    """
    loader = RubricsLoader()
    
    template = loader.create_custom_template(
        name=name,
        agent_type=agent_type,
        version=version,
        description=description,
        rubrics=rubrics
    )
    
    if save_path:
        loader.save_template(template, save_path)
    
    return template


if __name__ == "__main__":
    # Example usage
    print("=== Rubrics Loader Example ===\n")
    
    # Create loader
    loader = RubricsLoader()
    
    print(f"Loaded {len(loader.templates)} predefined templates:\n")
    
    for template in loader.list_templates():
        print(f"  {template.name} (v{template.version})")
        print(f"    Agent type: {template.agent_type}")
        print(f"    Rubrics: {len(template.rubrics)}")
        print(f"    Description: {template.description}")
        print()
    
    # Get template for agent type
    print("Getting template for 'technical' agent:")
    template = loader.get_template_for_agent("technical")
    if template:
        print(f"  Found: {template.name}")
        print(f"  Rubrics:")
        for rubric in template.rubrics:
            print(f"    - {rubric.criterion} (weight: {rubric.weight})")
    print()
    
    print("✅ Example completed!")
