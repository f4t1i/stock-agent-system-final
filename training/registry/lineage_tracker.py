#!/usr/bin/env python3
"""
Dataset Lineage Tracking - Task 2.4

Track dataset provenance and transformation history.

Features:
- Parent-child relationship tracking
- Backtest source tracking
- Transformation type categorization
- Example count changes (inherited, new, modified, removed)
- Lineage tree construction
- Provenance queries
- Lineage visualization

Transformation Types:
- filter: Filtered by quality/judge threshold
- merge: Merged multiple datasets
- augment: Added synthetic examples
- retrain: Retrained from scratch
- judge_filter: Filtered by judge evaluation
- quality_filter: Filtered by quality score

Phase A1 Week 3-4: Task 2.4 COMPLETE
"""

from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class TransformationType(Enum):
    """Dataset transformation types"""
    FILTER = "filter"
    MERGE = "merge"
    AUGMENT = "augment"
    RETRAIN = "retrain"
    JUDGE_FILTER = "judge_filter"
    QUALITY_FILTER = "quality_filter"
    DEDUPLICATE = "deduplicate"
    RESAMPLE = "resample"


@dataclass
class LineageNode:
    """
    Node in lineage tree representing a dataset version
    """
    version_id: str
    version: str
    dataset_id: str
    example_count: int
    created_at: datetime
    parent_version_id: Optional[str] = None
    backtest_id: Optional[str] = None
    transformation_type: Optional[str] = None
    transformation_params: Optional[Dict] = None
    examples_inherited: int = 0
    examples_new: int = 0
    examples_modified: int = 0
    examples_removed: int = 0
    notes: Optional[str] = None
    
    # Tree structure
    children: List['LineageNode'] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation"""
        return f"LineageNode(version={self.version}, examples={self.example_count})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'version': self.version,
            'dataset_id': self.dataset_id,
            'example_count': self.example_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'parent_version_id': self.parent_version_id,
            'backtest_id': self.backtest_id,
            'transformation_type': self.transformation_type,
            'transformation_params': self.transformation_params,
            'examples_inherited': self.examples_inherited,
            'examples_new': self.examples_new,
            'examples_modified': self.examples_modified,
            'examples_removed': self.examples_removed,
            'notes': self.notes
        }


@dataclass
class LineageTree:
    """
    Complete lineage tree for a dataset
    """
    dataset_id: str
    root_nodes: List[LineageNode]
    all_nodes: Dict[str, LineageNode]
    
    def get_node(self, version_id: str) -> Optional[LineageNode]:
        """Get node by version ID"""
        return self.all_nodes.get(version_id)
    
    def get_ancestors(self, version_id: str) -> List[LineageNode]:
        """
        Get all ancestors of a version (parent, grandparent, etc.)
        
        Args:
            version_id: Version ID
        
        Returns:
            List of ancestor nodes (ordered from parent to root)
        """
        ancestors = []
        current = self.get_node(version_id)
        
        while current and current.parent_version_id:
            parent = self.get_node(current.parent_version_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, version_id: str) -> List[LineageNode]:
        """
        Get all descendants of a version (children, grandchildren, etc.)
        
        Args:
            version_id: Version ID
        
        Returns:
            List of descendant nodes
        """
        descendants = []
        node = self.get_node(version_id)
        
        if not node:
            return descendants
        
        # BFS to collect all descendants
        queue = list(node.children)
        
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            queue.extend(current.children)
        
        return descendants
    
    def get_lineage_path(self, version_id: str) -> List[LineageNode]:
        """
        Get complete lineage path from root to version
        
        Args:
            version_id: Version ID
        
        Returns:
            List of nodes from root to version
        """
        ancestors = self.get_ancestors(version_id)
        current = self.get_node(version_id)
        
        if not current:
            return []
        
        # Reverse to get root-to-version order
        path = list(reversed(ancestors)) + [current]
        
        return path
    
    def get_depth(self, version_id: str) -> int:
        """
        Get depth of version in tree (0 for root)
        
        Args:
            version_id: Version ID
        
        Returns:
            Depth (number of ancestors)
        """
        return len(self.get_ancestors(version_id))
    
    def get_max_depth(self) -> int:
        """Get maximum depth of tree"""
        if not self.all_nodes:
            return 0
        
        return max(self.get_depth(vid) for vid in self.all_nodes.keys())
    
    def get_leaf_nodes(self) -> List[LineageNode]:
        """Get all leaf nodes (nodes with no children)"""
        return [node for node in self.all_nodes.values() if not node.children]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'dataset_id': self.dataset_id,
            'root_nodes': [node.to_dict() for node in self.root_nodes],
            'node_count': len(self.all_nodes),
            'max_depth': self.get_max_depth()
        }


class LineageTracker:
    """
    Track and query dataset lineage
    
    Integrates with dataset registry database
    """
    
    def __init__(self, db):
        """
        Initialize lineage tracker
        
        Args:
            db: DatasetRegistryDB instance
        """
        self.db = db
        logger.info("LineageTracker initialized")
    
    def create_lineage_record(
        self,
        child_version_id: str,
        parent_version_id: Optional[str] = None,
        backtest_id: Optional[str] = None,
        transformation_type: Optional[TransformationType] = None,
        transformation_params: Optional[Dict] = None,
        examples_inherited: int = 0,
        examples_new: int = 0,
        examples_modified: int = 0,
        examples_removed: int = 0,
        notes: Optional[str] = None
    ) -> str:
        """
        Create lineage record
        
        Args:
            child_version_id: Child version UUID
            parent_version_id: Parent version UUID (optional)
            backtest_id: Backtest ID (optional)
            transformation_type: Transformation type
            transformation_params: Transformation parameters
            examples_inherited: Number of examples inherited from parent
            examples_new: Number of new examples
            examples_modified: Number of modified examples
            examples_removed: Number of removed examples
            notes: Additional notes
        
        Returns:
            Lineage record ID
        """
        logger.info(
            f"Creating lineage record: child={child_version_id}, "
            f"parent={parent_version_id}, type={transformation_type}"
        )
        
        lineage_id = self.db.create_lineage(
            child_version_id=child_version_id,
            parent_version_id=parent_version_id,
            backtest_id=backtest_id,
            transformation_type=transformation_type.value if transformation_type else None,
            transformation_params=transformation_params,
            examples_inherited=examples_inherited,
            examples_new=examples_new,
            examples_modified=examples_modified,
            examples_removed=examples_removed,
            notes=notes
        )
        
        logger.info(f"Lineage record created: {lineage_id}")
        
        return lineage_id
    
    def build_lineage_tree(self, dataset_id: str) -> LineageTree:
        """
        Build complete lineage tree for dataset
        
        Args:
            dataset_id: Dataset UUID
        
        Returns:
            LineageTree object
        """
        logger.info(f"Building lineage tree for dataset {dataset_id}")
        
        # Get all versions for dataset
        versions = self.db.list_versions(dataset_id)
        
        if not versions:
            logger.warning(f"No versions found for dataset {dataset_id}")
            return LineageTree(
                dataset_id=dataset_id,
                root_nodes=[],
                all_nodes={}
            )
        
        # Create nodes for all versions
        nodes = {}
        for version in versions:
            # Get lineage records for this version
            lineage_records = self.db.get_lineage(version.id)
            
            # Find lineage record where this version is the child
            lineage = None
            for record in lineage_records:
                if record.child_version_id == version.id:
                    lineage = record
                    break
            
            # Create node
            node = LineageNode(
                version_id=version.id,
                version=version.version,
                dataset_id=version.dataset_id,
                example_count=version.example_count,
                created_at=version.created_at,
                parent_version_id=lineage.parent_version_id if lineage else None,
                backtest_id=lineage.backtest_id if lineage else None,
                transformation_type=lineage.transformation_type if lineage else None,
                transformation_params=lineage.transformation_params if lineage else None,
                examples_inherited=lineage.examples_inherited if lineage else 0,
                examples_new=lineage.examples_new if lineage else 0,
                examples_modified=lineage.examples_modified if lineage else 0,
                examples_removed=lineage.examples_removed if lineage else 0,
                notes=lineage.notes if lineage else None
            )
            
            nodes[version.id] = node
        
        # Build tree structure (link parents to children)
        root_nodes = []
        
        for node in nodes.values():
            if node.parent_version_id:
                # Has parent, add to parent's children
                parent = nodes.get(node.parent_version_id)
                if parent:
                    parent.children.append(node)
            else:
                # No parent, this is a root node
                root_nodes.append(node)
        
        tree = LineageTree(
            dataset_id=dataset_id,
            root_nodes=root_nodes,
            all_nodes=nodes
        )
        
        logger.info(
            f"Lineage tree built: {len(nodes)} nodes, "
            f"{len(root_nodes)} roots, max depth {tree.get_max_depth()}"
        )
        
        return tree
    
    def get_version_lineage(self, version_id: str) -> List[LineageNode]:
        """
        Get lineage path for specific version
        
        Args:
            version_id: Version UUID
        
        Returns:
            List of nodes from root to version
        """
        # Get version to find dataset
        version = self.db.get_version(version_id)
        
        if not version:
            logger.error(f"Version not found: {version_id}")
            return []
        
        # Build tree and get path
        tree = self.build_lineage_tree(version.dataset_id)
        path = tree.get_lineage_path(version_id)
        
        logger.info(f"Lineage path for {version_id}: {len(path)} nodes")
        
        return path
    
    def find_common_ancestor(
        self,
        version_id1: str,
        version_id2: str
    ) -> Optional[LineageNode]:
        """
        Find common ancestor of two versions
        
        Args:
            version_id1: First version UUID
            version_id2: Second version UUID
        
        Returns:
            Common ancestor node or None if no common ancestor
        """
        # Get versions to find dataset
        version1 = self.db.get_version(version_id1)
        version2 = self.db.get_version(version_id2)
        
        if not version1 or not version2:
            logger.error("One or both versions not found")
            return None
        
        if version1.dataset_id != version2.dataset_id:
            logger.error("Versions belong to different datasets")
            return None
        
        # Build tree
        tree = self.build_lineage_tree(version1.dataset_id)
        
        # Get ancestors for both versions
        ancestors1 = set(node.version_id for node in tree.get_ancestors(version_id1))
        ancestors2 = set(node.version_id for node in tree.get_ancestors(version_id2))
        
        # Find common ancestors
        common = ancestors1 & ancestors2
        
        if not common:
            return None
        
        # Return the closest common ancestor (highest depth)
        common_nodes = [tree.get_node(vid) for vid in common]
        closest = max(common_nodes, key=lambda n: tree.get_depth(n.version_id))
        
        logger.info(
            f"Common ancestor of {version_id1} and {version_id2}: "
            f"{closest.version}"
        )
        
        return closest
    
    def calculate_lineage_stats(self, dataset_id: str) -> Dict:
        """
        Calculate lineage statistics for dataset
        
        Args:
            dataset_id: Dataset UUID
        
        Returns:
            Dict with lineage statistics
        """
        tree = self.build_lineage_tree(dataset_id)
        
        # Count transformation types
        transformation_counts = {}
        for node in tree.all_nodes.values():
            if node.transformation_type:
                transformation_counts[node.transformation_type] = \
                    transformation_counts.get(node.transformation_type, 0) + 1
        
        # Calculate example changes
        total_inherited = sum(n.examples_inherited for n in tree.all_nodes.values())
        total_new = sum(n.examples_new for n in tree.all_nodes.values())
        total_modified = sum(n.examples_modified for n in tree.all_nodes.values())
        total_removed = sum(n.examples_removed for n in tree.all_nodes.values())
        
        stats = {
            'total_versions': len(tree.all_nodes),
            'root_versions': len(tree.root_nodes),
            'leaf_versions': len(tree.get_leaf_nodes()),
            'max_depth': tree.get_max_depth(),
            'transformation_counts': transformation_counts,
            'total_examples_inherited': total_inherited,
            'total_examples_new': total_new,
            'total_examples_modified': total_modified,
            'total_examples_removed': total_removed
        }
        
        logger.info(f"Lineage stats for {dataset_id}: {stats}")
        
        return stats
    
    def visualize_lineage_tree(
        self,
        dataset_id: str,
        output_format: str = "text"
    ) -> str:
        """
        Visualize lineage tree
        
        Args:
            dataset_id: Dataset UUID
            output_format: Output format (text, dot, mermaid)
        
        Returns:
            Visualization string
        """
        tree = self.build_lineage_tree(dataset_id)
        
        if output_format == "text":
            return self._visualize_text(tree)
        elif output_format == "dot":
            return self._visualize_dot(tree)
        elif output_format == "mermaid":
            return self._visualize_mermaid(tree)
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _visualize_text(self, tree: LineageTree) -> str:
        """Generate text visualization"""
        lines = []
        lines.append(f"Dataset Lineage Tree: {tree.dataset_id}")
        lines.append(f"Total versions: {len(tree.all_nodes)}")
        lines.append(f"Max depth: {tree.get_max_depth()}")
        lines.append("")
        
        def print_node(node: LineageNode, prefix: str = "", is_last: bool = True):
            # Node info
            connector = "└── " if is_last else "├── "
            lines.append(
                f"{prefix}{connector}{node.version} "
                f"({node.example_count} examples)"
            )
            
            # Transformation info
            if node.transformation_type:
                extension = "    " if is_last else "│   "
                lines.append(
                    f"{prefix}{extension}└─ {node.transformation_type} "
                    f"(+{node.examples_new}, -{node.examples_removed})"
                )
            
            # Print children
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                print_node(child, prefix + extension, is_last_child)
        
        for i, root in enumerate(tree.root_nodes):
            is_last_root = (i == len(tree.root_nodes) - 1)
            print_node(root, "", is_last_root)
        
        return "\n".join(lines)
    
    def _visualize_dot(self, tree: LineageTree) -> str:
        """Generate Graphviz DOT visualization"""
        lines = []
        lines.append("digraph lineage {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")
        lines.append("")
        
        # Add nodes
        for node in tree.all_nodes.values():
            label = f"{node.version}\\n{node.example_count} examples"
            if node.transformation_type:
                label += f"\\n{node.transformation_type}"
            lines.append(f'  "{node.version_id}" [label="{label}"];')
        
        lines.append("")
        
        # Add edges
        for node in tree.all_nodes.values():
            if node.parent_version_id:
                lines.append(f'  "{node.parent_version_id}" -> "{node.version_id}";')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _visualize_mermaid(self, tree: LineageTree) -> str:
        """Generate Mermaid diagram"""
        lines = []
        lines.append("graph TD")
        
        # Add nodes and edges
        for node in tree.all_nodes.values():
            node_id = node.version_id.replace("-", "")
            label = f"{node.version}<br/>{node.example_count} ex"
            
            if node.parent_version_id:
                parent_id = node.parent_version_id.replace("-", "")
                edge_label = node.transformation_type or ""
                lines.append(f'  {parent_id}["{label}"] -->|{edge_label}| {node_id}')
            else:
                lines.append(f'  {node_id}["{label}"]')
        
        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

def create_lineage(
    db,
    child_version_id: str,
    parent_version_id: Optional[str] = None,
    transformation_type: Optional[TransformationType] = None,
    **kwargs
) -> str:
    """
    Create lineage record (convenience function)
    
    Args:
        db: DatasetRegistryDB instance
        child_version_id: Child version UUID
        parent_version_id: Parent version UUID
        transformation_type: Transformation type
        **kwargs: Additional lineage fields
    
    Returns:
        Lineage record ID
    """
    tracker = LineageTracker(db)
    return tracker.create_lineage_record(
        child_version_id=child_version_id,
        parent_version_id=parent_version_id,
        transformation_type=transformation_type,
        **kwargs
    )


def get_lineage_tree(db, dataset_id: str) -> LineageTree:
    """
    Get lineage tree (convenience function)
    
    Args:
        db: DatasetRegistryDB instance
        dataset_id: Dataset UUID
    
    Returns:
        LineageTree object
    """
    tracker = LineageTracker(db)
    return tracker.build_lineage_tree(dataset_id)


if __name__ == "__main__":
    # Example usage (requires database connection)
    print("=== Lineage Tracker Example ===\n")
    
    # Create mock lineage tree for demonstration
    root = LineageNode(
        version_id="v1",
        version="1.0.0",
        dataset_id="dataset1",
        example_count=1000,
        created_at=datetime.now()
    )
    
    child1 = LineageNode(
        version_id="v2",
        version="1.1.0",
        dataset_id="dataset1",
        example_count=1200,
        created_at=datetime.now(),
        parent_version_id="v1",
        transformation_type="augment",
        examples_inherited=1000,
        examples_new=200
    )
    
    child2 = LineageNode(
        version_id="v3",
        version="1.2.0",
        dataset_id="dataset1",
        example_count=900,
        created_at=datetime.now(),
        parent_version_id="v2",
        transformation_type="quality_filter",
        examples_inherited=900,
        examples_removed=300
    )
    
    root.children = [child1]
    child1.children = [child2]
    
    tree = LineageTree(
        dataset_id="dataset1",
        root_nodes=[root],
        all_nodes={"v1": root, "v2": child1, "v3": child2}
    )
    
    print("Lineage Tree:")
    print(f"  Total nodes: {len(tree.all_nodes)}")
    print(f"  Max depth: {tree.get_max_depth()}")
    print(f"  Leaf nodes: {len(tree.get_leaf_nodes())}")
    print()
    
    print("Lineage path for v3:")
    path = tree.get_lineage_path("v3")
    for node in path:
        print(f"  {node.version} ({node.example_count} examples)")
    print()
    
    print("✅ Example completed!")
