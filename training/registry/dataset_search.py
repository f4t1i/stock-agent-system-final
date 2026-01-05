#!/usr/bin/env python3
"""
Dataset Search and Filter Queries - Task 2.5

Advanced search and filtering for dataset registry.

Features:
- Full-text search on dataset names/descriptions
- Filter by agent type, format, status
- Filter by version constraints
- Filter by quality score thresholds
- Filter by date ranges
- Complex query combinations
- Pagination support
- Sorting options

Phase A1 Week 3-4: Task 2.5 COMPLETE
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

from training.registry.semantic_version import SemanticVersion, VersionConstraint


class SortOrder(Enum):
    """Sort order"""
    ASC = "ASC"
    DESC = "DESC"


class SortField(Enum):
    """Sort field"""
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    VERSION = "version"
    EXAMPLE_COUNT = "example_count"
    QUALITY_SCORE = "avg_quality_score"
    NAME = "name"


@dataclass
class SearchFilter:
    """
    Search filter for datasets
    """
    # Text search
    query: Optional[str] = None
    
    # Dataset filters
    agent_type: Optional[str] = None
    format: Optional[str] = None
    dataset_type: Optional[str] = None
    status: Optional[str] = None
    
    # Version filters
    version_constraint: Optional[str] = None
    is_latest: Optional[bool] = None
    
    # Quality filters
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    min_judge_score: Optional[float] = None
    min_judge_pass_rate: Optional[float] = None
    
    # Example count filters
    min_example_count: Optional[int] = None
    max_example_count: Optional[int] = None
    
    # Date filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    
    # Tags filter
    tags: Optional[List[str]] = None
    
    # Pagination
    limit: Optional[int] = None
    offset: Optional[int] = 0
    
    # Sorting
    sort_by: SortField = SortField.CREATED_AT
    sort_order: SortOrder = SortOrder.DESC
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'agent_type': self.agent_type,
            'format': self.format,
            'dataset_type': self.dataset_type,
            'status': self.status,
            'version_constraint': self.version_constraint,
            'is_latest': self.is_latest,
            'min_quality_score': self.min_quality_score,
            'max_quality_score': self.max_quality_score,
            'min_judge_score': self.min_judge_score,
            'min_judge_pass_rate': self.min_judge_pass_rate,
            'min_example_count': self.min_example_count,
            'max_example_count': self.max_example_count,
            'created_after': self.created_after.isoformat() if self.created_after else None,
            'created_before': self.created_before.isoformat() if self.created_before else None,
            'updated_after': self.updated_after.isoformat() if self.updated_after else None,
            'updated_before': self.updated_before.isoformat() if self.updated_before else None,
            'tags': self.tags,
            'limit': self.limit,
            'offset': self.offset,
            'sort_by': self.sort_by.value,
            'sort_order': self.sort_order.value
        }


@dataclass
class SearchResult:
    """
    Search result
    """
    datasets: List[Dict]
    versions: List[Dict]
    total_count: int
    page_count: int
    current_page: int
    has_next: bool
    has_prev: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'datasets': self.datasets,
            'versions': self.versions,
            'total_count': self.total_count,
            'page_count': self.page_count,
            'current_page': self.current_page,
            'has_next': self.has_next,
            'has_prev': self.has_prev
        }


class DatasetSearchEngine:
    """
    Search engine for dataset registry
    
    Provides advanced search and filtering capabilities
    """
    
    def __init__(self, db):
        """
        Initialize search engine
        
        Args:
            db: DatasetRegistryDB instance
        """
        self.db = db
        logger.info("DatasetSearchEngine initialized")
    
    def search_datasets(
        self,
        search_filter: SearchFilter
    ) -> SearchResult:
        """
        Search datasets with filters
        
        Args:
            search_filter: Search filter
        
        Returns:
            SearchResult object
        """
        logger.info(f"Searching datasets with filter: {search_filter.to_dict()}")
        
        # Build SQL query
        query, params = self._build_dataset_query(search_filter)
        
        # Execute query
        results = self.db.execute_query(query, tuple(params), fetch=True)
        
        # Get total count (without pagination)
        count_query, count_params = self._build_count_query(search_filter, "datasets")
        count_result = self.db.execute_query(count_query, tuple(count_params), fetch=True)
        total_count = count_result[0]['count'] if count_result else 0
        
        # Calculate pagination info
        limit = search_filter.limit or total_count
        offset = search_filter.offset or 0
        current_page = (offset // limit) + 1 if limit > 0 else 1
        page_count = (total_count + limit - 1) // limit if limit > 0 else 1
        has_next = offset + limit < total_count
        has_prev = offset > 0
        
        logger.info(
            f"Search complete: {len(results)} results, "
            f"total {total_count}, page {current_page}/{page_count}"
        )
        
        return SearchResult(
            datasets=results,
            versions=[],
            total_count=total_count,
            page_count=page_count,
            current_page=current_page,
            has_next=has_next,
            has_prev=has_prev
        )
    
    def search_versions(
        self,
        search_filter: SearchFilter
    ) -> SearchResult:
        """
        Search dataset versions with filters
        
        Args:
            search_filter: Search filter
        
        Returns:
            SearchResult object
        """
        logger.info(f"Searching versions with filter: {search_filter.to_dict()}")
        
        # Build SQL query
        query, params = self._build_version_query(search_filter)
        
        # Execute query
        results = self.db.execute_query(query, tuple(params), fetch=True)
        
        # Get total count
        count_query, count_params = self._build_count_query(search_filter, "versions")
        count_result = self.db.execute_query(count_query, tuple(count_params), fetch=True)
        total_count = count_result[0]['count'] if count_result else 0
        
        # Calculate pagination info
        limit = search_filter.limit or total_count
        offset = search_filter.offset or 0
        current_page = (offset // limit) + 1 if limit > 0 else 1
        page_count = (total_count + limit - 1) // limit if limit > 0 else 1
        has_next = offset + limit < total_count
        has_prev = offset > 0
        
        logger.info(
            f"Search complete: {len(results)} results, "
            f"total {total_count}, page {current_page}/{page_count}"
        )
        
        return SearchResult(
            datasets=[],
            versions=results,
            total_count=total_count,
            page_count=page_count,
            current_page=current_page,
            has_next=has_next,
            has_prev=has_prev
        )
    
    def _build_dataset_query(
        self,
        search_filter: SearchFilter
    ) -> Tuple[str, List]:
        """
        Build SQL query for dataset search
        
        Args:
            search_filter: Search filter
        
        Returns:
            Tuple of (query, params)
        """
        query = "SELECT * FROM datasets WHERE is_deleted = FALSE"
        params = []
        
        # Text search (using ILIKE for case-insensitive search)
        if search_filter.query:
            query += " AND (name ILIKE %s OR description ILIKE %s)"
            search_term = f"%{search_filter.query}%"
            params.extend([search_term, search_term])
        
        # Agent type filter
        if search_filter.agent_type:
            query += " AND agent_type = %s"
            params.append(search_filter.agent_type)
        
        # Format filter
        if search_filter.format:
            query += " AND format = %s"
            params.append(search_filter.format)
        
        # Dataset type filter
        if search_filter.dataset_type:
            query += " AND dataset_type = %s"
            params.append(search_filter.dataset_type)
        
        # Status filter
        if search_filter.status:
            query += " AND status = %s"
            params.append(search_filter.status)
        
        # Example count filters
        if search_filter.min_example_count is not None:
            query += " AND total_examples >= %s"
            params.append(search_filter.min_example_count)
        
        if search_filter.max_example_count is not None:
            query += " AND total_examples <= %s"
            params.append(search_filter.max_example_count)
        
        # Date filters
        if search_filter.created_after:
            query += " AND created_at >= %s"
            params.append(search_filter.created_after)
        
        if search_filter.created_before:
            query += " AND created_at <= %s"
            params.append(search_filter.created_before)
        
        if search_filter.updated_after:
            query += " AND updated_at >= %s"
            params.append(search_filter.updated_after)
        
        if search_filter.updated_before:
            query += " AND updated_at <= %s"
            params.append(search_filter.updated_before)
        
        # Sorting
        sort_field = search_filter.sort_by.value
        sort_order = search_filter.sort_order.value
        query += f" ORDER BY {sort_field} {sort_order}"
        
        # Pagination
        if search_filter.limit:
            query += f" LIMIT {search_filter.limit}"
        
        if search_filter.offset:
            query += f" OFFSET {search_filter.offset}"
        
        return query, params
    
    def _build_version_query(
        self,
        search_filter: SearchFilter
    ) -> Tuple[str, List]:
        """
        Build SQL query for version search
        
        Args:
            search_filter: Search filter
        
        Returns:
            Tuple of (query, params)
        """
        query = """
            SELECT dv.*, d.agent_type, d.name as dataset_name
            FROM dataset_versions dv
            JOIN datasets d ON dv.dataset_id = d.id
            WHERE dv.is_deleted = FALSE AND d.is_deleted = FALSE
        """
        params = []
        
        # Text search on dataset name
        if search_filter.query:
            query += " AND d.name ILIKE %s"
            params.append(f"%{search_filter.query}%")
        
        # Agent type filter
        if search_filter.agent_type:
            query += " AND d.agent_type = %s"
            params.append(search_filter.agent_type)
        
        # Format filter
        if search_filter.format:
            query += " AND d.format = %s"
            params.append(search_filter.format)
        
        # Dataset type filter
        if search_filter.dataset_type:
            query += " AND d.dataset_type = %s"
            params.append(search_filter.dataset_type)
        
        # Is latest filter
        if search_filter.is_latest is not None:
            query += " AND dv.is_latest = %s"
            params.append(search_filter.is_latest)
        
        # Quality score filters
        if search_filter.min_quality_score is not None:
            query += " AND dv.avg_quality_score >= %s"
            params.append(search_filter.min_quality_score)
        
        if search_filter.max_quality_score is not None:
            query += " AND dv.avg_quality_score <= %s"
            params.append(search_filter.max_quality_score)
        
        # Judge score filters
        if search_filter.min_judge_score is not None:
            query += " AND dv.avg_judge_score >= %s"
            params.append(search_filter.min_judge_score)
        
        if search_filter.min_judge_pass_rate is not None:
            query += " AND dv.judge_pass_rate >= %s"
            params.append(search_filter.min_judge_pass_rate)
        
        # Example count filters
        if search_filter.min_example_count is not None:
            query += " AND dv.example_count >= %s"
            params.append(search_filter.min_example_count)
        
        if search_filter.max_example_count is not None:
            query += " AND dv.example_count <= %s"
            params.append(search_filter.max_example_count)
        
        # Tags filter (PostgreSQL array contains)
        if search_filter.tags:
            query += " AND dv.tags @> %s"
            params.append(search_filter.tags)
        
        # Date filters
        if search_filter.created_after:
            query += " AND dv.created_at >= %s"
            params.append(search_filter.created_after)
        
        if search_filter.created_before:
            query += " AND dv.created_at <= %s"
            params.append(search_filter.created_before)
        
        # Sorting
        sort_field_map = {
            SortField.CREATED_AT: "dv.created_at",
            SortField.VERSION: "dv.major_version, dv.minor_version, dv.patch_version",
            SortField.EXAMPLE_COUNT: "dv.example_count",
            SortField.QUALITY_SCORE: "dv.avg_quality_score",
            SortField.NAME: "d.name"
        }
        
        sort_field = sort_field_map.get(search_filter.sort_by, "dv.created_at")
        sort_order = search_filter.sort_order.value
        query += f" ORDER BY {sort_field} {sort_order}"
        
        # Pagination
        if search_filter.limit:
            query += f" LIMIT {search_filter.limit}"
        
        if search_filter.offset:
            query += f" OFFSET {search_filter.offset}"
        
        return query, params
    
    def _build_count_query(
        self,
        search_filter: SearchFilter,
        table: str
    ) -> Tuple[str, List]:
        """
        Build count query
        
        Args:
            search_filter: Search filter
            table: Table name (datasets or versions)
        
        Returns:
            Tuple of (query, params)
        """
        if table == "datasets":
            query = "SELECT COUNT(*) as count FROM datasets WHERE is_deleted = FALSE"
            params = []
            
            if search_filter.query:
                query += " AND (name ILIKE %s OR description ILIKE %s)"
                search_term = f"%{search_filter.query}%"
                params.extend([search_term, search_term])
            
            if search_filter.agent_type:
                query += " AND agent_type = %s"
                params.append(search_filter.agent_type)
            
            if search_filter.format:
                query += " AND format = %s"
                params.append(search_filter.format)
            
            if search_filter.status:
                query += " AND status = %s"
                params.append(search_filter.status)
        
        else:  # versions
            query = """
                SELECT COUNT(*) as count
                FROM dataset_versions dv
                JOIN datasets d ON dv.dataset_id = d.id
                WHERE dv.is_deleted = FALSE AND d.is_deleted = FALSE
            """
            params = []
            
            if search_filter.query:
                query += " AND d.name ILIKE %s"
                params.append(f"%{search_filter.query}%")
            
            if search_filter.agent_type:
                query += " AND d.agent_type = %s"
                params.append(search_filter.agent_type)
            
            if search_filter.is_latest is not None:
                query += " AND dv.is_latest = %s"
                params.append(search_filter.is_latest)
            
            if search_filter.min_quality_score is not None:
                query += " AND dv.avg_quality_score >= %s"
                params.append(search_filter.min_quality_score)
        
        return query, params
    
    def filter_versions_by_constraint(
        self,
        dataset_id: str,
        constraint: str
    ) -> List[Dict]:
        """
        Filter versions by version constraint
        
        Args:
            dataset_id: Dataset UUID
            constraint: Version constraint (e.g., "^1.2.0", ">=1.0.0")
        
        Returns:
            List of matching versions
        """
        logger.info(f"Filtering versions for {dataset_id} with constraint {constraint}")
        
        # Get all versions for dataset
        versions = self.db.list_versions(dataset_id)
        
        # Parse constraint
        version_constraint = VersionConstraint(constraint)
        
        # Filter versions
        matching = []
        for version in versions:
            try:
                sem_ver = SemanticVersion.parse(version.version)
                if version_constraint.satisfies(sem_ver):
                    matching.append(version)
            except ValueError:
                logger.warning(f"Invalid version format: {version.version}")
        
        logger.info(f"Found {len(matching)} matching versions")
        
        return matching


# ============================================================================
# Helper Functions
# ============================================================================

def search_datasets(
    db,
    query: Optional[str] = None,
    agent_type: Optional[str] = None,
    **kwargs
) -> SearchResult:
    """
    Search datasets (convenience function)
    
    Args:
        db: DatasetRegistryDB instance
        query: Text search query
        agent_type: Agent type filter
        **kwargs: Additional filter parameters
    
    Returns:
        SearchResult object
    """
    search_filter = SearchFilter(
        query=query,
        agent_type=agent_type,
        **kwargs
    )
    
    engine = DatasetSearchEngine(db)
    return engine.search_datasets(search_filter)


def search_versions(
    db,
    min_quality_score: Optional[float] = None,
    is_latest: Optional[bool] = None,
    **kwargs
) -> SearchResult:
    """
    Search versions (convenience function)
    
    Args:
        db: DatasetRegistryDB instance
        min_quality_score: Minimum quality score
        is_latest: Filter for latest versions only
        **kwargs: Additional filter parameters
    
    Returns:
        SearchResult object
    """
    search_filter = SearchFilter(
        min_quality_score=min_quality_score,
        is_latest=is_latest,
        **kwargs
    )
    
    engine = DatasetSearchEngine(db)
    return engine.search_versions(search_filter)


if __name__ == "__main__":
    # Example usage
    print("=== Dataset Search Engine Example ===\n")
    
    # Create search filter
    search_filter = SearchFilter(
        query="technical",
        agent_type="technical",
        min_quality_score=0.7,
        is_latest=True,
        limit=10,
        offset=0,
        sort_by=SortField.QUALITY_SCORE,
        sort_order=SortOrder.DESC
    )
    
    print("Search Filter:")
    print(f"  Query: {search_filter.query}")
    print(f"  Agent type: {search_filter.agent_type}")
    print(f"  Min quality: {search_filter.min_quality_score}")
    print(f"  Latest only: {search_filter.is_latest}")
    print(f"  Sort by: {search_filter.sort_by.value} {search_filter.sort_order.value}")
    print(f"  Pagination: limit={search_filter.limit}, offset={search_filter.offset}")
    print()
    
    print("✅ Example completed!")
