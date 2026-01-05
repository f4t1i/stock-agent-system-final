#!/usr/bin/env python3
"""
Semantic Versioning Logic - Task 2.2

Complete semantic versioning implementation following semver.org specification.

Features:
- Version parsing and validation (MAJOR.MINOR.PATCH)
- Version comparison (>, <, ==, >=, <=, !=)
- Automatic version bumping (major, minor, patch)
- Version constraints checking (^, ~, >=, etc.)
- Compatibility rules
- Pre-release and build metadata support (optional)

Specification: https://semver.org/

Phase A1 Week 3-4: Task 2.2 COMPLETE
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass
from functools import total_ordering
from loguru import logger


@dataclass
@total_ordering
class SemanticVersion:
    """
    Semantic version following semver.org specification
    
    Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    
    Examples:
        1.0.0
        1.2.3
        2.0.0-alpha
        1.0.0-beta.1
        1.0.0+20230101
        1.0.0-rc.1+build.123
    """
    
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    # Regex pattern for semantic versioning
    SEMVER_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
        r'(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )
    
    @classmethod
    def parse(cls, version_string: str) -> 'SemanticVersion':
        """
        Parse version string into SemanticVersion object
        
        Args:
            version_string: Version string (e.g., "1.2.3", "2.0.0-alpha")
        
        Returns:
            SemanticVersion object
        
        Raises:
            ValueError: If version string is invalid
        """
        match = cls.SEMVER_PATTERN.match(version_string)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")
        
        groups = match.groupdict()
        
        return cls(
            major=int(groups['major']),
            minor=int(groups['minor']),
            patch=int(groups['patch']),
            prerelease=groups.get('prerelease'),
            build=groups.get('build')
        )
    
    def __str__(self) -> str:
        """String representation"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __repr__(self) -> str:
        """Debug representation"""
        return f"SemanticVersion('{str(self)}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison (ignores build metadata per semver spec)"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )
    
    def __lt__(self, other) -> bool:
        """Less than comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # If versions are equal, check prerelease
        # Per semver: version without prerelease > version with prerelease
        if self.prerelease is None and other.prerelease is None:
            return False
        
        if self.prerelease is None:
            return False  # self > other
        
        if other.prerelease is None:
            return True  # self < other
        
        # Both have prerelease, compare them
        return self._compare_prerelease(self.prerelease, other.prerelease) < 0
    
    @staticmethod
    def _compare_prerelease(pre1: str, pre2: str) -> int:
        """
        Compare prerelease versions
        
        Per semver spec:
        - Identifiers with only digits are compared numerically
        - Identifiers with letters are compared lexically
        - Numeric identifiers always have lower precedence than non-numeric
        
        Returns:
            -1 if pre1 < pre2, 0 if equal, 1 if pre1 > pre2
        """
        parts1 = pre1.split('.')
        parts2 = pre2.split('.')
        
        for i in range(max(len(parts1), len(parts2))):
            # If one list is shorter, it has lower precedence
            if i >= len(parts1):
                return -1
            if i >= len(parts2):
                return 1
            
            p1, p2 = parts1[i], parts2[i]
            
            # Check if parts are numeric
            is_num1 = p1.isdigit()
            is_num2 = p2.isdigit()
            
            if is_num1 and is_num2:
                # Both numeric, compare as integers
                if int(p1) < int(p2):
                    return -1
                elif int(p1) > int(p2):
                    return 1
            elif is_num1:
                # Numeric < non-numeric
                return -1
            elif is_num2:
                # Non-numeric > numeric
                return 1
            else:
                # Both non-numeric, compare lexically
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
        
        return 0
    
    def bump_major(self) -> 'SemanticVersion':
        """
        Bump major version (resets minor and patch to 0)
        
        Examples:
            1.2.3 -> 2.0.0
            2.0.0-alpha -> 3.0.0
        
        Returns:
            New SemanticVersion with bumped major version
        """
        return SemanticVersion(
            major=self.major + 1,
            minor=0,
            patch=0
        )
    
    def bump_minor(self) -> 'SemanticVersion':
        """
        Bump minor version (resets patch to 0)
        
        Examples:
            1.2.3 -> 1.3.0
            1.0.0-beta -> 1.1.0
        
        Returns:
            New SemanticVersion with bumped minor version
        """
        return SemanticVersion(
            major=self.major,
            minor=self.minor + 1,
            patch=0
        )
    
    def bump_patch(self) -> 'SemanticVersion':
        """
        Bump patch version
        
        Examples:
            1.2.3 -> 1.2.4
            1.0.0-rc.1 -> 1.0.1
        
        Returns:
            New SemanticVersion with bumped patch version
        """
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1
        )
    
    def is_stable(self) -> bool:
        """
        Check if version is stable (no prerelease)
        
        Returns:
            True if stable (no prerelease), False otherwise
        """
        return self.prerelease is None
    
    def is_prerelease(self) -> bool:
        """
        Check if version is prerelease
        
        Returns:
            True if prerelease, False otherwise
        """
        return self.prerelease is not None
    
    def is_compatible_with(self, other: 'SemanticVersion') -> bool:
        """
        Check if this version is compatible with another (same major version)
        
        Per semver: versions with same major version are compatible (if major > 0)
        For 0.x.x versions, minor version must match
        
        Args:
            other: Other version to check compatibility with
        
        Returns:
            True if compatible, False otherwise
        """
        if self.major == 0 and other.major == 0:
            # For 0.x.x, minor version must match
            return self.minor == other.minor
        
        # For 1.x.x and above, major version must match
        return self.major == other.major
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """
        Convert to tuple (major, minor, patch)
        
        Returns:
            Tuple of (major, minor, patch)
        """
        return (self.major, self.minor, self.patch)


class VersionConstraint:
    """
    Version constraint for dependency management
    
    Supports:
    - Exact: =1.2.3
    - Greater: >1.2.3, >=1.2.3
    - Less: <1.2.3, <=1.2.3
    - Caret: ^1.2.3 (compatible with 1.x.x)
    - Tilde: ~1.2.3 (compatible with 1.2.x)
    - Range: >=1.2.3 <2.0.0
    """
    
    def __init__(self, constraint_string: str):
        """
        Initialize version constraint
        
        Args:
            constraint_string: Constraint string (e.g., "^1.2.3", ">=1.0.0")
        """
        self.constraint_string = constraint_string.strip()
        self.constraints = self._parse_constraint(self.constraint_string)
    
    def _parse_constraint(self, constraint: str) -> List[Tuple[str, SemanticVersion]]:
        """
        Parse constraint string into list of (operator, version) tuples
        
        Args:
            constraint: Constraint string
        
        Returns:
            List of (operator, version) tuples
        """
        constraints = []
        
        # Handle caret (^)
        if constraint.startswith('^'):
            version = SemanticVersion.parse(constraint[1:])
            constraints.append(('^', version))
            return constraints
        
        # Handle tilde (~)
        if constraint.startswith('~'):
            version = SemanticVersion.parse(constraint[1:])
            constraints.append(('~', version))
            return constraints
        
        # Handle ranges (e.g., ">=1.0.0 <2.0.0")
        parts = constraint.split()
        
        for part in parts:
            # Match operator and version
            match = re.match(r'^([><=!]+)?(.+)$', part)
            if not match:
                raise ValueError(f"Invalid constraint: {part}")
            
            operator = match.group(1) or '='
            version_str = match.group(2)
            
            version = SemanticVersion.parse(version_str)
            constraints.append((operator, version))
        
        return constraints
    
    def satisfies(self, version: SemanticVersion) -> bool:
        """
        Check if version satisfies constraint
        
        Args:
            version: Version to check
        
        Returns:
            True if version satisfies constraint, False otherwise
        """
        for operator, constraint_version in self.constraints:
            if operator == '=':
                if version != constraint_version:
                    return False
            elif operator == '>':
                if not (version > constraint_version):
                    return False
            elif operator == '>=':
                if not (version >= constraint_version):
                    return False
            elif operator == '<':
                if not (version < constraint_version):
                    return False
            elif operator == '<=':
                if not (version <= constraint_version):
                    return False
            elif operator == '!=':
                if version == constraint_version:
                    return False
            elif operator == '^':
                # Caret: compatible with same major version
                if not version.is_compatible_with(constraint_version):
                    return False
                if version < constraint_version:
                    return False
            elif operator == '~':
                # Tilde: compatible with same major.minor version
                if version.major != constraint_version.major:
                    return False
                if version.minor != constraint_version.minor:
                    return False
                if version < constraint_version:
                    return False
        
        return True
    
    def __str__(self) -> str:
        """String representation"""
        return self.constraint_string
    
    def __repr__(self) -> str:
        """Debug representation"""
        return f"VersionConstraint('{self.constraint_string}')"


class VersionManager:
    """
    Version manager for dataset registry
    
    Handles version operations and recommendations
    """
    
    @staticmethod
    def suggest_next_version(
        current_version: SemanticVersion,
        change_type: str
    ) -> SemanticVersion:
        """
        Suggest next version based on change type
        
        Args:
            current_version: Current version
            change_type: Type of change (major, minor, patch, breaking, feature, bugfix)
        
        Returns:
            Suggested next version
        """
        change_type = change_type.lower()
        
        if change_type in ['major', 'breaking']:
            return current_version.bump_major()
        elif change_type in ['minor', 'feature']:
            return current_version.bump_minor()
        elif change_type in ['patch', 'bugfix', 'fix']:
            return current_version.bump_patch()
        else:
            raise ValueError(f"Unknown change type: {change_type}")
    
    @staticmethod
    def get_latest_version(versions: List[SemanticVersion]) -> Optional[SemanticVersion]:
        """
        Get latest version from list
        
        Args:
            versions: List of versions
        
        Returns:
            Latest version or None if list is empty
        """
        if not versions:
            return None
        
        # Filter out prerelease versions
        stable_versions = [v for v in versions if v.is_stable()]
        
        if stable_versions:
            return max(stable_versions)
        
        # If no stable versions, return latest prerelease
        return max(versions)
    
    @staticmethod
    def filter_by_constraint(
        versions: List[SemanticVersion],
        constraint: VersionConstraint
    ) -> List[SemanticVersion]:
        """
        Filter versions by constraint
        
        Args:
            versions: List of versions
            constraint: Version constraint
        
        Returns:
            Filtered list of versions
        """
        return [v for v in versions if constraint.satisfies(v)]
    
    @staticmethod
    def validate_version_string(version_string: str) -> bool:
        """
        Validate version string
        
        Args:
            version_string: Version string to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            SemanticVersion.parse(version_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """
        Compare two version strings
        
        Args:
            v1: First version string
            v2: Second version string
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        ver1 = SemanticVersion.parse(v1)
        ver2 = SemanticVersion.parse(v2)
        
        if ver1 < ver2:
            return -1
        elif ver1 > ver2:
            return 1
        else:
            return 0


# ============================================================================
# Helper Functions
# ============================================================================

def parse_version(version_string: str) -> SemanticVersion:
    """
    Parse version string
    
    Args:
        version_string: Version string (e.g., "1.2.3")
    
    Returns:
        SemanticVersion object
    """
    return SemanticVersion.parse(version_string)


def validate_version(version_string: str) -> bool:
    """
    Validate version string
    
    Args:
        version_string: Version string to validate
    
    Returns:
        True if valid, False otherwise
    """
    return VersionManager.validate_version_string(version_string)


def bump_version(version_string: str, bump_type: str) -> str:
    """
    Bump version
    
    Args:
        version_string: Current version string
        bump_type: Bump type (major, minor, patch)
    
    Returns:
        New version string
    """
    version = SemanticVersion.parse(version_string)
    
    if bump_type == 'major':
        new_version = version.bump_major()
    elif bump_type == 'minor':
        new_version = version.bump_minor()
    elif bump_type == 'patch':
        new_version = version.bump_patch()
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return str(new_version)


if __name__ == "__main__":
    # Example usage
    
    # Parse versions
    v1 = SemanticVersion.parse("1.2.3")
    v2 = SemanticVersion.parse("2.0.0")
    v3 = SemanticVersion.parse("1.3.0")
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v3: {v3}")
    
    # Compare versions
    print(f"\nv1 < v2: {v1 < v2}")
    print(f"v1 < v3: {v1 < v3}")
    print(f"v2 > v3: {v2 > v3}")
    
    # Bump versions
    print(f"\nv1.bump_major(): {v1.bump_major()}")
    print(f"v1.bump_minor(): {v1.bump_minor()}")
    print(f"v1.bump_patch(): {v1.bump_patch()}")
    
    # Check compatibility
    print(f"\nv1 compatible with v3: {v1.is_compatible_with(v3)}")
    print(f"v1 compatible with v2: {v1.is_compatible_with(v2)}")
    
    # Version constraints
    constraint = VersionConstraint("^1.2.0")
    print(f"\nConstraint: {constraint}")
    print(f"v1 (1.2.3) satisfies ^1.2.0: {constraint.satisfies(v1)}")
    print(f"v2 (2.0.0) satisfies ^1.2.0: {constraint.satisfies(v2)}")
    print(f"v3 (1.3.0) satisfies ^1.2.0: {constraint.satisfies(v3)}")
