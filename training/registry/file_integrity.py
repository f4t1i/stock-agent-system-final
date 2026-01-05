#!/usr/bin/env python3
"""
File Integrity Module - Task 2.3

SHA256-based file integrity checking for dataset registry.

Features:
- SHA256 hash calculation for files
- Hash verification and validation
- Integrity checking for datasets
- Corruption detection
- Batch hash calculation
- Progress tracking for large files

Use Cases:
- Verify dataset file integrity after download
- Detect file corruption
- Ensure dataset consistency across systems
- Track file changes

Phase A1 Week 3-4: Task 2.3 COMPLETE
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class FileHash:
    """
    File hash information
    """
    file_path: str
    sha256: str
    size_bytes: int
    algorithm: str = "sha256"
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.sha256} ({self.size_bytes} bytes)"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'file_path': self.file_path,
            'sha256': self.sha256,
            'size_bytes': self.size_bytes,
            'algorithm': self.algorithm
        }


@dataclass
class IntegrityCheckResult:
    """
    Result of integrity check
    """
    file_path: str
    is_valid: bool
    expected_hash: str
    actual_hash: str
    size_bytes: int
    error: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation"""
        if self.is_valid:
            return f"✓ {self.file_path}: Valid"
        else:
            return f"✗ {self.file_path}: Invalid ({self.error or 'Hash mismatch'})"


class FileIntegrityChecker:
    """
    File integrity checker using SHA256
    
    Provides methods for calculating and verifying file hashes
    """
    
    # Buffer size for reading files (8MB)
    BUFFER_SIZE = 8 * 1024 * 1024
    
    @staticmethod
    def calculate_hash(
        file_path: str,
        algorithm: str = "sha256",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> FileHash:
        """
        Calculate hash for a file
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, sha1, md5)
            progress_callback: Optional callback(bytes_read, total_bytes)
        
        Returns:
            FileHash object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If algorithm is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create hash object
        if algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm == "md5":
            hash_obj = hashlib.md5()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Read file in chunks and update hash
        bytes_read = 0
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(FileIntegrityChecker.BUFFER_SIZE)
                if not chunk:
                    break
                
                hash_obj.update(chunk)
                bytes_read += len(chunk)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(bytes_read, file_size)
        
        # Get hex digest
        hash_value = hash_obj.hexdigest()
        
        logger.debug(f"Calculated {algorithm} hash for {file_path}: {hash_value}")
        
        return FileHash(
            file_path=file_path,
            sha256=hash_value,
            size_bytes=file_size,
            algorithm=algorithm
        )
    
    @staticmethod
    def calculate_sha256(file_path: str) -> str:
        """
        Calculate SHA256 hash for a file (convenience method)
        
        Args:
            file_path: Path to file
        
        Returns:
            SHA256 hash as hex string
        """
        file_hash = FileIntegrityChecker.calculate_hash(file_path, algorithm="sha256")
        return file_hash.sha256
    
    @staticmethod
    def verify_hash(
        file_path: str,
        expected_hash: str,
        algorithm: str = "sha256"
    ) -> IntegrityCheckResult:
        """
        Verify file hash against expected value
        
        Args:
            file_path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm
        
        Returns:
            IntegrityCheckResult object
        """
        try:
            # Calculate actual hash
            file_hash = FileIntegrityChecker.calculate_hash(file_path, algorithm)
            actual_hash = file_hash.sha256
            
            # Compare hashes (case-insensitive)
            is_valid = actual_hash.lower() == expected_hash.lower()
            
            result = IntegrityCheckResult(
                file_path=file_path,
                is_valid=is_valid,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                size_bytes=file_hash.size_bytes,
                error=None if is_valid else "Hash mismatch"
            )
            
            if is_valid:
                logger.info(f"✓ File integrity verified: {file_path}")
            else:
                logger.warning(
                    f"✗ File integrity check failed: {file_path}\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual_hash}"
                )
            
            return result
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            return IntegrityCheckResult(
                file_path=file_path,
                is_valid=False,
                expected_hash=expected_hash,
                actual_hash="",
                size_bytes=0,
                error=f"File not found: {e}"
            )
        
        except Exception as e:
            logger.error(f"Error verifying hash for {file_path}: {e}")
            return IntegrityCheckResult(
                file_path=file_path,
                is_valid=False,
                expected_hash=expected_hash,
                actual_hash="",
                size_bytes=0,
                error=str(e)
            )
    
    @staticmethod
    def batch_calculate_hashes(
        file_paths: List[str],
        algorithm: str = "sha256"
    ) -> List[FileHash]:
        """
        Calculate hashes for multiple files
        
        Args:
            file_paths: List of file paths
            algorithm: Hash algorithm
        
        Returns:
            List of FileHash objects
        """
        hashes = []
        
        for file_path in file_paths:
            try:
                file_hash = FileIntegrityChecker.calculate_hash(file_path, algorithm)
                hashes.append(file_hash)
            except Exception as e:
                logger.error(f"Error calculating hash for {file_path}: {e}")
        
        return hashes
    
    @staticmethod
    def batch_verify_hashes(
        file_hashes: Dict[str, str],
        algorithm: str = "sha256"
    ) -> List[IntegrityCheckResult]:
        """
        Verify hashes for multiple files
        
        Args:
            file_hashes: Dict mapping file paths to expected hashes
            algorithm: Hash algorithm
        
        Returns:
            List of IntegrityCheckResult objects
        """
        results = []
        
        for file_path, expected_hash in file_hashes.items():
            result = FileIntegrityChecker.verify_hash(
                file_path,
                expected_hash,
                algorithm
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def compare_files(file1: str, file2: str) -> bool:
        """
        Compare two files by hash
        
        Args:
            file1: Path to first file
            file2: Path to second file
        
        Returns:
            True if files are identical, False otherwise
        """
        hash1 = FileIntegrityChecker.calculate_sha256(file1)
        hash2 = FileIntegrityChecker.calculate_sha256(file2)
        
        are_equal = hash1 == hash2
        
        if are_equal:
            logger.info(f"Files are identical: {file1} == {file2}")
        else:
            logger.info(f"Files are different: {file1} != {file2}")
        
        return are_equal
    
    @staticmethod
    def detect_duplicates(file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Detect duplicate files by hash
        
        Args:
            file_paths: List of file paths to check
        
        Returns:
            Dict mapping hashes to lists of file paths with that hash
        """
        hash_to_files = {}
        
        for file_path in file_paths:
            try:
                file_hash = FileIntegrityChecker.calculate_sha256(file_path)
                
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                
                hash_to_files[file_hash].append(file_path)
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Filter to only duplicates (hash appears more than once)
        duplicates = {
            hash_val: files
            for hash_val, files in hash_to_files.items()
            if len(files) > 1
        }
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} sets of duplicate files")
        else:
            logger.info("No duplicate files found")
        
        return duplicates


class DatasetIntegrityManager:
    """
    Integrity manager for dataset registry
    
    Integrates with dataset storage to track and verify file integrity
    """
    
    def __init__(self):
        """Initialize integrity manager"""
        self.checker = FileIntegrityChecker()
        logger.info("DatasetIntegrityManager initialized")
    
    def calculate_dataset_hash(
        self,
        dataset_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> FileHash:
        """
        Calculate hash for dataset file
        
        Args:
            dataset_path: Path to dataset file
            progress_callback: Optional progress callback
        
        Returns:
            FileHash object
        """
        logger.info(f"Calculating hash for dataset: {dataset_path}")
        
        file_hash = self.checker.calculate_hash(
            dataset_path,
            algorithm="sha256",
            progress_callback=progress_callback
        )
        
        logger.info(
            f"Dataset hash calculated: {file_hash.sha256} "
            f"({file_hash.size_bytes} bytes)"
        )
        
        return file_hash
    
    def verify_dataset_integrity(
        self,
        dataset_path: str,
        expected_hash: str
    ) -> IntegrityCheckResult:
        """
        Verify dataset file integrity
        
        Args:
            dataset_path: Path to dataset file
            expected_hash: Expected SHA256 hash
        
        Returns:
            IntegrityCheckResult object
        """
        logger.info(f"Verifying integrity for dataset: {dataset_path}")
        
        result = self.checker.verify_hash(
            dataset_path,
            expected_hash,
            algorithm="sha256"
        )
        
        if result.is_valid:
            logger.info(f"✓ Dataset integrity verified: {dataset_path}")
        else:
            logger.error(f"✗ Dataset integrity check failed: {dataset_path}")
        
        return result
    
    def verify_multiple_datasets(
        self,
        datasets: Dict[str, str]
    ) -> Tuple[List[IntegrityCheckResult], int, int]:
        """
        Verify integrity for multiple datasets
        
        Args:
            datasets: Dict mapping dataset paths to expected hashes
        
        Returns:
            Tuple of (results, valid_count, invalid_count)
        """
        logger.info(f"Verifying integrity for {len(datasets)} datasets")
        
        results = self.checker.batch_verify_hashes(datasets, algorithm="sha256")
        
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        
        logger.info(
            f"Integrity check complete: "
            f"{valid_count} valid, {invalid_count} invalid"
        )
        
        return results, valid_count, invalid_count
    
    def create_integrity_manifest(
        self,
        file_paths: List[str],
        output_path: str
    ):
        """
        Create integrity manifest file
        
        Args:
            file_paths: List of file paths to include
            output_path: Path to output manifest file
        """
        import json
        
        logger.info(f"Creating integrity manifest for {len(file_paths)} files")
        
        hashes = self.checker.batch_calculate_hashes(file_paths)
        
        manifest = {
            'version': '1.0',
            'algorithm': 'sha256',
            'files': [h.to_dict() for h in hashes]
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Integrity manifest saved to {output_path}")
    
    def verify_from_manifest(
        self,
        manifest_path: str
    ) -> Tuple[List[IntegrityCheckResult], int, int]:
        """
        Verify files from integrity manifest
        
        Args:
            manifest_path: Path to manifest file
        
        Returns:
            Tuple of (results, valid_count, invalid_count)
        """
        import json
        
        logger.info(f"Verifying files from manifest: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        file_hashes = {
            file_info['file_path']: file_info['sha256']
            for file_info in manifest['files']
        }
        
        return self.verify_multiple_datasets(file_hashes)


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash for a file (convenience function)
    
    Args:
        file_path: Path to file
    
    Returns:
        SHA256 hash as hex string
    """
    return FileIntegrityChecker.calculate_sha256(file_path)


def verify_file_integrity(file_path: str, expected_hash: str) -> bool:
    """
    Verify file integrity (convenience function)
    
    Args:
        file_path: Path to file
        expected_hash: Expected SHA256 hash
    
    Returns:
        True if valid, False otherwise
    """
    result = FileIntegrityChecker.verify_hash(file_path, expected_hash)
    return result.is_valid


def compare_files(file1: str, file2: str) -> bool:
    """
    Compare two files by hash (convenience function)
    
    Args:
        file1: Path to first file
        file2: Path to second file
    
    Returns:
        True if files are identical, False otherwise
    """
    return FileIntegrityChecker.compare_files(file1, file2)


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create test files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, World!")
        test_file1 = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, World!")
        test_file2 = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Different content")
        test_file3 = f.name
    
    print("=== File Integrity Checker ===\n")
    
    # Calculate hash
    print("1. Calculate hash:")
    file_hash = FileIntegrityChecker.calculate_hash(test_file1)
    print(f"   File: {test_file1}")
    print(f"   Hash: {file_hash.sha256}")
    print(f"   Size: {file_hash.size_bytes} bytes\n")
    
    # Verify hash
    print("2. Verify hash:")
    result = FileIntegrityChecker.verify_hash(test_file1, file_hash.sha256)
    print(f"   {result}\n")
    
    # Compare files
    print("3. Compare files:")
    are_equal = FileIntegrityChecker.compare_files(test_file1, test_file2)
    print(f"   {test_file1} == {test_file2}: {are_equal}")
    
    are_equal = FileIntegrityChecker.compare_files(test_file1, test_file3)
    print(f"   {test_file1} == {test_file3}: {are_equal}\n")
    
    # Detect duplicates
    print("4. Detect duplicates:")
    duplicates = FileIntegrityChecker.detect_duplicates([test_file1, test_file2, test_file3])
    if duplicates:
        for hash_val, files in duplicates.items():
            print(f"   Hash {hash_val[:16]}... found in:")
            for file_path in files:
                print(f"     - {file_path}")
    else:
        print("   No duplicates found\n")
    
    # Clean up
    os.unlink(test_file1)
    os.unlink(test_file2)
    os.unlink(test_file3)
    
    print("\n✅ All examples completed!")
