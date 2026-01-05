#!/usr/bin/env python3
"""
Dataset Upload & Validation - Task 5.2

Upload and validate datasets for OpenAI fine-tuning.

Features:
- Dataset format validation (JSONL)
- ChatML format verification
- File upload to OpenAI
- Upload progress tracking
- File management (list, retrieve, delete)
- Validation errors handling

Phase A1 Week 5-6: Task 5.2 COMPLETE
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from .openai_client import OpenAIClient
except ImportError:
    from training.finetuning.openai.openai_client import OpenAIClient


@dataclass
class ValidationResult:
    """Dataset validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    example_count: int
    total_tokens: Optional[int] = None
    
    def __str__(self) -> str:
        status = "✓ Valid" if self.is_valid else "✗ Invalid"
        return (
            f"{status}\n"
            f"Examples: {self.example_count}\n"
            f"Errors: {len(self.errors)}\n"
            f"Warnings: {len(self.warnings)}"
        )


@dataclass
class UploadResult:
    """File upload result"""
    file_id: str
    filename: str
    bytes: int
    purpose: str
    status: str
    created_at: int
    
    @classmethod
    def from_openai_file(cls, file_obj) -> 'UploadResult':
        """Create from OpenAI file object"""
        return cls(
            file_id=file_obj.id,
            filename=file_obj.filename,
            bytes=file_obj.bytes,
            purpose=file_obj.purpose,
            status=file_obj.status,
            created_at=file_obj.created_at
        )


class DatasetValidator:
    """
    Validate datasets for OpenAI fine-tuning
    
    Validates:
    - JSONL format
    - ChatML message structure
    - Required fields
    - Token limits
    """
    
    def __init__(self, max_tokens_per_example: int = 4096):
        """
        Initialize validator
        
        Args:
            max_tokens_per_example: Maximum tokens per example
        """
        self.max_tokens_per_example = max_tokens_per_example
        logger.info(f"DatasetValidator initialized (max_tokens={max_tokens_per_example})")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate dataset file
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        example_count = 0
        total_tokens = 0
        
        # Check file exists
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                example_count=0
            )
        
        # Check file extension
        if not file_path.endswith('.jsonl'):
            warnings.append("File should have .jsonl extension")
        
        # Validate each line
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        example = json.loads(line)
                        example_count += 1
                        
                        # Validate example structure
                        result = self._validate_example(example, line_num)
                        errors.extend(result['errors'])
                        warnings.extend(result['warnings'])
                        total_tokens += result.get('tokens', 0)
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
        
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
        
        # Check minimum examples
        if example_count < 10:
            warnings.append(f"Only {example_count} examples. Recommended: 50-100+")
        
        is_valid = len(errors) == 0
        
        logger.info(
            f"Validation {'passed' if is_valid else 'failed'}: "
            f"{example_count} examples, {len(errors)} errors, {len(warnings)} warnings"
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            example_count=example_count,
            total_tokens=total_tokens if total_tokens > 0 else None
        )
    
    def _validate_example(
        self,
        example: Dict[str, Any],
        line_num: int
    ) -> Dict[str, Any]:
        """
        Validate single example
        
        Args:
            example: Example dict
            line_num: Line number
        
        Returns:
            Dict with errors, warnings, tokens
        """
        errors = []
        warnings = []
        tokens = 0
        
        # Check for 'messages' field
        if 'messages' not in example:
            errors.append(f"Line {line_num}: Missing 'messages' field")
            return {'errors': errors, 'warnings': warnings, 'tokens': tokens}
        
        messages = example['messages']
        
        # Check messages is a list
        if not isinstance(messages, list):
            errors.append(f"Line {line_num}: 'messages' must be a list")
            return {'errors': errors, 'warnings': warnings, 'tokens': tokens}
        
        # Check minimum messages
        if len(messages) < 2:
            errors.append(f"Line {line_num}: Need at least 2 messages")
            return {'errors': errors, 'warnings': warnings, 'tokens': tokens}
        
        # Validate each message
        for msg_idx, msg in enumerate(messages):
            # Check message structure
            if not isinstance(msg, dict):
                errors.append(f"Line {line_num}, message {msg_idx}: Must be a dict")
                continue
            
            # Check required fields
            if 'role' not in msg:
                errors.append(f"Line {line_num}, message {msg_idx}: Missing 'role'")
            
            if 'content' not in msg:
                errors.append(f"Line {line_num}, message {msg_idx}: Missing 'content'")
            
            # Check role value
            role = msg.get('role')
            if role not in ['system', 'user', 'assistant']:
                errors.append(
                    f"Line {line_num}, message {msg_idx}: "
                    f"Invalid role '{role}'. Must be system/user/assistant"
                )
            
            # Check content
            content = msg.get('content', '')
            if not content or not isinstance(content, str):
                warnings.append(
                    f"Line {line_num}, message {msg_idx}: Empty or invalid content"
                )
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            tokens += len(content) // 4
        
        # Check token limit
        if tokens > self.max_tokens_per_example:
            warnings.append(
                f"Line {line_num}: ~{tokens} tokens exceeds limit "
                f"({self.max_tokens_per_example})"
            )
        
        # Check message sequence
        if messages[0].get('role') not in ['system', 'user']:
            warnings.append(
                f"Line {line_num}: First message should be system or user"
            )
        
        if messages[-1].get('role') != 'assistant':
            warnings.append(
                f"Line {line_num}: Last message should be assistant"
            )
        
        return {'errors': errors, 'warnings': warnings, 'tokens': tokens}
    
    def validate_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate list of examples
        
        Args:
            examples: List of example dicts
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        total_tokens = 0
        
        for idx, example in enumerate(examples, 1):
            result = self._validate_example(example, idx)
            errors.extend(result['errors'])
            warnings.extend(result['warnings'])
            total_tokens += result.get('tokens', 0)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            example_count=len(examples),
            total_tokens=total_tokens
        )


class DatasetUploader:
    """
    Upload datasets to OpenAI for fine-tuning
    
    Handles:
    - File upload
    - Progress tracking
    - File management
    """
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize uploader
        
        Args:
            client: OpenAI client
        """
        self.client = client
        self.validator = DatasetValidator()
        logger.info("DatasetUploader initialized")
    
    def upload_file(
        self,
        file_path: str,
        purpose: str = "fine-tune",
        validate: bool = True
    ) -> UploadResult:
        """
        Upload dataset file to OpenAI
        
        Args:
            file_path: Path to JSONL file
            purpose: File purpose (default: "fine-tune")
            validate: Validate before upload
        
        Returns:
            Upload result
        
        Raises:
            ValueError: If validation fails
        """
        # Validate file
        if validate:
            logger.info(f"Validating file: {file_path}")
            validation = self.validator.validate_file(file_path)
            
            if not validation.is_valid:
                error_msg = "\n".join(validation.errors)
                raise ValueError(f"Validation failed:\n{error_msg}")
            
            if validation.warnings:
                logger.warning(f"Validation warnings:\n" + "\n".join(validation.warnings))
            
            logger.info(f"✓ Validation passed: {validation.example_count} examples")
        
        # Upload file
        try:
            logger.info(f"Uploading file: {file_path}")
            
            with open(file_path, 'rb') as f:
                file_obj = self.client.client.files.create(
                    file=f,
                    purpose=purpose
                )
            
            result = UploadResult.from_openai_file(file_obj)
            
            logger.info(
                f"✓ Upload complete: {result.file_id} "
                f"({result.bytes} bytes, {result.status})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def list_files(self, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List uploaded files
        
        Args:
            purpose: Filter by purpose (e.g., "fine-tune")
        
        Returns:
            List of file objects
        """
        try:
            files = self.client.client.files.list()
            
            files_list = list(files.data)
            
            # Filter by purpose
            if purpose:
                files_list = [f for f in files_list if f.purpose == purpose]
            
            logger.info(f"Listed {len(files_list)} files")
            return files_list
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
    
    def get_file(self, file_id: str) -> Dict[str, Any]:
        """
        Get file details
        
        Args:
            file_id: File ID
        
        Returns:
            File object
        """
        try:
            file_obj = self.client.client.files.retrieve(file_id)
            logger.info(f"Retrieved file: {file_id}")
            return file_obj
            
        except Exception as e:
            logger.error(f"Failed to get file {file_id}: {e}")
            raise
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete uploaded file
        
        Args:
            file_id: File ID
        
        Returns:
            True if deletion successful
        """
        try:
            result = self.client.client.files.delete(file_id)
            logger.info(f"Deleted file: {file_id}")
            return result.deleted
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise
    
    def download_file(self, file_id: str, output_path: str) -> None:
        """
        Download file content
        
        Args:
            file_id: File ID
            output_path: Output file path
        """
        try:
            content = self.client.client.files.content(file_id)
            
            with open(output_path, 'wb') as f:
                f.write(content.read())
            
            logger.info(f"Downloaded file {file_id} to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            raise


# Helper functions

def validate_dataset(file_path: str) -> ValidationResult:
    """
    Validate dataset file
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        Validation result
    """
    validator = DatasetValidator()
    return validator.validate_file(file_path)


def upload_dataset(
    client: OpenAIClient,
    file_path: str,
    validate: bool = True
) -> UploadResult:
    """
    Upload dataset file
    
    Args:
        client: OpenAI client
        file_path: Path to JSONL file
        validate: Validate before upload
    
    Returns:
        Upload result
    """
    uploader = DatasetUploader(client)
    return uploader.upload_file(file_path, validate=validate)


if __name__ == "__main__":
    # Example usage
    import sys
    import tempfile
    
    print("=== Dataset Upload Test ===\n")
    
    # Create test dataset
    test_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.jsonl',
        delete=False
    )
    
    # Write test examples
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        }
    ]
    
    for example in examples:
        test_file.write(json.dumps(example) + '\n')
    
    test_file.close()
    
    print(f"Created test file: {test_file.name}\n")
    
    # Test 1: Validate dataset
    try:
        print("Test 1: Validate dataset")
        result = validate_dataset(test_file.name)
        print(f"{result}\n")
        
        if not result.is_valid:
            print("✗ Validation failed\n")
            sys.exit(1)
        
        print("✓ Validation passed\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        sys.exit(1)
    finally:
        os.unlink(test_file.name)
    
    print("=== Tests Complete ===")
