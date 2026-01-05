#!/usr/bin/env python3
"""
Model Deployment & Testing - Task 5.5

Deploy and test fine-tuned models for OpenAI.

Features:
- Fine-tuned model deployment
- Model testing with sample inputs
- Performance comparison (base vs fine-tuned)
- Model validation
- Inference testing
- Response quality evaluation
- Cost estimation

Phase A1 Week 5-6: Task 5.5 COMPLETE
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from .openai_client import OpenAIClient
except ImportError:
    from training.finetuning.openai.openai_client import OpenAIClient


@dataclass
class InferenceRequest:
    """Inference request"""
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class InferenceResponse:
    """Inference response"""
    model: str
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    
    @property
    def cost_estimate(self) -> float:
        """
        Estimate cost in USD
        
        Rough estimates (as of 2024):
        - gpt-3.5-turbo: $0.0015/1K input, $0.002/1K output
        - gpt-4: $0.03/1K input, $0.06/1K output
        """
        # Use gpt-3.5-turbo pricing as default
        input_cost = (self.prompt_tokens / 1000) * 0.0015
        output_cost = (self.completion_tokens / 1000) * 0.002
        return input_cost + output_cost


@dataclass
class ComparisonResult:
    """Model comparison result"""
    base_model: str
    fine_tuned_model: str
    test_cases: int
    base_responses: List[InferenceResponse]
    fine_tuned_responses: List[InferenceResponse]
    
    @property
    def avg_latency_improvement(self) -> float:
        """Average latency improvement (negative = slower)"""
        base_avg = sum(r.latency_ms for r in self.base_responses) / len(self.base_responses)
        ft_avg = sum(r.latency_ms for r in self.fine_tuned_responses) / len(self.fine_tuned_responses)
        return ((base_avg - ft_avg) / base_avg) * 100
    
    @property
    def avg_token_difference(self) -> float:
        """Average token difference"""
        base_avg = sum(r.total_tokens for r in self.base_responses) / len(self.base_responses)
        ft_avg = sum(r.total_tokens for r in self.fine_tuned_responses) / len(self.fine_tuned_responses)
        return ft_avg - base_avg
    
    def __str__(self) -> str:
        return (
            f"Comparison Results ({self.test_cases} tests)\n"
            f"Base: {self.base_model}\n"
            f"Fine-tuned: {self.fine_tuned_model}\n"
            f"Latency improvement: {self.avg_latency_improvement:.1f}%\n"
            f"Token difference: {self.avg_token_difference:+.1f}"
        )


class ModelDeployer:
    """
    Deploy and manage fine-tuned models
    
    Handles:
    - Model deployment verification
    - Inference testing
    - Performance validation
    """
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize deployer
        
        Args:
            client: OpenAI client
        """
        self.client = client
        logger.info("ModelDeployer initialized")
    
    def verify_model(self, model_id: str) -> bool:
        """
        Verify model is deployed and accessible
        
        Args:
            model_id: Model ID
        
        Returns:
            True if model is accessible
        """
        try:
            model = self.client.get_model(model_id)
            logger.info(f"✓ Model verified: {model_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Model verification failed: {e}")
            return False
    
    def test_inference(
        self,
        model_id: str,
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        Test model inference
        
        Args:
            model_id: Model ID
            request: Inference request
        
        Returns:
            Inference response
        """
        try:
            start_time = time.time()
            
            # Call chat completion API
            response = self.client.client.chat.completions.create(
                model=model_id,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            choice = response.choices[0]
            usage = response.usage
            
            result = InferenceResponse(
                model=response.model,
                content=choice.message.content,
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=latency_ms
            )
            
            logger.info(
                f"Inference complete: {result.total_tokens} tokens, "
                f"{result.latency_ms:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def batch_test(
        self,
        model_id: str,
        test_cases: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        Run batch inference tests
        
        Args:
            model_id: Model ID
            test_cases: List of test requests
        
        Returns:
            List of responses
        """
        logger.info(f"Running {len(test_cases)} test cases")
        
        responses = []
        for idx, request in enumerate(test_cases, 1):
            try:
                response = self.test_inference(model_id, request)
                responses.append(response)
                logger.info(f"Test {idx}/{len(test_cases)} complete")
            except Exception as e:
                logger.error(f"Test {idx} failed: {e}")
        
        logger.info(f"Batch test complete: {len(responses)}/{len(test_cases)} succeeded")
        return responses
    
    def compare_models(
        self,
        base_model: str,
        fine_tuned_model: str,
        test_cases: List[InferenceRequest]
    ) -> ComparisonResult:
        """
        Compare base model vs fine-tuned model
        
        Args:
            base_model: Base model ID
            fine_tuned_model: Fine-tuned model ID
            test_cases: List of test requests
        
        Returns:
            Comparison result
        """
        logger.info(f"Comparing {base_model} vs {fine_tuned_model}")
        
        # Test base model
        logger.info("Testing base model...")
        base_responses = self.batch_test(base_model, test_cases)
        
        # Test fine-tuned model
        logger.info("Testing fine-tuned model...")
        ft_responses = self.batch_test(fine_tuned_model, test_cases)
        
        result = ComparisonResult(
            base_model=base_model,
            fine_tuned_model=fine_tuned_model,
            test_cases=len(test_cases),
            base_responses=base_responses,
            fine_tuned_responses=ft_responses
        )
        
        logger.info(f"Comparison complete:\n{result}")
        return result
    
    def validate_model(
        self,
        model_id: str,
        validation_cases: List[Tuple[InferenceRequest, str]]
    ) -> Dict[str, Any]:
        """
        Validate model with expected outputs
        
        Args:
            model_id: Model ID
            validation_cases: List of (request, expected_output) tuples
        
        Returns:
            Validation results dict
        """
        logger.info(f"Validating model: {model_id}")
        
        passed = 0
        failed = 0
        results = []
        
        for idx, (request, expected) in enumerate(validation_cases, 1):
            try:
                response = self.test_inference(model_id, request)
                
                # Simple validation: check if expected text is in response
                is_valid = expected.lower() in response.content.lower()
                
                if is_valid:
                    passed += 1
                    logger.info(f"✓ Validation {idx} passed")
                else:
                    failed += 1
                    logger.warning(f"✗ Validation {idx} failed")
                
                results.append({
                    'test_id': idx,
                    'passed': is_valid,
                    'expected': expected,
                    'actual': response.content
                })
                
            except Exception as e:
                failed += 1
                logger.error(f"✗ Validation {idx} error: {e}")
                results.append({
                    'test_id': idx,
                    'passed': False,
                    'error': str(e)
                })
        
        validation_result = {
            'model': model_id,
            'total': len(validation_cases),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(validation_cases) if validation_cases else 0,
            'results': results
        }
        
        logger.info(
            f"Validation complete: {passed}/{len(validation_cases)} passed "
            f"({validation_result['pass_rate']:.1%})"
        )
        
        return validation_result
    
    def estimate_costs(
        self,
        responses: List[InferenceResponse]
    ) -> Dict[str, float]:
        """
        Estimate costs for responses
        
        Args:
            responses: List of inference responses
        
        Returns:
            Cost breakdown dict
        """
        total_cost = sum(r.cost_estimate for r in responses)
        avg_cost = total_cost / len(responses) if responses else 0
        
        total_tokens = sum(r.total_tokens for r in responses)
        
        return {
            'total_cost_usd': total_cost,
            'avg_cost_per_request_usd': avg_cost,
            'total_tokens': total_tokens,
            'total_requests': len(responses)
        }


# Helper functions

def test_model(
    client: OpenAIClient,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7
) -> InferenceResponse:
    """
    Test model with simple interface
    
    Args:
        client: OpenAI client
        model_id: Model ID
        messages: Chat messages
        temperature: Temperature
    
    Returns:
        Inference response
    """
    deployer = ModelDeployer(client)
    request = InferenceRequest(
        messages=messages,
        temperature=temperature
    )
    return deployer.test_inference(model_id, request)


def compare_models(
    client: OpenAIClient,
    base_model: str,
    fine_tuned_model: str,
    test_messages: List[List[Dict[str, str]]]
) -> ComparisonResult:
    """
    Compare models with simple interface
    
    Args:
        client: OpenAI client
        base_model: Base model ID
        fine_tuned_model: Fine-tuned model ID
        test_messages: List of message lists
    
    Returns:
        Comparison result
    """
    deployer = ModelDeployer(client)
    test_cases = [
        InferenceRequest(messages=msgs)
        for msgs in test_messages
    ]
    return deployer.compare_models(base_model, fine_tuned_model, test_cases)


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("=== Model Deployment Test ===\n")
    
    # Test 1: Create inference request
    try:
        print("Test 1: Inference request")
        request = InferenceRequest(
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.7
        )
        print(f"✓ Request created: {len(request.messages)} messages\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        sys.exit(1)
    
    # Test 2: Create inference response
    try:
        print("Test 2: Inference response")
        response = InferenceResponse(
            model="gpt-3.5-turbo",
            content="2+2 equals 4.",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=250.5
        )
        print(f"✓ Response created: {response.total_tokens} tokens\n")
        print(f"  Cost estimate: ${response.cost_estimate:.6f}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Test 3: Comparison result
    try:
        print("Test 3: Comparison result")
        base_resp = InferenceResponse(
            model="gpt-3.5-turbo",
            content="Base response",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=300.0
        )
        ft_resp = InferenceResponse(
            model="ft:gpt-3.5-turbo:custom",
            content="Fine-tuned response",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=250.0
        )
        
        comparison = ComparisonResult(
            base_model="gpt-3.5-turbo",
            fine_tuned_model="ft:gpt-3.5-turbo:custom",
            test_cases=1,
            base_responses=[base_resp],
            fine_tuned_responses=[ft_resp]
        )
        
        print(f"{comparison}\n")
        print("✓ Comparison created\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    print("=== Tests Complete ===")
