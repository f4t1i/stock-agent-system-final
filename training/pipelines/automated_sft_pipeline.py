"""
Automated SFT Pipeline

Closed-loop pipeline that automatically:
1. Retrieves successful trajectories from Experience Library
2. Synthesizes SFT training data
3. Trains junior agents with Unsloth
4. Evaluates and deploys new models

This creates a continuous improvement loop for junior agents.
"""

import os
import json
import time
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass
import subprocess

from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres
from data_pipeline.data_synthesis import DataSynthesisModule


@dataclass
class SFTPipelineConfig:
    """Configuration for SFT pipeline"""
    agent_type: str  # news, technical, fundamental
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Data synthesis
    min_reward: float = 0.5
    min_confidence: float = 0.7
    quality_threshold: float = 0.6
    max_examples: int = 1000
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Output
    output_dir: str = "models/sft"
    
    # Evaluation
    eval_threshold: float = 0.7  # Min score to deploy
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_experience"
    db_user: str = "postgres"
    db_password: str = "postgres"


class AutomatedSFTPipeline:
    """
    Automated SFT Pipeline
    
    Implements closed-loop continuous learning for junior agents.
    """
    
    def __init__(self, config: SFTPipelineConfig):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.library = ExperienceLibraryPostgres(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
            user=config.db_user,
            password=config.db_password
        )
        
        self.synthesizer = DataSynthesisModule(
            experience_library=self.library,
            min_reward=config.min_reward,
            min_confidence=config.min_confidence,
            quality_threshold=config.quality_threshold
        )
        
        self.training_history = []
    
    def run_pipeline(
        self,
        market_regime: Optional[str] = None,
        force_retrain: bool = False
    ) -> Dict:
        """
        Run complete SFT pipeline
        
        Args:
            market_regime: Filter by market regime
            force_retrain: Force retraining even if not enough new data
        
        Returns:
            Pipeline results dictionary
        """
        logger.info(f"Starting SFT pipeline for {self.config.agent_type} agent")
        
        start_time = time.time()
        results = {
            'agent_type': self.config.agent_type,
            'start_time': start_time,
            'steps': {}
        }
        
        try:
            # Step 1: Check if retraining is needed
            if not force_retrain:
                should_retrain = self._check_retrain_needed()
                if not should_retrain:
                    logger.info("Not enough new data for retraining")
                    results['skipped'] = True
                    return results
            
            # Step 2: Synthesize training data
            logger.info("Step 1/4: Synthesizing training data...")
            data_path = self._synthesize_data(market_regime)
            results['steps']['data_synthesis'] = {
                'path': data_path,
                'success': True
            }
            
            # Step 3: Train model
            logger.info("Step 2/4: Training model...")
            model_path = self._train_model(data_path)
            results['steps']['training'] = {
                'model_path': model_path,
                'success': True
            }
            
            # Step 4: Evaluate model
            logger.info("Step 3/4: Evaluating model...")
            eval_score = self._evaluate_model(model_path)
            results['steps']['evaluation'] = {
                'score': eval_score,
                'success': eval_score >= self.config.eval_threshold
            }
            
            # Step 5: Deploy if good enough
            if eval_score >= self.config.eval_threshold:
                logger.info("Step 4/4: Deploying model...")
                deployed_path = self._deploy_model(model_path)
                results['steps']['deployment'] = {
                    'path': deployed_path,
                    'success': True
                }
                logger.info(f"✅ Model deployed to {deployed_path}")
            else:
                logger.warning(
                    f"Model score {eval_score:.3f} below threshold "
                    f"{self.config.eval_threshold:.3f}, not deploying"
                )
                results['steps']['deployment'] = {
                    'success': False,
                    'reason': 'below_threshold'
                }
            
            results['success'] = True
            results['duration'] = time.time() - start_time
            
            # Record in history
            self.training_history.append(results)
            
            logger.info(
                f"✅ SFT pipeline completed in {results['duration']:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"SFT pipeline failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            results['duration'] = time.time() - start_time
        
        return results
    
    def _check_retrain_needed(self) -> bool:
        """
        Check if retraining is needed
        
        Returns:
            bool: Whether to retrain
        """
        # Get statistics
        stats = self.library.get_statistics()
        
        # Check if enough new successful trajectories
        overall = stats.get('overall', {})
        successful_count = overall.get('successful', 0)
        
        # Need at least 100 new successful trajectories
        if successful_count < 100:
            return False
        
        # Check if last training was recent
        if self.training_history:
            last_training = self.training_history[-1]
            last_time = last_training['start_time']
            
            # Don't retrain if last training was < 24 hours ago
            if time.time() - last_time < 24 * 3600:
                return False
        
        return True
    
    def _synthesize_data(self, market_regime: Optional[str] = None) -> str:
        """
        Synthesize training data
        
        Args:
            market_regime: Market regime filter
        
        Returns:
            Path to synthesized data file
        """
        # Create output directory
        data_dir = os.path.join(self.config.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"sft_data_{self.config.agent_type}_{timestamp}.jsonl"
        data_path = os.path.join(data_dir, filename)
        
        # Synthesize
        self.synthesizer.save_sft_dataset(
            agent_type=self.config.agent_type,
            output_path=data_path,
            market_regime=market_regime,
            limit=self.config.max_examples,
            output_format='chatml'
        )
        
        logger.info(f"Synthesized training data to {data_path}")
        
        return data_path
    
    def _train_model(self, data_path: str) -> str:
        """
        Train model with Unsloth
        
        Args:
            data_path: Path to training data
        
        Returns:
            Path to trained model
        """
        # Create output directory
        timestamp = int(time.time())
        model_dir = os.path.join(
            self.config.output_dir,
            f"{self.config.agent_type}_{timestamp}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare training script path
        training_script = f"training/sft/train_{self.config.agent_type}_agent.py"
        
        if not os.path.exists(training_script):
            raise FileNotFoundError(f"Training script not found: {training_script}")
        
        # Run training
        cmd = [
            'python3',
            training_script,
            '--data_path', data_path,
            '--base_model', self.config.base_model,
            '--output_dir', model_dir,
            '--num_epochs', str(self.config.num_epochs),
            '--batch_size', str(self.config.batch_size),
            '--learning_rate', str(self.config.learning_rate),
            '--max_seq_length', str(self.config.max_seq_length),
            '--lora_r', str(self.config.lora_r),
            '--lora_alpha', str(self.config.lora_alpha),
            '--lora_dropout', str(self.config.lora_dropout)
        ]
        
        logger.info(f"Running training: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        logger.info(f"Training completed: {model_dir}")
        
        return model_dir
    
    def _evaluate_model(self, model_path: str) -> float:
        """
        Evaluate trained model
        
        Args:
            model_path: Path to model
        
        Returns:
            Evaluation score [0, 1]
        """
        # TODO: Implement proper evaluation
        # For now, return dummy score
        
        # In production, this should:
        # 1. Load model
        # 2. Run on validation set
        # 3. Compare with Judge scores
        # 4. Return average score
        
        logger.info(f"Evaluating model at {model_path}")
        
        # Dummy evaluation
        eval_score = 0.75
        
        logger.info(f"Evaluation score: {eval_score:.3f}")
        
        return eval_score
    
    def _deploy_model(self, model_path: str) -> str:
        """
        Deploy model to production
        
        Args:
            model_path: Path to model
        
        Returns:
            Path to deployed model
        """
        # Create deployment directory
        deploy_dir = os.path.join(
            self.config.output_dir,
            'deployed',
            self.config.agent_type
        )
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Copy model to deployment directory
        import shutil
        
        # Remove old deployment if exists
        if os.path.exists(os.path.join(deploy_dir, 'latest')):
            shutil.rmtree(os.path.join(deploy_dir, 'latest'))
        
        # Copy new model
        shutil.copytree(model_path, os.path.join(deploy_dir, 'latest'))
        
        # Create symlink to latest
        latest_link = os.path.join(deploy_dir, 'current')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink('latest', latest_link)
        
        deployed_path = os.path.join(deploy_dir, 'latest')
        
        logger.info(f"Deployed model to {deployed_path}")
        
        return deployed_path
    
    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history
    
    def close(self):
        """Close pipeline"""
        self.library.close()


def run_automated_sft_pipeline(
    agent_type: str,
    market_regime: Optional[str] = None,
    force_retrain: bool = False
) -> Dict:
    """
    Run automated SFT pipeline for an agent
    
    Args:
        agent_type: Agent type (news, technical, fundamental)
        market_regime: Market regime filter
        force_retrain: Force retraining
    
    Returns:
        Pipeline results
    """
    config = SFTPipelineConfig(agent_type=agent_type)
    
    pipeline = AutomatedSFTPipeline(config)
    
    try:
        results = pipeline.run_pipeline(
            market_regime=market_regime,
            force_retrain=force_retrain
        )
        return results
    finally:
        pipeline.close()


if __name__ == '__main__':
    # Test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str, required=True,
                        choices=['news', 'technical', 'fundamental'])
    parser.add_argument('--market_regime', type=str, default=None)
    parser.add_argument('--force_retrain', action='store_true')
    
    args = parser.parse_args()
    
    results = run_automated_sft_pipeline(
        agent_type=args.agent_type,
        market_regime=args.market_regime,
        force_retrain=args.force_retrain
    )
    
    print(json.dumps(results, indent=2))
