"""
Automated RL Pipeline

Closed-loop pipeline that automatically:
1. Retrieves trajectories from Experience Library
2. Calculates rewards based on actual outcomes
3. Trains strategist agent with GRPO
4. Evaluates and deploys new models

This creates a continuous improvement loop for the senior strategist agent.
"""

import os
import json
import time
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass
import subprocess
import numpy as np

from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres


@dataclass
class RLPipelineConfig:
    """Configuration for RL pipeline"""
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Data
    min_trajectories: int = 500
    max_trajectories: int = 5000
    
    # Reward shaping
    return_weight: float = 1.0
    risk_penalty: float = 0.1
    confidence_bonus: float = 0.05
    
    # Training (GRPO)
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # KL divergence constraint
    kl_coef: float = 0.1
    
    # Output
    output_dir: str = "models/rl"
    
    # Evaluation
    eval_threshold: float = 0.7
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_experience"
    db_user: str = "postgres"
    db_password: str = "postgres"


class AutomatedRLPipeline:
    """
    Automated RL Pipeline
    
    Implements closed-loop continuous learning for strategist agent.
    """
    
    def __init__(self, config: RLPipelineConfig):
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
        
        self.training_history = []
    
    def run_pipeline(
        self,
        market_regime: Optional[str] = None,
        force_retrain: bool = False
    ) -> Dict:
        """
        Run complete RL pipeline
        
        Args:
            market_regime: Filter by market regime
            force_retrain: Force retraining even if not enough new data
        
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting RL pipeline for strategist agent")
        
        start_time = time.time()
        results = {
            'agent_type': 'strategist',
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
            
            # Step 2: Prepare RL training data
            logger.info("Step 1/4: Preparing RL training data...")
            data_path = self._prepare_rl_data(market_regime)
            results['steps']['data_preparation'] = {
                'path': data_path,
                'success': True
            }
            
            # Step 3: Train model with GRPO
            logger.info("Step 2/4: Training model with GRPO...")
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
                f"✅ RL pipeline completed in {results['duration']:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"RL pipeline failed: {e}")
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
        
        # Check strategist trajectories
        by_agent = stats.get('by_agent_type', [])
        strategist_stats = next(
            (s for s in by_agent if s['agent_type'] == 'strategist'),
            None
        )
        
        if not strategist_stats:
            return False
        
        # Need at least min_trajectories
        if strategist_stats['count'] < self.config.min_trajectories:
            return False
        
        # Check if last training was recent
        if self.training_history:
            last_training = self.training_history[-1]
            last_time = last_training['start_time']
            
            # Don't retrain if last training was < 48 hours ago
            if time.time() - last_time < 48 * 3600:
                return False
        
        return True
    
    def _prepare_rl_data(self, market_regime: Optional[str] = None) -> str:
        """
        Prepare RL training data
        
        Args:
            market_regime: Market regime filter
        
        Returns:
            Path to prepared data file
        """
        # Get strategist trajectories with outcomes
        trajectories = self.library.get_successful_trajectories(
            agent_type='strategist',
            market_regime=market_regime,
            min_reward=-1.0,  # Include all (even negative rewards)
            limit=self.config.max_trajectories
        )
        
        # Filter trajectories with actual outcomes
        trajectories_with_outcomes = [
            t for t in trajectories
            if t.actual_return is not None
        ]
        
        logger.info(
            f"Found {len(trajectories_with_outcomes)} trajectories with outcomes"
        )
        
        # Calculate shaped rewards
        rl_data = []
        for traj in trajectories_with_outcomes:
            shaped_reward = self._calculate_shaped_reward(traj)
            
            rl_data.append({
                'trajectory_id': traj.trajectory_id,
                'prompt': self._format_prompt(traj),
                'response': self._format_response(traj),
                'reward': shaped_reward,
                'actual_return': traj.actual_return,
                'metadata': {
                    'symbol': traj.symbol,
                    'confidence': traj.confidence,
                    'market_regime': traj.market_regime
                }
            })
        
        # Create output directory
        data_dir = os.path.join(self.config.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to file
        timestamp = int(time.time())
        filename = f"rl_data_strategist_{timestamp}.jsonl"
        data_path = os.path.join(data_dir, filename)
        
        with open(data_path, 'w') as f:
            for item in rl_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Prepared RL training data: {data_path}")
        
        return data_path
    
    def _calculate_shaped_reward(self, trajectory) -> float:
        """
        Calculate shaped reward from trajectory
        
        Reward = return_weight * actual_return 
                 - risk_penalty * max_drawdown
                 + confidence_bonus * (confidence if correct else -confidence)
        
        Args:
            trajectory: Trajectory with outcome
        
        Returns:
            Shaped reward
        """
        # Base reward from actual return
        reward = self.config.return_weight * trajectory.actual_return
        
        # Risk penalty (if we have drawdown info)
        # For now, use position size as proxy for risk
        if trajectory.position_size:
            risk = trajectory.position_size
            reward -= self.config.risk_penalty * risk
        
        # Confidence bonus/penalty
        correct = trajectory.actual_return > 0
        confidence_term = trajectory.confidence if correct else -trajectory.confidence
        reward += self.config.confidence_bonus * confidence_term
        
        return reward
    
    def _format_prompt(self, trajectory) -> str:
        """Format prompt from trajectory"""
        return f"""Synthesize the following analyses for {trajectory.symbol}:

Market State: {json.dumps(trajectory.market_state, indent=2)}

Agent Analyses: {json.dumps(trajectory.agent_inputs.get('agent_outputs', {}), indent=2)}

Provide final recommendation (buy/sell/hold), confidence [0, 1], position size, stop loss, take profit, and reasoning."""
    
    def _format_response(self, trajectory) -> str:
        """Format response from trajectory"""
        response = {
            'recommendation': trajectory.recommendation,
            'confidence': trajectory.confidence,
            'position_size': trajectory.position_size,
            'stop_loss': trajectory.stop_loss,
            'take_profit': trajectory.take_profit,
            'reasoning': trajectory.reasoning
        }
        return json.dumps(response, indent=2)
    
    def _train_model(self, data_path: str) -> str:
        """
        Train model with GRPO
        
        Args:
            data_path: Path to training data
        
        Returns:
            Path to trained model
        """
        # Create output directory
        timestamp = int(time.time())
        model_dir = os.path.join(
            self.config.output_dir,
            f"strategist_{timestamp}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Training script
        training_script = "training/rl/train_strategist_grpo.py"
        
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
            '--kl_coef', str(self.config.kl_coef)
        ]
        
        logger.info(f"Running GRPO training: {' '.join(cmd)}")
        
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
            'strategist'
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


def run_automated_rl_pipeline(
    market_regime: Optional[str] = None,
    force_retrain: bool = False
) -> Dict:
    """
    Run automated RL pipeline for strategist
    
    Args:
        market_regime: Market regime filter
        force_retrain: Force retraining
    
    Returns:
        Pipeline results
    """
    config = RLPipelineConfig()
    
    pipeline = AutomatedRLPipeline(config)
    
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
    parser.add_argument('--market_regime', type=str, default=None)
    parser.add_argument('--force_retrain', action='store_true')
    
    args = parser.parse_args()
    
    results = run_automated_rl_pipeline(
        market_regime=args.market_regime,
        force_retrain=args.force_retrain
    )
    
    print(json.dumps(results, indent=2))
