#!/usr/bin/env python3
"""
Complete Training Pipeline - 1-2 Week Plan

Orchestrates the complete training workflow:
Week 1: SFT Training (100+ trajectories)
Week 2: RL Training (500+ trajectories)
Final: Evaluation & Deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

from loguru import logger
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_sft_training import SFTTrainingOrchestrator
from scripts.run_rl_training import RLTrainingOrchestrator
from scripts.evaluate_models import ModelEvaluator
from scripts.deploy_models import ModelDeployer


class CompleteTrainingPipeline:
    """
    Complete 1-2 week training pipeline.

    Workflow:
    1. Week 1: SFT Training
       - Generate synthetic data (100+ examples/agent)
       - Train news, technical, fundamental agents
       - Evaluate SFT models

    2. Week 2: RL Training
       - Collect trajectories (500+)
       - Train strategist (GRPO/PPO)
       - Evaluate RL models

    3. Deployment
       - Compare all models
       - Deploy best models
       - Verify deployment
    """

    def __init__(
        self,
        output_dir: str = "training_runs",
        use_wandb: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            output_dir: Base output directory
            use_wandb: Enable W&B tracking
        """
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb

        # Create pipeline run directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"complete_pipeline_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="stock-agent-system",
                name=f"complete-pipeline-{self.run_id}",
                tags=["complete-pipeline", "sft", "rl"],
                config={
                    "run_id": self.run_id,
                    "pipeline": "complete-1-2-week"
                }
            )

        logger.info("="*70)
        logger.info("COMPLETE TRAINING PIPELINE - 1-2 WEEK PLAN")
        logger.info("="*70)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output: {self.run_dir}")
        logger.info(f"W&B Tracking: {'Enabled' if use_wandb else 'Disabled'}")
        logger.info("="*70 + "\n")

    def week1_sft_training(
        self,
        num_examples: int = 100,
        max_steps: int = 500,
        provider: str = "openai",
        model: str = "gpt-4o"
    ) -> Dict:
        """
        Week 1: SFT Training.

        Args:
            num_examples: Examples per agent
            max_steps: Training steps per agent
            provider: LLM provider for data generation
            model: LLM model for data generation

        Returns:
            SFT training results
        """
        logger.info("\n" + "="*70)
        logger.info("WEEK 1: SFT TRAINING")
        logger.info("="*70)
        logger.info(f"Examples per agent: {num_examples}")
        logger.info(f"Training steps: {max_steps}")
        logger.info(f"Data generation: {provider}/{model}")
        logger.info("="*70 + "\n")

        # Initialize SFT orchestrator
        sft_orchestrator = SFTTrainingOrchestrator(
            output_dir=str(self.run_dir / "sft"),
            use_wandb=self.use_wandb,
            provider=provider,
            model=model
        )

        # Run SFT pipeline
        sft_results = sft_orchestrator.run_full_pipeline(
            num_examples=num_examples,
            max_steps=max_steps
        )

        # Save results
        sft_results_file = self.run_dir / "week1_sft_results.json"
        with open(sft_results_file, 'w') as f:
            json.dump(sft_results, f, indent=2)

        if self.use_wandb:
            wandb.log({
                "week1/sft_complete": True,
                "week1/successful_agents": sum(
                    1 for r in sft_results.values() if r['status'] == 'success'
                )
            })

        logger.info("\n✓ Week 1 (SFT Training) Complete\n")

        return sft_results

    def week2_rl_training(
        self,
        num_trajectories: int = 500,
        num_iterations: int = 100,
        algorithm: str = "grpo",
        symbols: list = None
    ) -> Dict:
        """
        Week 2: RL Training.

        Args:
            num_trajectories: Number of trajectories to collect
            num_iterations: Training iterations
            algorithm: RL algorithm (grpo or ppo)
            symbols: Trading symbols

        Returns:
            RL training results
        """
        logger.info("\n" + "="*70)
        logger.info("WEEK 2: RL TRAINING")
        logger.info("="*70)
        logger.info(f"Algorithm: {algorithm.upper()}")
        logger.info(f"Trajectories: {num_trajectories}")
        logger.info(f"Iterations: {num_iterations}")
        logger.info("="*70 + "\n")

        # Initialize RL orchestrator
        rl_orchestrator = RLTrainingOrchestrator(
            output_dir=str(self.run_dir / "rl"),
            use_wandb=self.use_wandb,
            algorithm=algorithm
        )

        # Run RL pipeline
        try:
            rl_orchestrator.run_full_pipeline(
                num_trajectories=num_trajectories,
                num_iterations=num_iterations,
                symbols=symbols
            )

            rl_results = {
                'status': 'success',
                'trajectories': num_trajectories,
                'iterations': num_iterations,
                'algorithm': algorithm
            }

        except Exception as e:
            logger.error(f"RL training failed: {e}")
            rl_results = {
                'status': 'failed',
                'error': str(e)
            }

        # Save results
        rl_results_file = self.run_dir / "week2_rl_results.json"
        with open(rl_results_file, 'w') as f:
            json.dump(rl_results, f, indent=2)

        if self.use_wandb:
            wandb.log({
                "week2/rl_complete": rl_results['status'] == 'success',
                "week2/trajectories": num_trajectories
            })

        logger.info("\n✓ Week 2 (RL Training) Complete\n")

        return rl_results

    def evaluate_and_deploy(
        self,
        sft_results: Dict,
        rl_results: Dict,
        min_sharpe: float = 0.5,
        test_symbols: list = None,
        test_days: int = 90,
        auto_deploy: bool = False
    ):
        """
        Evaluate models and optionally deploy.

        Args:
            sft_results: SFT training results
            rl_results: RL training results
            min_sharpe: Minimum Sharpe for deployment
            test_symbols: Test symbols
            test_days: Test period days
            auto_deploy: Auto-deploy if validation passes
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUATION & DEPLOYMENT")
        logger.info("="*70)

        if test_symbols is None:
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        # Prepare model configs for evaluation
        model_configs = []

        # Add baseline (if exists)
        baseline_path = Path("models/baseline")
        if baseline_path.exists():
            model_configs.append({
                'name': 'baseline',
                'model_path': str(baseline_path)
            })

        # Add SFT models
        sft_dir = self.run_dir / "sft"
        for sft_run in sft_dir.glob("sft_run_*"):
            models_dir = sft_run / "models"
            if models_dir.exists():
                model_configs.append({
                    'name': f'sft_{sft_run.name}',
                    'model_path': str(models_dir)
                })

        # Add RL models
        rl_dir = self.run_dir / "rl"
        for rl_run in rl_dir.glob("rl_*_run_*"):
            models_dir = rl_run / "models"
            if models_dir.exists():
                model_configs.append({
                    'name': f'rl_{rl_run.name}',
                    'model_path': str(models_dir)
                })

        if not model_configs:
            logger.warning("No models found for evaluation")
            return

        logger.info(f"Evaluating {len(model_configs)} model(s)")

        # Evaluate models
        evaluator = ModelEvaluator(
            output_dir=str(self.run_dir / "evaluation")
        )

        eval_results = evaluator.evaluate_pipeline(
            model_configs=model_configs,
            test_symbols=test_symbols,
            test_days=test_days
        )

        if self.use_wandb:
            wandb.log({
                "evaluation/models_evaluated": len(eval_results['evaluations']),
                "evaluation/best_sharpe": eval_results['best_model']['sharpe_ratio']
            })

        # Check if best model meets deployment criteria
        best_sharpe = eval_results['best_model']['sharpe_ratio']

        if best_sharpe < min_sharpe:
            logger.warning(
                f"Best model Sharpe ({best_sharpe:.3f}) below minimum ({min_sharpe:.3f})"
            )
            logger.warning("Deployment not recommended")

            if self.use_wandb:
                wandb.log({"deployment/recommended": False})

            return

        logger.info(f"✓ Best model meets deployment criteria (Sharpe: {best_sharpe:.3f})")

        # Deploy if auto-deploy enabled
        if auto_deploy:
            logger.info("\nAuto-deployment enabled, deploying best models...")

            self._deploy_best_models(
                sft_results=sft_results,
                rl_results=rl_results,
                eval_results=eval_results,
                min_sharpe=min_sharpe
            )

        else:
            logger.info("\nAuto-deployment disabled. To deploy, run:")
            logger.info(f"  python scripts/deploy_models.py --models ...")

        logger.info("\n✓ Evaluation Complete\n")

    def _deploy_best_models(
        self,
        sft_results: Dict,
        rl_results: Dict,
        eval_results: Dict,
        min_sharpe: float
    ):
        """Deploy best models"""

        logger.info("Deploying best models...")

        # Find model paths
        deploy_configs = []

        # SFT models
        sft_dir = self.run_dir / "sft"
        for sft_run in sft_dir.glob("sft_run_*"):
            models_dir = sft_run / "models"

            for agent_type in ['news', 'technical', 'fundamental']:
                agent_dir = models_dir / agent_type / 'final'
                if agent_dir.exists():
                    deploy_configs.append({
                        'agent_type': agent_type,
                        'model_path': str(agent_dir),
                        'evaluation_file': str(self.run_dir / "evaluation" / "eval_*" / "evaluation_results.json")
                    })

        # RL model
        rl_dir = self.run_dir / "rl"
        for rl_run in rl_dir.glob("rl_*_run_*"):
            strategist_dir = rl_run / "models" / "final"
            if strategist_dir.exists():
                deploy_configs.append({
                    'agent_type': 'strategist',
                    'model_path': str(strategist_dir),
                    'evaluation_file': str(self.run_dir / "evaluation" / "eval_*" / "evaluation_results.json")
                })

        if not deploy_configs:
            logger.error("No models found for deployment")
            return

        # Deploy
        deployer = ModelDeployer()

        try:
            deployer.deploy_pipeline(
                model_configs=deploy_configs,
                min_sharpe=min_sharpe,
                force=False,
                skip_backup=False
            )

            if self.use_wandb:
                wandb.log({
                    "deployment/deployed": True,
                    "deployment/models_count": len(deploy_configs)
                })

            logger.info("✓ Deployment Complete")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

            if self.use_wandb:
                wandb.log({"deployment/deployed": False})

    def run_complete_pipeline(
        self,
        # SFT parameters
        sft_examples: int = 100,
        sft_steps: int = 500,
        data_provider: str = "openai",
        data_model: str = "gpt-4o",
        # RL parameters
        rl_trajectories: int = 500,
        rl_iterations: int = 100,
        rl_algorithm: str = "grpo",
        rl_symbols: list = None,
        # Evaluation parameters
        test_symbols: list = None,
        test_days: int = 90,
        min_sharpe: float = 0.5,
        # Deployment
        auto_deploy: bool = False
    ):
        """
        Run complete 1-2 week training pipeline.

        Args:
            sft_examples: SFT examples per agent
            sft_steps: SFT training steps
            data_provider: Data generation provider
            data_model: Data generation model
            rl_trajectories: RL trajectories to collect
            rl_iterations: RL training iterations
            rl_algorithm: RL algorithm
            rl_symbols: Trading symbols
            test_symbols: Evaluation symbols
            test_days: Evaluation period
            min_sharpe: Minimum Sharpe for deployment
            auto_deploy: Auto-deploy if validation passes
        """
        start_time = datetime.now()

        logger.info("Starting complete 1-2 week training pipeline\n")

        try:
            # Week 1: SFT Training
            sft_results = self.week1_sft_training(
                num_examples=sft_examples,
                max_steps=sft_steps,
                provider=data_provider,
                model=data_model
            )

            # Week 2: RL Training
            rl_results = self.week2_rl_training(
                num_trajectories=rl_trajectories,
                num_iterations=rl_iterations,
                algorithm=rl_algorithm,
                symbols=rl_symbols
            )

            # Evaluation & Deployment
            self.evaluate_and_deploy(
                sft_results=sft_results,
                rl_results=rl_results,
                min_sharpe=min_sharpe,
                test_symbols=test_symbols,
                test_days=test_days,
                auto_deploy=auto_deploy
            )

            # Final summary
            duration = datetime.now() - start_time

            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETE")
            logger.info("="*70)
            logger.info(f"Run ID: {self.run_id}")
            logger.info(f"Duration: {duration}")
            logger.info(f"Output: {self.run_dir}")
            logger.info("="*70 + "\n")

            # Save final summary
            summary = {
                'run_id': self.run_id,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'sft_results': sft_results,
                'rl_results': rl_results,
                'parameters': {
                    'sft_examples': sft_examples,
                    'sft_steps': sft_steps,
                    'rl_trajectories': rl_trajectories,
                    'rl_iterations': rl_iterations,
                    'rl_algorithm': rl_algorithm
                }
            }

            summary_file = self.run_dir / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary saved to {summary_file}")

            if self.use_wandb:
                wandb.log({
                    "pipeline/complete": True,
                    "pipeline/duration_hours": duration.total_seconds() / 3600
                })
                wandb.finish()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

            if self.use_wandb:
                wandb.log({"pipeline/complete": False})
                wandb.finish()

            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete 1-2 week training pipeline"
    )

    # SFT parameters
    sft_group = parser.add_argument_group('SFT Training (Week 1)')
    sft_group.add_argument('--sft-examples', type=int, default=100,
                          help='Examples per agent (default: 100)')
    sft_group.add_argument('--sft-steps', type=int, default=500,
                          help='Training steps (default: 500)')
    sft_group.add_argument('--data-provider', type=str, default='openai',
                          choices=['openai', 'anthropic'],
                          help='Data generation provider')
    sft_group.add_argument('--data-model', type=str, default='gpt-4o',
                          help='Data generation model')

    # RL parameters
    rl_group = parser.add_argument_group('RL Training (Week 2)')
    rl_group.add_argument('--rl-trajectories', type=int, default=500,
                         help='Trajectories to collect (default: 500)')
    rl_group.add_argument('--rl-iterations', type=int, default=100,
                         help='Training iterations (default: 100)')
    rl_group.add_argument('--rl-algorithm', type=str, default='grpo',
                         choices=['grpo', 'ppo'],
                         help='RL algorithm (default: grpo)')
    rl_group.add_argument('--rl-symbols', type=str, nargs='+',
                         help='Trading symbols (default: top 15)')

    # Evaluation parameters
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--test-symbols', type=str, nargs='+',
                           default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                           help='Test symbols')
    eval_group.add_argument('--test-days', type=int, default=90,
                           help='Test period days (default: 90)')
    eval_group.add_argument('--min-sharpe', type=float, default=0.5,
                           help='Minimum Sharpe for deployment (default: 0.5)')

    # General parameters
    parser.add_argument('--output-dir', type=str, default='training_runs',
                       help='Output directory')
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Auto-deploy if validation passes')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B tracking')

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run pipeline
    pipeline = CompleteTrainingPipeline(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )

    pipeline.run_complete_pipeline(
        sft_examples=args.sft_examples,
        sft_steps=args.sft_steps,
        data_provider=args.data_provider,
        data_model=args.data_model,
        rl_trajectories=args.rl_trajectories,
        rl_iterations=args.rl_iterations,
        rl_algorithm=args.rl_algorithm,
        rl_symbols=args.rl_symbols,
        test_symbols=args.test_symbols,
        test_days=args.test_days,
        min_sharpe=args.min_sharpe,
        auto_deploy=args.auto_deploy
    )


if __name__ == '__main__':
    main()
