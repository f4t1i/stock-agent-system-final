#!/usr/bin/env python3
"""
Model Deployment Script

Deploys trained models to production:
1. Validate model quality
2. Backup current models
3. Deploy new models
4. Update configuration
5. Verify deployment
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelDeployer:
    """
    Deploy trained models to production.

    Safety features:
    - Model validation before deployment
    - Automatic backup of current models
    - Rollback capability
    - Deployment verification
    """

    def __init__(
        self,
        production_dir: str = "models/production",
        backup_dir: str = "models/backups"
    ):
        """
        Initialize deployer.

        Args:
            production_dir: Production models directory
            backup_dir: Backup directory
        """
        self.production_dir = Path(production_dir)
        self.backup_dir = Path(backup_dir)

        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Model Deployer initialized")
        logger.info(f"Production directory: {self.production_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Deployment ID: {self.deployment_id}")

    def validate_model(
        self,
        model_path: str,
        min_sharpe: float = 0.5,
        evaluation_file: str = None
    ) -> bool:
        """
        Validate model meets deployment criteria.

        Args:
            model_path: Path to model
            min_sharpe: Minimum Sharpe ratio required
            evaluation_file: Path to evaluation results

        Returns:
            True if model passes validation
        """
        logger.info(f"Validating model: {model_path}")

        model_path = Path(model_path)

        # Check model exists
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        # Check for required files
        required_files = ['config.json', 'pytorch_model.bin']

        for file_name in required_files:
            if not (model_path / file_name).exists():
                logger.warning(f"Missing file: {file_name}")

        # Validate performance if evaluation file provided
        if evaluation_file:
            eval_file = Path(evaluation_file)

            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)

                # Check Sharpe ratio
                sharpe = eval_results.get('best_model', {}).get('sharpe_ratio', 0)

                if sharpe < min_sharpe:
                    logger.error(f"Model Sharpe ratio {sharpe:.3f} below minimum {min_sharpe:.3f}")
                    return False

                logger.info(f"✓ Model Sharpe ratio: {sharpe:.3f}")

        logger.info(f"✓ Model validation passed")
        return True

    def backup_current_models(self) -> str:
        """
        Backup current production models.

        Returns:
            Backup directory path
        """
        logger.info("Backing up current models")

        backup_path = self.backup_dir / f"backup_{self.deployment_id}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup all model directories
        backed_up = 0

        for model_dir in self.production_dir.iterdir():
            if model_dir.is_dir():
                backup_dest = backup_path / model_dir.name

                logger.info(f"Backing up {model_dir.name}...")
                shutil.copytree(model_dir, backup_dest)

                backed_up += 1

        logger.info(f"✓ Backed up {backed_up} models to {backup_path}")

        return str(backup_path)

    def deploy_model(
        self,
        model_path: str,
        agent_type: str,
        force: bool = False
    ):
        """
        Deploy model to production.

        Args:
            model_path: Source model path
            agent_type: Agent type (news, technical, fundamental, strategist)
            force: Force deployment without validation
        """
        logger.info(f"Deploying {agent_type} model")

        model_path = Path(model_path)
        dest_path = self.production_dir / agent_type

        # Validate unless forced
        if not force:
            if not self.validate_model(str(model_path)):
                logger.error("Model validation failed, aborting deployment")
                raise ValueError("Model validation failed")

        # Remove existing model
        if dest_path.exists():
            logger.info(f"Removing existing {agent_type} model")
            shutil.rmtree(dest_path)

        # Copy new model
        logger.info(f"Copying model to {dest_path}")
        shutil.copytree(model_path, dest_path)

        logger.info(f"✓ {agent_type} model deployed")

    def update_config(
        self,
        agent_configs: Dict[str, str]
    ):
        """
        Update system configuration with new model paths.

        Args:
            agent_configs: Dict mapping agent type to model path
        """
        logger.info("Updating system configuration")

        config_file = Path("config/system.yaml")

        if not config_file.exists():
            logger.warning("System config not found, creating new one")
            config = {'agents': {}}
        else:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

        # Update model paths
        for agent_type, model_path in agent_configs.items():
            if agent_type not in config['agents']:
                config['agents'][agent_type] = {}

            config['agents'][agent_type]['model_path'] = model_path
            config['agents'][agent_type]['enabled'] = True

        # Save updated config
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"✓ Configuration updated")

    def verify_deployment(self) -> bool:
        """
        Verify deployment succeeded.

        Returns:
            True if deployment is valid
        """
        logger.info("Verifying deployment")

        # Check all expected models exist
        expected_agents = ['news', 'technical', 'fundamental', 'strategist']
        all_exist = True

        for agent in expected_agents:
            model_path = self.production_dir / agent

            if model_path.exists():
                logger.info(f"✓ {agent}: Found")
            else:
                logger.warning(f"✗ {agent}: Not found")
                all_exist = False

        if all_exist:
            logger.info("✓ Deployment verification passed")
        else:
            logger.warning("⚠ Some models missing, deployment may be incomplete")

        return all_exist

    def rollback(
        self,
        backup_path: str
    ):
        """
        Rollback to previous models.

        Args:
            backup_path: Path to backup
        """
        logger.info(f"Rolling back to backup: {backup_path}")

        backup_path = Path(backup_path)

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            raise ValueError("Backup not found")

        # Restore each model
        for model_dir in backup_path.iterdir():
            if model_dir.is_dir():
                dest_path = self.production_dir / model_dir.name

                logger.info(f"Restoring {model_dir.name}...")

                if dest_path.exists():
                    shutil.rmtree(dest_path)

                shutil.copytree(model_dir, dest_path)

        logger.info("✓ Rollback complete")

    def deploy_pipeline(
        self,
        model_configs: List[Dict],
        min_sharpe: float = 0.5,
        force: bool = False,
        skip_backup: bool = False
    ):
        """
        Deploy multiple models.

        Args:
            model_configs: List of model configurations
            min_sharpe: Minimum Sharpe ratio
            force: Force deployment
            skip_backup: Skip backup step
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL DEPLOYMENT PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Deployment ID: {self.deployment_id}")
        logger.info(f"Models to deploy: {len(model_configs)}")
        logger.info(f"{'='*60}\n")

        backup_path = None

        try:
            # Step 1: Backup current models
            if not skip_backup:
                logger.info("[STEP 1/4] Backing up current models")
                backup_path = self.backup_current_models()
            else:
                logger.info("[STEP 1/4] Skipping backup (--skip-backup)")

            # Step 2: Validate all models
            logger.info("\n[STEP 2/4] Validating models")

            for config in model_configs:
                if not force:
                    valid = self.validate_model(
                        model_path=config['model_path'],
                        min_sharpe=min_sharpe,
                        evaluation_file=config.get('evaluation_file')
                    )

                    if not valid:
                        raise ValueError(f"Model validation failed: {config['agent_type']}")

            # Step 3: Deploy models
            logger.info("\n[STEP 3/4] Deploying models")

            agent_configs = {}

            for config in model_configs:
                self.deploy_model(
                    model_path=config['model_path'],
                    agent_type=config['agent_type'],
                    force=force
                )

                agent_configs[config['agent_type']] = str(
                    self.production_dir / config['agent_type']
                )

            # Update configuration
            self.update_config(agent_configs)

            # Step 4: Verify deployment
            logger.info("\n[STEP 4/4] Verifying deployment")

            if not self.verify_deployment():
                raise ValueError("Deployment verification failed")

            # Success
            logger.info(f"\n{'='*60}")
            logger.info("DEPLOYMENT SUCCESS")
            logger.info(f"{'='*60}")
            logger.info(f"Deployment ID: {self.deployment_id}")
            logger.info(f"Models deployed: {len(model_configs)}")
            if backup_path:
                logger.info(f"Backup location: {backup_path}")
            logger.info(f"Production directory: {self.production_dir}")
            logger.info(f"{'='*60}\n")

            # Save deployment record
            self._save_deployment_record(model_configs, backup_path)

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

            if backup_path and not skip_backup:
                logger.info("Attempting rollback...")
                try:
                    self.rollback(backup_path)
                    logger.info("✓ Rollback successful")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")

            raise

    def _save_deployment_record(
        self,
        model_configs: List[Dict],
        backup_path: str
    ):
        """Save deployment record"""

        record = {
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'models': model_configs,
            'backup_path': backup_path,
            'production_dir': str(self.production_dir)
        }

        record_file = self.production_dir / f"deployment_{self.deployment_id}.json"

        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)

        logger.info(f"✓ Deployment record saved: {record_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy trained models to production"
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Models to deploy (format: agent_type:model_path[:eval_file])'
    )

    parser.add_argument(
        '--min-sharpe',
        type=float,
        default=0.5,
        help='Minimum Sharpe ratio for deployment (default: 0.5)'
    )

    parser.add_argument(
        '--production-dir',
        type=str,
        default='models/production',
        help='Production directory (default: models/production)'
    )

    parser.add_argument(
        '--backup-dir',
        type=str,
        default='models/backups',
        help='Backup directory (default: models/backups)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force deployment without validation'
    )

    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip backup step (not recommended)'
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Parse model configs
    model_configs = []

    for model_spec in args.models:
        parts = model_spec.split(':')

        if len(parts) < 2:
            logger.error(f"Invalid model spec: {model_spec}")
            continue

        config = {
            'agent_type': parts[0],
            'model_path': parts[1]
        }

        if len(parts) >= 3:
            config['evaluation_file'] = parts[2]

        model_configs.append(config)

    # Deploy
    deployer = ModelDeployer(
        production_dir=args.production_dir,
        backup_dir=args.backup_dir
    )

    deployer.deploy_pipeline(
        model_configs=model_configs,
        min_sharpe=args.min_sharpe,
        force=args.force,
        skip_backup=args.skip_backup
    )


if __name__ == '__main__':
    main()
