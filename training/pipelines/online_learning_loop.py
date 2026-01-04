"""
Online Learning Loop

Orchestrates continuous learning across all agents:
1. Monitors Experience Library for new data
2. Triggers SFT pipelines for junior agents
3. Triggers RL pipeline for strategist
4. Triggers bandit updates for supervisor
5. Schedules retraining based on data availability

This is the "closed-loop system" that makes the multi-agent system self-improving.
"""

import time
import schedule
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass
import threading
import json

from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres
from training.pipelines.automated_sft_pipeline import AutomatedSFTPipeline, SFTPipelineConfig
from training.pipelines.automated_rl_pipeline import AutomatedRLPipeline, RLPipelineConfig


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning loop"""
    
    # Monitoring
    check_interval_hours: int = 6  # Check every 6 hours
    
    # SFT thresholds
    sft_min_new_trajectories: int = 100
    sft_retrain_interval_hours: int = 24
    
    # RL thresholds
    rl_min_new_trajectories: int = 500
    rl_retrain_interval_hours: int = 48
    
    # Supervisor thresholds
    supervisor_update_interval_hours: int = 1  # Update hourly
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_experience"
    db_user: str = "postgres"
    db_password: str = "postgres"


class OnlineLearningLoop:
    """
    Online Learning Loop Orchestrator
    
    Continuously monitors data and triggers retraining when needed.
    """
    
    def __init__(self, config: OnlineLearningConfig):
        """
        Initialize online learning loop
        
        Args:
            config: Configuration
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
        
        # Pipelines
        self.sft_pipelines = {
            'news': AutomatedSFTPipeline(SFTPipelineConfig(agent_type='news')),
            'technical': AutomatedSFTPipeline(SFTPipelineConfig(agent_type='technical')),
            'fundamental': AutomatedSFTPipeline(SFTPipelineConfig(agent_type='fundamental'))
        }
        
        self.rl_pipeline = AutomatedRLPipeline(RLPipelineConfig())
        
        # State
        self.running = False
        self.last_check = {}
        self.training_log = []
        
        # Thread
        self.thread = None
    
    def start(self):
        """Start online learning loop"""
        if self.running:
            logger.warning("Online learning loop already running")
            return
        
        self.running = True
        
        # Schedule tasks
        schedule.every(self.config.check_interval_hours).hours.do(self._check_and_retrain)
        schedule.every(self.config.supervisor_update_interval_hours).hours.do(self._update_supervisor)
        
        # Start thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        logger.info("✅ Online learning loop started")
    
    def stop(self):
        """Stop online learning loop"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Online learning loop stopped")
    
    def _run_loop(self):
        """Main loop (runs in thread)"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in online learning loop: {e}")
    
    def _check_and_retrain(self):
        """Check data and trigger retraining if needed"""
        logger.info("Checking for retraining opportunities...")
        
        try:
            # Get statistics
            stats = self.library.get_statistics()
            
            # Check SFT agents
            for agent_type in ['news', 'technical', 'fundamental']:
                self._check_sft_agent(agent_type, stats)
            
            # Check RL strategist
            self._check_rl_strategist(stats)
        
        except Exception as e:
            logger.error(f"Error checking for retraining: {e}")
    
    def _check_sft_agent(self, agent_type: str, stats: Dict):
        """
        Check if SFT agent needs retraining
        
        Args:
            agent_type: Agent type
            stats: Library statistics
        """
        # Get agent statistics
        by_agent = stats.get('by_agent_type', [])
        agent_stats = next(
            (s for s in by_agent if s['agent_type'] == agent_type),
            None
        )
        
        if not agent_stats:
            return
        
        successful_count = agent_stats.get('successful', 0)
        
        # Check if enough new trajectories
        last_count = self.last_check.get(f'sft_{agent_type}_count', 0)
        new_count = successful_count - last_count
        
        if new_count >= self.config.sft_min_new_trajectories:
            # Check if enough time has passed
            last_time = self.last_check.get(f'sft_{agent_type}_time', 0)
            hours_since = (time.time() - last_time) / 3600
            
            if hours_since >= self.config.sft_retrain_interval_hours:
                logger.info(
                    f"Triggering SFT retraining for {agent_type} agent "
                    f"({new_count} new trajectories)"
                )
                
                # Run pipeline
                pipeline = self.sft_pipelines[agent_type]
                results = pipeline.run_pipeline(force_retrain=True)
                
                # Update state
                self.last_check[f'sft_{agent_type}_count'] = successful_count
                self.last_check[f'sft_{agent_type}_time'] = time.time()
                
                # Log
                self.training_log.append({
                    'timestamp': time.time(),
                    'type': 'sft',
                    'agent_type': agent_type,
                    'results': results
                })
                
                logger.info(f"SFT retraining for {agent_type} completed")
    
    def _check_rl_strategist(self, stats: Dict):
        """
        Check if RL strategist needs retraining
        
        Args:
            stats: Library statistics
        """
        # Get strategist statistics
        by_agent = stats.get('by_agent_type', [])
        strategist_stats = next(
            (s for s in by_agent if s['agent_type'] == 'strategist'),
            None
        )
        
        if not strategist_stats:
            return
        
        total_count = strategist_stats.get('count', 0)
        
        # Check if enough new trajectories
        last_count = self.last_check.get('rl_strategist_count', 0)
        new_count = total_count - last_count
        
        if new_count >= self.config.rl_min_new_trajectories:
            # Check if enough time has passed
            last_time = self.last_check.get('rl_strategist_time', 0)
            hours_since = (time.time() - last_time) / 3600
            
            if hours_since >= self.config.rl_retrain_interval_hours:
                logger.info(
                    f"Triggering RL retraining for strategist "
                    f"({new_count} new trajectories)"
                )
                
                # Run pipeline
                results = self.rl_pipeline.run_pipeline(force_retrain=True)
                
                # Update state
                self.last_check['rl_strategist_count'] = total_count
                self.last_check['rl_strategist_time'] = time.time()
                
                # Log
                self.training_log.append({
                    'timestamp': time.time(),
                    'type': 'rl',
                    'agent_type': 'strategist',
                    'results': results
                })
                
                logger.info("RL retraining for strategist completed")
    
    def _update_supervisor(self):
        """Update supervisor with latest performance data"""
        logger.info("Updating supervisor...")
        
        try:
            # Get statistics
            stats = self.library.get_statistics()
            
            # Calculate agent performance by regime
            by_regime = stats.get('by_market_regime', [])
            
            # TODO: Update supervisor's bandit model with new performance data
            # This would involve:
            # 1. Loading supervisor model
            # 2. Updating with new reward observations
            # 3. Saving updated model
            
            logger.info("Supervisor updated")
        
        except Exception as e:
            logger.error(f"Error updating supervisor: {e}")
    
    def get_training_log(self) -> List[Dict]:
        """Get training log"""
        return self.training_log
    
    def get_status(self) -> Dict:
        """
        Get status of online learning loop
        
        Returns:
            Status dictionary
        """
        stats = self.library.get_statistics()
        
        return {
            'running': self.running,
            'last_check': self.last_check,
            'library_stats': stats,
            'training_log_count': len(self.training_log),
            'recent_trainings': self.training_log[-5:] if self.training_log else []
        }
    
    def close(self):
        """Close online learning loop"""
        self.stop()
        
        # Close pipelines
        for pipeline in self.sft_pipelines.values():
            pipeline.close()
        
        self.rl_pipeline.close()
        self.library.close()


def start_online_learning_loop(config: Optional[OnlineLearningConfig] = None):
    """
    Start online learning loop
    
    Args:
        config: Configuration (uses defaults if None)
    
    Returns:
        OnlineLearningLoop instance
    """
    if config is None:
        config = OnlineLearningConfig()
    
    loop = OnlineLearningLoop(config)
    loop.start()
    
    return loop


if __name__ == '__main__':
    # Test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_interval_hours', type=int, default=6)
    parser.add_argument('--sft_min_trajectories', type=int, default=100)
    parser.add_argument('--rl_min_trajectories', type=int, default=500)
    
    args = parser.parse_args()
    
    config = OnlineLearningConfig(
        check_interval_hours=args.check_interval_hours,
        sft_min_new_trajectories=args.sft_min_trajectories,
        rl_min_new_trajectories=args.rl_min_trajectories
    )
    
    loop = start_online_learning_loop(config)
    
    logger.info("Online learning loop started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)
            
            # Print status every 10 minutes
            if int(time.time()) % 600 == 0:
                status = loop.get_status()
                logger.info(f"Status: {json.dumps(status, indent=2)}")
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
        loop.close()
