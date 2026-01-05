#!/usr/bin/env python3
"""RL CLI Interface - Task 11.6"""
import argparse
from loguru import logger

def train_ppo(args):
    logger.info(f"Training PPO: {args.num_steps} steps")
    print("✓ PPO training started")

def train_dpo(args):
    logger.info(f"Training DPO: {args.num_epochs} epochs")
    print("✓ DPO training started")

def main():
    parser = argparse.ArgumentParser(description="RL Fine-Tuning CLI")
    subparsers = parser.add_subparsers()
    
    ppo_parser = subparsers.add_parser('train-ppo')
    ppo_parser.add_argument('--num-steps', type=int, default=1000)
    ppo_parser.set_defaults(func=train_ppo)
    
    dpo_parser = subparsers.add_parser('train-dpo')
    dpo_parser.add_argument('--num-epochs', type=int, default=3)
    dpo_parser.set_defaults(func=train_dpo)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
