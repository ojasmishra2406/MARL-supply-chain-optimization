import argparse
from rl.phase10_trainer import Phase10Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase4_ippo.yaml")
    args = parser.parse_args()
    
    # We want to run MAPPO (centralized critic) with the GNN
    ablation = {
        "communication": False,
        "centralized_critic": True,
        "parameter_sharing": False
    }
    
    print("Starting GNN-MAPPO Training...")
    trainer = Phase10Trainer(args.config, ablation_config=ablation)
    
    # Train for a few iterations just to test integration
    trainer.train(total_updates=2, use_wandb=False)
    print("GNN-MAPPO end-to-end test completed successfully!")

if __name__ == "__main__":
    main()
