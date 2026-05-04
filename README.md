# gnn-marl-traffic

## Training Commands

### 1) Normal training (lane order deterministic by default)

No extra flag is needed for deterministic lane ordering.  
Just run training with your scenario/seed:

```powershell
python experiments/train.py --scenario grid_2x2 --agent gat_dqn --episodes 50 --seed 42
```

### 2) Resume training from checkpoint

Use `--resume-checkpoint` and optionally `--resume-log-dir`:

```powershell
python experiments/train.py --scenario grid_3x3 --agent gat_dqn --episodes 300 --seed 42 --resume-checkpoint logs/gat_dqn_grid_3x3_20260503_120000/checkpoint_ep100.pt --resume-log-dir logs/gat_dqn_grid_3x3_20260503_120000
```

Notes:
- `--episodes` is the final target episode count (not additional episodes).
- If `--resume-log-dir` is omitted, training continues using the checkpoint folder.
