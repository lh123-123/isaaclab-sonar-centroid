# IsaacLab Claude Code Setup

This project uses Claude Code with specialized skills for IsaacLab and reinforcement learning development.

## Available Skills

### Deep RL Skills
- **spinning-up-deep-rl**: Knowledge base from OpenAI's "Spinning Up in Deep RL" covering algorithms (SAC, PPO, DDPG, etc.), debugging, and experimentation
- **karpathy-coding-guidelines**: Behavioral guidelines to reduce common LLM coding mistakes
- **performance-profiler**: Systematic performance profiling for Python/PyTorch applications
- **superpowers-methodology**: Complete development methodology with TDD, systematic debugging, and planning

## IsaacLab Specific Instructions

### When working on reinforcement learning components:
- Use the spinning-up-deep-rl skill for algorithm references and debugging strategies
- Apply performance-profiler when optimizing training speed or memory usage
- Follow Karpathy's guidelines for careful, thoughtful implementation
- Use Superpowers methodology for complex feature development

### File Locations
- IsaacLab source: `source/`
- RL training scripts: `scripts/reinforcement_learning/`
- AUV tracking environment: `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/`
- Models and checkpoints: `models/`

### IsaacLab APIs to prioritize:
- Use `omni.isaac.core` for simulation setup
- Use `omni.replicator` for synthetic data generation
- Use `rsl_rl` or `stable_baselines3` for RL algorithms
- Leverage GPU acceleration wherever possible

## Development Best Practices for IsaacLab
- Always consider physics simulation accuracy vs. training speed trade-offs
- Validate robot kinematics and dynamics before complex training
- Use proper seeding for reproducible RL experiments
- Monitor memory usage during long training runs
- Follow IsaacLab's component-based architecture patterns