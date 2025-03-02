# Flappy Bird AI

A reinforcement learning project that trains an AI agent to play Flappy Bird using DQN (Deep Q-Network) with Stable-Baselines3.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install gymnasium pygame stable-baselines3
```

## Project Structure

- `train.py` - Script to train the DQN model
- `test_model.py` - Script to test the trained model
- `play.py` - Play Flappy Bird manually
- `env/FlappyBirdEnv/` - Custom Gymnasium environment for Flappy Bird

## Usage

### Play manually
```bash
python play.py
```

### Train the model
```bash
python train.py
```
This will:
- Train for 3 million timesteps
- Save checkpoints in `./models/`
- Save the best model in `./best_model/`
- Log training metrics to tensorboard

### Test trained model
```bash
python test_model.py
```

## Environment

The environment provides:
- Observation space: [bird_y, velocity_y, distance_to_pipe, upper_pipe_y, lower_pipe_y]
- Action space: [0, 1] (don't flap, flap)
- Reward: +0.1 for staying alive, +1 for passing pipe, -1 for collision