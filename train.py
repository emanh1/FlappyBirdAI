import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import FlappyBirdEnv

env = gym.make('FlappyBirdEnv/FlappyBird-v0')
env = Monitor(env)

# Create separate env for evaluation
eval_env = gym.make('FlappyBirdEnv/FlappyBird-v0')
eval_env = Monitor(eval_env)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="tensorboard"
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="flappy_dqn"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

callbacks = [checkpoint_callback, eval_callback]

TIMESTEPS = 3000000
model.learn(
    total_timesteps=TIMESTEPS,
    callback=callbacks,
    progress_bar=True
)

model.save("flappy_bird_dqn_final2")

env.close()
