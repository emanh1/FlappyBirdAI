from gymnasium.envs.registration import register

register(
    id="FlappyBirdEnv/FlappyBird-v0",
    entry_point="FlappyBirdEnv.envs:FlappyBirdEnv",
)
