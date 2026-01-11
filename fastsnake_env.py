from verl.envs.environments.FastSnake.src.env import FastSnakeEnv
from verl.envs.environments.FastSnake.base import FastSnakeLLMAgentsWrapper


def make_fastsnake_env(env_name, task, config, render_mode=None):

    fastsnake_kwargs = dict(config.envs.fastsnake_kwargs)

    env = FastSnakeEnv(**fastsnake_kwargs)
    
    # Prepare kwargs for the wrapper, checking prompt config first
    env_kwargs = dict(config.envs)
    
    # Check if prompt config has environment_instruction (takes priority over config.envs.instruction_prompt)
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        environment_instruction = getattr(config.prompt.prompt, 'environment_instruction', None)
        if environment_instruction is not None:
            env_kwargs['instruction_prompt'] = environment_instruction
    
    env = FastSnakeLLMAgentsWrapper(env, **env_kwargs)

    return env
