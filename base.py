import gymnasium as gym
from verl.envs.environments.FastSnake import ACTIONS


class FastSnakeLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.format_penalty = kwargs.get('format_penalty', 0.1)
        
        self.instruction_prompt = kwargs.get('instruction_prompt', None)
        if self.instruction_prompt is None:
            self.instruction_prompt = self._default_instruction_prompt()

    def _default_instruction_prompt(self):
        action_strings = ",\n".join(f"\"{action}\": {description}" for action, description in ACTIONS.items())
        instruction = f"""[Instructions]
        You are a helpful assistant. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)
You are controlling a snake in a multi-player Snake game

[Available Actions]
{action_strings}

[Rules]
- You can move your head one space up, down, left, or right
- If you move onto an apple, you get 1 point and you gain a body segment
- You die if you move into a wall, another snake, or yourself
"""    
        return instruction

    def get_instruction_prompt(self):
        return self.instruction_prompt
        
    @property
    def max_steps(self):
        return getattr(self.env, 'max_rounds', 100)

    @classmethod
    def language_action_space(cls):
        return list(ACTIONS.keys())
        
    @classmethod
    def default_action(cls):
        return cls.actions()[0]
        
    @classmethod
    def actions(cls):
        return cls.language_action_space()
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def restructure_obs(self, obs):
        return {'text': {'long_term_context': '', 'short_term_context': self.env.game_state_text()},
            'state': obs}

    def step(self, action, is_valid=True):
        # Convert text action to integer for the underlying env
        if isinstance(action, str):
            action_int = self.env.STRING_ACTION_MAP.get(action.lower(), 0)
        else:
            action_int = action
            
        obs, reward, terminated, truncated, info = self.env.step(action_int)

        info['action_was_valid'] = is_valid
        if not is_valid:
            reward = reward-self.format_penalty
        obs = self.restructure_obs(obs)
        return obs, reward*1.0, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.restructure_obs(obs)
        return obs, info

    def get_text_action(self, action):
        return self.language_action_space[action]

    @staticmethod
    def extract_action_from_xml_tag(text: str, tag: str = "action") -> str:
        """Extract action from XML-style tags like <{tag}>UP</{tag}>."""
        try:
            return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip().lower()
        except (IndexError, AttributeError):
            return None

    @classmethod
    def extract_action(cls, action):

        full_action = str(action)
        action = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        if action is None:
            action = "__invalid__"
        is_valid = action in cls.language_action_space()
        extracted_action = action
        valid_action = action if is_valid else cls.default_action()

        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }

        return full_action, extracted_action, valid_action, is_valid, metrics

    def get_stats(self):
        return {}

