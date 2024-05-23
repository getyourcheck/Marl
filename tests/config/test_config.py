# %%
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].resolve()
from marl.config.config import Config

config = Config.from_file(f"{ROOT_DIR}/projects/ppo/internlm2/1B/actor_2gpu.py")
print(config)

# %%
