
# %%
ROOT_DIR="../.."
import sys
sys.path.insert(0, ROOT_DIR)
from marl.config import Config
config = Config.from_file(f'{ROOT_DIR}/projects/ppo/internlm2/8gpu.py')
print(config)

# %%
