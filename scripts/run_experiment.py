import os
from pathlib import Path
import sys

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR / 'src'))
os.environ.setdefault('HF_HOME', str(LAB_DIR / '.cache' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from aisec_laya_attack.experiment import main


if __name__ == '__main__':
    main()
