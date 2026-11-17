"""Label-flipping defaults have no trigger and do not modify backdoor settings."""

from dataclasses import dataclass

from aisec_laya_attack.config import BaseConfig


@dataclass
class LabelFlippingConfig(BaseConfig):
    poison_fraction: float | None = None
    poison_count: int | None = 7
    output_dir: str = 'outputs/label_flipping'


CONFIG = LabelFlippingConfig()
