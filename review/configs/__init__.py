from pathlib import Path
import yaml

conf_file = Path(__file__).parent / "config.yaml"
with open(conf_file, encoding="utf-8") as f:
    CONFMAP = yaml.load(f.read(), Loader=yaml.FullLoader)
