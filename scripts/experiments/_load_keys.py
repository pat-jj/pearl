"""Load API keys from ${PROJECT_PARENT}/.apikey into environment."""
import os

def load_api_keys():
    keyfile = "${PROJECT_PARENT}/.apikey"
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"'))
