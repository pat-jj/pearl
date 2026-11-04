"""Load API keys from a local environment file."""
import os

def load_api_keys():
    keyfile = os.environ.get("ARTIFACT_APIKEY_FILE", ".apikey")
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"'))
