import yaml
from pathlib import Path

def load_config(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"{path} config file does not exist")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        if not cfg:
            raise RuntimeError(f"{path} config file is empty or invalid")
        return cfg

def validate_config(cfg: dict) -> None:
    required_sections = ["dates", "data", "universe"]
    required_subsections = {
        "dates": ["start", "end"],
        "data": ["source"],
        "universe": ["us","pl"]
    }
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")
        
        for subsection in required_subsections[section]:
            if subsection not in cfg[section]:
                raise ValueError(f"Missing required config subsection: {section}.{subsection}")
            
    # Additional type checks
    if not isinstance(cfg["dates"]["start"], str):
        raise ValueError("dates.start must be a string")
    if not isinstance(cfg["universe"]["us"], list):
        raise ValueError("universe.us must be a list")
    if not isinstance(cfg["universe"]["pl"], list):
        raise ValueError("universe.pl must be a list")
    if not isinstance(cfg["data"]["source"], str):
        raise ValueError("data.source must be a string")
    
    if not cfg["universe"]["us"] and not cfg["universe"]["pl"]:
        raise ValueError("At least one of universe.us or universe.pl must be non-empty")