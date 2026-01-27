import yaml
from pathlib import Path
from datetime import date

def validate_date(value: str,field_name: str) -> date:
    """Validates that a string is in YYYY-MM-DD format and converts it to a date object."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format, got: {value}")
    
def load_config(path: Path) -> dict:
    """Loads and validates a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(f"{path} config file does not exist")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        if not cfg:
            raise RuntimeError(f"{path} config file is empty or invalid")
        validate_config(cfg)
        return cfg

def validate_config(cfg: dict) -> None:
    """Validates the configuration dictionary structure and types."""

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
    if not isinstance(cfg["dates"]["end"], str):
        raise ValueError("dates.end must be a string")
    start = validate_date(cfg["dates"]["start"], "dates.start")
    end = validate_date(cfg["dates"]["end"], "dates.end")
    if start > end:
        raise ValueError("dates.start must be <= dates.end")
    
    if not isinstance(cfg["universe"]["us"], list):
        raise ValueError("universe.us must be a list")
    for t in cfg["universe"]["us"]:
        if (not isinstance(t, str) or (not t.strip())):
            raise ValueError("universe.us must be a list of strings")   
    if not isinstance(cfg["universe"]["pl"], list):
        raise ValueError("universe.pl must be a list")
    for t in cfg["universe"]["pl"]:
        if (not isinstance(t, str)or (not t.strip())):
            raise ValueError("universe.pl must be a list of strings")
    if not isinstance(cfg["data"]["source"], str):
        raise ValueError("data.source must be a string")
    
    if not cfg["universe"]["us"] and not cfg["universe"]["pl"]:
        raise ValueError("At least one of universe.us or universe.pl must be non-empty")