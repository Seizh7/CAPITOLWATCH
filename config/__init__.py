# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
from dotenv import load_dotenv
from .settings import Config

# Load .env from the project root (never committed).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

CONFIG = Config(project_root=_PROJECT_ROOT)
