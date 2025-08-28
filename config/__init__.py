# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

# config/__init__.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Application environment (default: "development")
APP_ENV = os.getenv("APP_ENV", "development").lower()

# Load the corresponding .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILES = {
    "development": PROJECT_ROOT / ".env.dev",
    "production": PROJECT_ROOT / ".env.prod",
}

# Pick the right file
dotenv_path = ENV_FILES.get(APP_ENV)
load_dotenv(dotenv_path, override=True)
loaded_env = str(dotenv_path)

# Import the correct configuration class
if APP_ENV == "production":
    from .production import Config
else:
    from .development import Config

CONFIG = Config()
