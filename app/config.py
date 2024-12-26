import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_FILENAME : str
    PREFERENCES_SERVICE_HOST : str
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    )

settings = Settings()
