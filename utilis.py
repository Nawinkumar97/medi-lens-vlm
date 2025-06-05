# config/config.py

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for MediLens application."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./data/medical_knowledge")
    LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
    
    # Model Configuration
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Retrieval Configuration
    MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration."""
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        # Create logs directory if it doesn't exist
        Path(cls.LOGS_PATH).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{cls.LOGS_PATH}/medilens.log"),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        directories = [
            cls.VECTOR_STORE_PATH,
            cls.KNOWLEDGE_BASE_PATH,
            cls.LOGS_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"

class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    # Use different paths for testing
    VECTOR_STORE_PATH = "./test_data/vector_store"
    KNOWLEDGE_BASE_PATH = "./test_data/medical_knowledge"

# Get configuration based on environment
def get_config():
    """Get configuration based on ENVIRONMENT variable."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)

# Initialize configuration
config = get_config()

# Validate and setup
if __name__ == "__main__":
    try:
        config.validate_config()
        config.setup_logging()
        config.create_directories()
        print("Configuration validated and setup complete!")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Setup error: {e}")