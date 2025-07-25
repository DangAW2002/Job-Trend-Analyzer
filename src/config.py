"""
Configuration module for Job Trend Analyzer
Quản lý các cấu hình API keys, model settings, và parameters
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """Configuration for API keys and endpoints"""
    together_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Validate API keys
        if not self.together_api_key:
            print("Warning: TOGETHER_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")

@dataclass
class ModelConfig:
    """Configuration for ML models and embeddings"""
    embedding_model: str = "togethercomputer/m2-bert-80M-32k-retrieval"
    llm_model: str = "gemini-2.5-flash"
    embedding_dimension: int = 768
    max_sequence_length: int = 512

@dataclass
class ProcessingConfig:
    """Configuration for text processing and clustering"""
    ngram_range: tuple = (1, 3)  # Unigrams to trigrams
    top_k_ngrams: int = 100
    n_clusters: int = 10
    random_state: int = 42
    stopwords_language: str = "english"
    min_word_length: int = 2
    
@dataclass
class PathConfig:
    """Configuration for file paths"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "output"
    models_dir: str = "models"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for directory in [self.raw_data_dir, self.processed_data_dir, 
                         self.output_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)

# Global configuration instances
api_config = APIConfig()
model_config = ModelConfig()
processing_config = ProcessingConfig()
path_config = PathConfig()

@dataclass
class Config:
    """Main configuration class that combines all config sections"""
    api: APIConfig = None
    model: ModelConfig = None
    processing: ProcessingConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        if self.api is None:
            self.api = api_config
        if self.model is None:
            self.model = model_config
        if self.processing is None:
            self.processing = processing_config
        if self.paths is None:
            self.paths = path_config

# Validation function
def validate_config() -> bool:
    """Validate that all required configurations are set"""
    if not api_config.together_api_key:
        print("Error: Together API key is required")
        return False
    
    if not api_config.gemini_api_key:
        print("Error: Gemini API key is required")
        return False
    
    print("✅ Configuration validation passed")
    return True

# Environment setup helper
def setup_environment():
    """Setup environment and validate configuration"""
    print("🔧 Setting up Job Trend Analyzer environment...")
    
    # Create .env template if it doesn't exist
    env_file_path = ".env"
    if not os.path.exists(env_file_path):
        env_template = """# Job Trend Analyzer API Keys
TOGETHER_API_KEY=your_together_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional settings
PYTHONPATH=./src
"""
        with open(env_file_path, "w", encoding="utf-8") as f:
            f.write(env_template)
        print(f"📄 Created {env_file_path} template. Please add your API keys.")
    
    # Validate configuration
    is_valid = validate_config()
    
    if is_valid:
        print("🚀 Environment setup completed successfully!")
    else:
        print("⚠️  Environment setup completed with warnings. Check API keys.")
    
    return is_valid

if __name__ == "__main__":
    setup_environment()
