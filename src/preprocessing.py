"""
Text Preprocessing Module
Xử lý và làm sạch văn bản từ job descriptions
"""

import re
import string
import nltk
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Download required NLTK data (safe, no punkt_tab)
def ensure_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

class TextPreprocessor:
    """Class để xử lý và làm sạch văn bản"""
    
    def __init__(self, 
                 language: str = 'english',
                 min_word_length: int = 2,
                 use_stemming: bool = False,
                 use_lemmatization: bool = True,
                 remove_numbers: bool = True,
                 remove_punctuation: bool = True,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize text preprocessor
        
        Args:
            language: Language for stopwords (default: 'english')
            min_word_length: Minimum word length to keep
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_numbers: Whether to remove numbers
            remove_punctuation: Whether to remove punctuation
            custom_stopwords: Additional stopwords to remove
        """
        self.language = language
        self.min_word_length = min_word_length
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        
        # Initialize NLTK tools
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Setup stopwords
        self.stopwords = set(stopwords.words(language))
        
        # Add job-specific stopwords
        job_stopwords = {
            'job', 'position', 'role', 'candidate', 'applicant', 'experience',
            'work', 'company', 'team', 'office', 'location', 'salary',
            'benefit', 'requirement', 'qualification', 'responsibility',
            'opportunity', 'career', 'employment', 'hire', 'hiring'
        }
        self.stopwords.update(job_stopwords)
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and apply filtering. Fallback to split() if punkt is missing.
        """
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback if punkt is missing
            tokens = text.split()
        except Exception:
            tokens = text.split()

        filtered_tokens = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if token in self.stopwords:
                continue
            if not token.isalpha():
                continue
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            elif self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            filtered_tokens.append(token)
        return filtered_tokens
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

# Convenience functions
def clean_text(text: str, **kwargs) -> str:
    """
    Quick text cleaning function
    
    Args:
        text: Text to clean
        **kwargs: Additional parameters for TextPreprocessor
        
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)

def clean_job_descriptions(job_descriptions: List[str], **kwargs) -> List[str]:
    """
    Clean a list of job descriptions
    
    Args:
        job_descriptions: List of job descriptions
        **kwargs: Additional parameters for TextPreprocessor
        
    Returns:
        List of cleaned job descriptions
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess_batch(job_descriptions)

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    sample_job_desc = """
    We are looking for a Senior Python Developer with 5+ years of experience 
    in machine learning and data science. The candidate should have expertise 
    in TensorFlow, PyTorch, and scikit-learn. 
    
    Requirements:
    - Bachelor's degree in Computer Science
    - Experience with AWS/GCP cloud platforms
    - Strong knowledge of SQL and NoSQL databases
    - Excellent communication skills
    
    Salary: $120,000 - $150,000 per year
    Location: San Francisco, CA
    Email: jobs@company.com
    Phone: (555) 123-4567
    """
    
    print("🧪 Testing Text Preprocessor")
    print("=" * 50)
    print("Original text:")
    print(sample_job_desc)
    print("\n" + "=" * 50)
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess(sample_job_desc)
    
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "=" * 50)
    
    # Test batch processing
    sample_texts = [
        "Python developer needed with Django experience",
        "Java backend engineer for microservices architecture",
        "Data scientist with machine learning expertise"
    ]
    
    cleaned_batch = clean_job_descriptions(sample_texts)
    print("Batch processing results:")
    for i, text in enumerate(cleaned_batch):
        print(f"{i+1}. {text}")
