"""
N-gram Extraction Module
Trích xuất các n-gram từ văn bản đã được xử lý
"""

import re
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

class NGramExtractor:
    """Class để trích xuất và phân tích n-gram từ văn bản"""
    
    def __init__(self,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_features: Optional[int] = None,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 use_tfidf: bool = False,
                 stop_words: Optional[List[str]] = None):
        """
        Initialize N-gram extractor
        
        Args:
            ngram_range: Range of n-gram sizes (min_n, max_n)
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            use_tfidf: Whether to use TF-IDF instead of count
            stop_words: List of stop words to exclude
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_tfidf = use_tfidf
        self.stop_words = stop_words or []
        
        # Initialize vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
            )
        
        self.is_fitted = False
        self.feature_names = None
        self.feature_scores = None
    
    def fit(self, texts: List[str]) -> 'NGramExtractor':
        """
        Fit the n-gram extractor on texts
        
        Args:
            texts: List of texts to fit on
            
        Returns:
            Self for method chaining
        """
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        
        if not texts:
            raise ValueError("No valid texts provided for fitting")
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        
        return self
    
    def extract_ngrams(self, texts: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract n-grams from texts
        
        Args:
            texts: List of texts to extract n-grams from
            top_k: Number of top n-grams to return
            
        Returns:
            List of (ngram, score) tuples sorted by score
        """
        if not self.is_fitted:
            raise ValueError("NGramExtractor must be fitted before extracting n-grams")
        
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        
        if not texts:
            return []
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Calculate scores (sum across documents)
        scores = X.sum(axis=0).A1
        
        # Create list of (ngram, score) tuples
        ngram_scores = list(zip(self.feature_names, scores))
        
        # Sort by score (descending)
        ngram_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k:
            ngram_scores = ngram_scores[:top_k]
        
        return ngram_scores
    
    def fit_extract(self, texts: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Fit and extract n-grams in one step
        
        Args:
            texts: List of texts to process
            top_k: Number of top n-grams to return
            
        Returns:
            List of (ngram, score) tuples
        """
        return self.fit(texts).extract_ngrams(texts, top_k)
    
    def get_ngrams_by_length(self, texts: List[str], top_k_per_length: int = 20) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get n-grams grouped by their length
        
        Args:
            texts: List of texts to process
            top_k_per_length: Number of top n-grams per length
            
        Returns:
            Dictionary mapping n-gram length to list of (ngram, score) tuples
        """
        if not self.is_fitted:
            self.fit(texts)
        
        all_ngrams = self.extract_ngrams(texts)
        
        # Group by n-gram length
        ngrams_by_length = defaultdict(list)
        for ngram, score in all_ngrams:
            length = len(ngram.split())
            ngrams_by_length[length].append((ngram, score))
        
        # Keep top_k for each length
        result = {}
        for length, ngrams in ngrams_by_length.items():
            result[length] = ngrams[:top_k_per_length]
        
        return result
    
    def analyze_ngram_trends(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze n-gram trends and create a summary DataFrame
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with n-gram analysis
        """
        if not self.is_fitted:
            self.fit(texts)
        
        # Get n-grams by length
        ngrams_by_length = self.get_ngrams_by_length(texts, top_k_per_length=50)
        
        # Create analysis DataFrame
        analysis_data = []
        for length, ngrams in ngrams_by_length.items():
            for ngram, score in ngrams:
                analysis_data.append({
                    'ngram': ngram,
                    'length': length,
                    'score': score,
                    'normalized_score': score / len(texts),
                    'word_count': len(ngram.split())
                })
        
        df = pd.DataFrame(analysis_data)
        
        # Add rank within each length category
        df['rank_in_length'] = df.groupby('length')['score'].rank(method='dense', ascending=False)
        
        return df.sort_values(['length', 'rank_in_length'])
    
    def get_skill_phrases(self, texts: List[str], top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Extract skill-related phrases using pattern matching
        
        Args:
            texts: List of texts to analyze
            top_k: Number of top skills to return
            
        Returns:
            List of (skill_phrase, score) tuples
        """
        if not self.is_fitted:
            self.fit(texts)
        
        all_ngrams = self.extract_ngrams(texts)
        
        # Skill-related patterns
        skill_patterns = [
            r'\b\w*(?:python|java|javascript|sql|html|css|react|angular|vue)\w*\b',
            r'\b\w*(?:machine learning|deep learning|ai|data science)\w*\b',
            r'\b\w*(?:cloud|aws|azure|gcp|kubernetes|docker)\w*\b',
            r'\b\w*(?:backend|frontend|fullstack|devops)\w*\b',
            r'\b\w*(?:api|rest|graphql|microservices)\w*\b',
            r'\b\w*(?:database|mysql|postgresql|mongodb|redis)\w*\b',
        ]
        
        skill_ngrams = []
        for ngram, score in all_ngrams:
            for pattern in skill_patterns:
                if re.search(pattern, ngram, re.IGNORECASE):
                    skill_ngrams.append((ngram, score))
                    break
        
        # Remove duplicates and sort
        skill_ngrams = list(set(skill_ngrams))
        skill_ngrams.sort(key=lambda x: x[1], reverse=True)
        
        return skill_ngrams[:top_k]

# Convenience functions
def extract_ngrams(texts: List[str], 
                  ngram_range: Tuple[int, int] = (1, 3),
                  top_k: int = 50,
                  use_tfidf: bool = False) -> List[Tuple[str, float]]:
    """
    Quick n-gram extraction function
    
    Args:
        texts: List of texts to process
        ngram_range: Range of n-gram sizes
        top_k: Number of top n-grams to return
        use_tfidf: Whether to use TF-IDF scoring
        
    Returns:
        List of (ngram, score) tuples
    """
    extractor = NGramExtractor(ngram_range=ngram_range, use_tfidf=use_tfidf)
    return extractor.fit_extract(texts, top_k)

def get_job_skills(texts: List[str], top_k: int = 30) -> List[Tuple[str, float]]:
    """
    Extract job skills from texts
    
    Args:
        texts: List of job descriptions
        top_k: Number of top skills to return
        
    Returns:
        List of (skill, score) tuples
    """
    extractor = NGramExtractor(ngram_range=(1, 3), use_tfidf=True)
    extractor.fit(texts)
    return extractor.get_skill_phrases(texts, top_k)

# Example usage and testing
if __name__ == "__main__":
    # Test data
    sample_job_descriptions = [
        "python developer machine learning tensorflow pytorch",
        "java backend spring boot microservices aws",
        "javascript react frontend angular html css",
        "data scientist python pandas numpy machine learning",
        "devops engineer kubernetes docker aws cloud",
        "fullstack developer python javascript react postgresql",
        "backend engineer java spring boot rest api",
        "ai engineer deep learning tensorflow python",
        "cloud architect aws azure kubernetes microservices",
        "data engineer python sql spark hadoop"
    ]
    
    print("🧪 Testing N-gram Extractor")
    print("=" * 50)
    
    # Test basic n-gram extraction
    extractor = NGramExtractor(ngram_range=(1, 3), use_tfidf=True)
    ngrams = extractor.fit_extract(sample_job_descriptions, top_k=20)
    
    print("Top N-grams:")
    for i, (ngram, score) in enumerate(ngrams[:10]):
        print(f"{i+1:2d}. {ngram:<25} (score: {score:.3f})")
    
    print("\n" + "=" * 50)
    
    # Test skill extraction
    skills = get_job_skills(sample_job_descriptions, top_k=15)
    print("Top Skills:")
    for i, (skill, score) in enumerate(skills):
        print(f"{i+1:2d}. {skill:<25} (score: {score:.3f})")
    
    print("\n" + "=" * 50)
    
    # Test n-grams by length
    ngrams_by_length = extractor.get_ngrams_by_length(sample_job_descriptions, top_k_per_length=5)
    for length, ngrams in ngrams_by_length.items():
        print(f"\n{length}-grams:")
        for ngram, score in ngrams:
            print(f"  - {ngram:<20} (score: {score:.3f})")
