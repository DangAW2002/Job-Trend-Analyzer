"""
Pipeline Module
Tích hợp toàn bộ luồng xử lý từ văn bản thô đến báo cáo xu hướng
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import local modules (với error handling cho missing dependencies)
try:
    from config import api_config, model_config, processing_config, path_config
    from preprocessing import TextPreprocessor, clean_job_descriptions
    from ngram_extractor import NGramExtractor
    from embedding import EmbeddingGenerator
    from cluster import EmbeddingClusterer, ClusterResult
    from llm_agent import GeminiLLMAgent, AnalysisRequest, AnalysisResult, AnalysisType
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""
    # Text preprocessing
    min_word_length: int = 2
    use_stemming: bool = False
    use_lemmatization: bool = True
    
    # N-gram extraction
    ngram_range: Tuple[int, int] = (1, 3)
    top_k_ngrams: int = 100
    use_tfidf: bool = True
    
    # Clustering
    n_clusters: int = 10
    clustering_algorithm: str = 'kmeans'
    normalize_embeddings: bool = True
    
    # LLM Analysis
    analysis_types: List[str] = None
    llm_temperature: float = 0.3
    max_tokens: int = 1500
    
    # General
    save_intermediate_results: bool = True
    output_format: str = 'json'  # 'json', 'html', 'markdown'
    
    def __post_init__(self):
        if self.analysis_types is None:
            self.analysis_types = ["trend_analysis", "skill_grouping"]

@dataclass
class PipelineResult:
    """Complete result from the pipeline"""
    # Input data
    original_texts: List[str]
    cleaned_texts: List[str]
    
    # N-gram results
    ngrams: List[Tuple[str, float]]
    
    # Embedding results
    embeddings_created: int
    embedding_dimension: int
    
    # Clustering results
    clusters: List[ClusterResult]
    clustering_metrics: Dict[str, float]
    
    # LLM Analysis results
    analyses: Dict[str, AnalysisResult]
    
    # Metadata
    pipeline_config: PipelineConfig
    processing_time: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result_dict = asdict(self)
        
        # Convert AnalysisResult objects to dictionaries
        result_dict['analyses'] = {
            key: analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
            for key, analysis in self.analyses.items()
        }
        
        # Convert ClusterResult objects to dictionaries
        result_dict['clusters'] = [
            cluster.to_dict() if hasattr(cluster, 'to_dict') else cluster
            for cluster in self.clusters
        ]
        
        return result_dict
    
    def save(self, output_path: str) -> str:
        """Save results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.pipeline_config.output_format == 'json':
            file_path = output_path.with_suffix('.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        elif self.pipeline_config.output_format == 'markdown':
            file_path = output_path.with_suffix('.md')
            markdown_content = self._to_markdown()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        
        else:
            raise ValueError(f"Unsupported output format: {self.pipeline_config.output_format}")
        
        return str(file_path)
    
    def _to_markdown(self) -> str:
        """Convert results to markdown format"""
        md = "# Job Trend Analysis Report\n\n"
        md += f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n"
        md += f"**Processing time:** {self.processing_time:.2f} seconds\n\n"
        
        # Summary statistics
        md += "## Summary\n\n"
        md += f"- **Original texts:** {len(self.original_texts)}\n"
        md += f"- **Top N-grams extracted:** {len(self.ngrams)}\n"
        md += f"- **Embeddings created:** {self.embeddings_created}\n"
        md += f"- **Clusters found:** {len(self.clusters)}\n\n"
        
        # Top N-grams
        md += "## Top N-grams\n\n"
        for i, (ngram, score) in enumerate(self.ngrams[:20]):
            md += f"{i+1}. **{ngram}** (score: {score:.2f})\n"
        md += "\n"
        
        # Clusters
        md += "## Skill Clusters\n\n"
        for cluster in self.clusters[:10]:
            md += f"### Cluster {cluster.cluster_id + 1}\n"
            md += f"**Size:** {cluster.size} items | **Avg Score:** {cluster.avg_score:.2f}\n\n"
            md += "**Items:**\n"
            for item in cluster.items[:10]:
                md += f"- {item}\n"
            md += "\n"
        
        # Analysis results
        md += "## AI Analysis\n\n"
        for analysis_type, analysis in self.analyses.items():
            md += f"### {analysis_type.replace('_', ' ').title()}\n\n"
            md += f"**Confidence Score:** {analysis.confidence_score:.2f}\n\n"
            md += f"**Summary:**\n{analysis.summary}\n\n"
            
            if analysis.recommendations:
                md += "**Recommendations:**\n"
                for rec in analysis.recommendations:
                    md += f"- {rec}\n"
                md += "\n"
        
        return md

class JobTrendPipeline:
    """Main pipeline class để xử lý toàn bộ luồng phân tích"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.results = None
        
        # Initialize components
        self._init_components()
        
        logger.info("🚀 Job Trend Pipeline initialized")
    
    def _init_components(self):
        """Initialize pipeline components"""
        try:
            # Text preprocessor
            self.preprocessor = TextPreprocessor(
                min_word_length=self.config.min_word_length,
                use_stemming=self.config.use_stemming,
                use_lemmatization=self.config.use_lemmatization
            )
            
            # N-gram extractor
            self.ngram_extractor = NGramExtractor(
                ngram_range=self.config.ngram_range,
                use_tfidf=self.config.use_tfidf
            )
            
            # Embedding generator
            self.embedding_generator = EmbeddingGenerator()
            
            # Clusterer
            self.clusterer = EmbeddingClusterer(
                algorithm=self.config.clustering_algorithm,
                n_clusters=self.config.n_clusters,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            # LLM Agent
            self.llm_agent = GeminiLLMAgent()
            
            logger.info("✅ All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize some components: {e}")
            logger.warning("Some features may not be available")
    
    def run(self, job_descriptions: List[str], 
            context: Optional[str] = None) -> PipelineResult:
        """
        Run the complete pipeline
        
        Args:
            job_descriptions: List of raw job descriptions
            context: Additional context for LLM analysis
            
        Returns:
            PipelineResult object with all results
        """
        start_time = time.time()
        logger.info(f"🏃 Starting pipeline with {len(job_descriptions)} job descriptions")
        
        try:
            # Step 1: Text preprocessing
            logger.info("📝 Step 1: Text preprocessing...")
            cleaned_texts = self._preprocess_texts(job_descriptions)
            
            # Step 2: N-gram extraction
            logger.info("🔤 Step 2: N-gram extraction...")
            ngrams = self._extract_ngrams(cleaned_texts)
            
            # Step 3: Create embeddings
            logger.info("🧠 Step 3: Creating embeddings...")
            ngram_embeddings = self._create_embeddings(ngrams)
            
            # Step 4: Clustering
            logger.info("📊 Step 4: Clustering...")
            clusters, clustering_metrics = self._cluster_embeddings(ngram_embeddings)
            
            # Step 5: LLM Analysis
            logger.info("🤖 Step 5: LLM Analysis...")
            analyses = self._analyze_with_llm(clusters, context)
            
            # Create final result
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                original_texts=job_descriptions,
                cleaned_texts=cleaned_texts,
                ngrams=ngrams,
                embeddings_created=len(ngram_embeddings),
                embedding_dimension=len(ngram_embeddings[0][2]) if ngram_embeddings else 0,
                clusters=clusters,
                clustering_metrics=clustering_metrics,
                analyses=analyses,
                pipeline_config=self.config,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
            self.results = result
            
            logger.info(f"✅ Pipeline completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess raw texts"""
        return self.preprocessor.preprocess_batch(texts)
    
    def _extract_ngrams(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Extract n-grams from preprocessed texts"""
        return self.ngram_extractor.fit_extract(texts, top_k=self.config.top_k_ngrams)
    
    def _create_embeddings(self, ngrams: List[Tuple[str, float]]) -> List[Tuple[str, float, List[float]]]:
        """Create embeddings for n-grams"""
        return self.embedding_generator.create_ngram_embeddings(ngrams)
    
    def _cluster_embeddings(self, ngram_embeddings: List[Tuple[str, float, List[float]]]) -> Tuple[List[ClusterResult], Dict[str, float]]:
        """Cluster the embeddings"""
        if not ngram_embeddings:
            return [], {}
        
        items = [item[0] for item in ngram_embeddings]
        scores = [item[1] for item in ngram_embeddings]
        embeddings = [item[2] for item in ngram_embeddings]
        
        clusters = self.clusterer.cluster_embeddings(items, scores, embeddings)
        
        # Get clustering metrics
        labels = self.clusterer.fit_predict(embeddings)
        metrics = self.clusterer.evaluate_clustering(embeddings, labels)
        
        return clusters, metrics
    
    def _analyze_with_llm(self, clusters: List[ClusterResult], 
                         context: Optional[str] = None) -> Dict[str, AnalysisResult]:
        """Analyze clusters with LLM"""
        analyses = {}
        
        # Convert clusters to dictionary format
        cluster_dict = {}
        for cluster in clusters:
            cluster_dict[cluster.cluster_id] = cluster.items
        
        # Perform different types of analysis
        for analysis_type in self.config.analysis_types:
            try:
                request = AnalysisRequest(
                    clusters=cluster_dict,
                    analysis_type=AnalysisType(analysis_type),
                    context=context,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.max_tokens
                )
                
                result = self.llm_agent.analyze(request)
                analyses[analysis_type] = result
                
            except Exception as e:
                logger.warning(f"Failed to perform {analysis_type} analysis: {e}")
        
        return analyses
    
    def save_results(self, output_path: str) -> str:
        """Save pipeline results to file"""
        if not self.results:
            raise ValueError("No results to save. Run the pipeline first.")
        
        return self.results.save(output_path)

# Convenience functions
def analyze_job_market(job_descriptions: List[str],
                      output_path: Optional[str] = None,
                      config: Optional[PipelineConfig] = None,
                      context: Optional[str] = None) -> PipelineResult:
    """
    Quick function to analyze job market trends
    
    Args:
        job_descriptions: List of job descriptions
        output_path: Optional path to save results
        config: Pipeline configuration
        context: Additional context for analysis
        
    Returns:
        PipelineResult object
    """
    pipeline = JobTrendPipeline(config)
    result = pipeline.run(job_descriptions, context)
    
    if output_path:
        saved_path = result.save(output_path)
        logger.info(f"📄 Results saved to: {saved_path}")
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test data
    sample_job_descriptions = [
        "Python developer with machine learning experience using TensorFlow and PyTorch",
        "Senior Java backend engineer working with Spring Boot and microservices architecture",
        "Frontend developer specializing in React, Angular, and modern JavaScript frameworks",
        "Data scientist proficient in Python, pandas, scikit-learn, and statistical analysis",
        "DevOps engineer experienced with AWS, Kubernetes, Docker, and CI/CD pipelines",
        "Full-stack developer using Python Django backend and React frontend",
        "Mobile developer creating apps with React Native and Flutter frameworks",
        "AI engineer developing deep learning models with TensorFlow and PyTorch",
        "Cloud architect designing scalable solutions on AWS and Azure platforms",
        "Backend engineer building REST APIs with Node.js and Express framework"
    ]
    
    print("🧪 Testing Complete Pipeline")
    print("=" * 50)
    
    try:
        # Test with minimal configuration
        config = PipelineConfig(
            top_k_ngrams=30,
            n_clusters=5,
            analysis_types=["trend_analysis"],
            save_intermediate_results=False
        )
        
        # Run pipeline
        result = analyze_job_market(
            job_descriptions=sample_job_descriptions,
            config=config,
            context="Sample job market data for testing",
            output_path="test_results"
        )
        
        print(f"✅ Pipeline completed successfully!")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        print(f"   N-grams extracted: {len(result.ngrams)}")
        print(f"   Clusters found: {len(result.clusters)}")
        print(f"   Analyses completed: {len(result.analyses)}")
        
        # Show top n-grams
        print(f"\n📊 Top 10 N-grams:")
        for i, (ngram, score) in enumerate(result.ngrams[:10]):
            print(f"   {i+1:2d}. {ngram:<25} (score: {score:.2f})")
        
        # Show clusters
        print(f"\n🔍 Clusters:")
        for cluster in result.clusters[:3]:
            print(f"   Cluster {cluster.cluster_id}: {', '.join(cluster.items[:5])}...")
        
        # Show analysis summary
        for analysis_type, analysis in result.analyses.items():
            print(f"\n🤖 {analysis_type}:")
            print(f"   Confidence: {analysis.confidence_score:.2f}")
            print(f"   Summary: {analysis.summary[:150]}...")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        print("This is expected if dependencies are not installed or API keys are not set")
