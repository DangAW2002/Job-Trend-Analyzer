"""
Main entry point for Job Trend Analyzer
Cung cấp CLI interface và demo functions
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_job_descriptions_from_file(file_path: str) -> List[str]:
    """Load job descriptions from various file formats"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict):
            # Try common keys for job descriptions
            for key in ['job_descriptions', 'descriptions', 'jobs', 'data']:
                if key in data and isinstance(data[key], list):
                    return [str(item) for item in data[key]]
            # If no list found, return all string values
            return [str(v) for v in data.values() if isinstance(v, str)]
        else:
            return [str(data)]
    
    elif file_path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (assuming each job description is separated)
        descriptions = [desc.strip() for desc in content.split('\n\n') if desc.strip()]
        return descriptions
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def create_sample_data():
    """Create sample job descriptions for testing"""
    sample_jobs = [
        """
        Senior Python Developer
        We are looking for an experienced Python developer with expertise in machine learning 
        and data science. Must have knowledge of TensorFlow, PyTorch, scikit-learn, and pandas. 
        Experience with cloud platforms (AWS, GCP) is a plus.
        Requirements: 5+ years Python, ML algorithms, REST APIs, SQL databases.
        """,
        
        """
        Java Backend Engineer
        Join our team as a Java backend engineer to build scalable microservices. 
        Required skills: Spring Boot, REST APIs, MySQL, Redis, Kafka. 
        Experience with Docker and Kubernetes preferred.
        Location: Remote-friendly. Salary: $120k-$160k.
        """,
        
        """
        Frontend React Developer
        We need a skilled frontend developer proficient in React, TypeScript, and modern CSS.
        Experience with Redux, Next.js, and testing frameworks is required.
        Must have: 3+ years React, responsive design, Git, Agile methodologies.
        """,
        
        """
        Data Scientist
        Seeking a data scientist to analyze large datasets and build predictive models.
        Required: Python, pandas, NumPy, scikit-learn, SQL, statistical analysis.
        Preferred: TensorFlow, deep learning, data visualization (Matplotlib, Plotly).
        """,
        
        """
        DevOps Engineer
        Looking for a DevOps engineer to manage our cloud infrastructure and CI/CD pipelines.
        Skills needed: AWS/Azure, Kubernetes, Docker, Terraform, Jenkins, monitoring tools.
        Experience with Python scripting and Linux administration required.
        """,
        
        """
        Full Stack Developer
        Full stack position requiring both frontend and backend development skills.
        Frontend: React, Vue.js, HTML/CSS, JavaScript/TypeScript.
        Backend: Node.js, Express, Python Django/Flask, PostgreSQL.
        """,
        
        """
        AI/ML Engineer
        AI engineer role focusing on deep learning and computer vision applications.
        Must have: TensorFlow, PyTorch, OpenCV, Python, neural networks.
        Experience with GPU computing (CUDA) and model deployment preferred.
        """,
        
        """
        Cloud Architect
        Senior cloud architect position for designing scalable cloud solutions.
        Required: AWS/Azure/GCP certification, microservices, serverless computing.
        Leadership experience and enterprise architecture knowledge essential.
        """,
        
        """
        Mobile Developer
        React Native developer for cross-platform mobile applications.
        Skills: React Native, JavaScript/TypeScript, native iOS/Android development.
        Experience with app store deployment and mobile UI/UX design.
        """,
        
        """
        Cybersecurity Analyst
        Information security analyst to protect our systems and data.
        Required: network security, penetration testing, vulnerability assessment.
        Certifications preferred: CISSP, CEH, Security+. Python scripting skills.
        """
    ]
    
    # Save to file
    sample_file = Path("data/raw/sample_jobs.json")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_jobs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📄 Sample data created: {sample_file}")
    return str(sample_file)

def run_analysis(args):
    """Run the job trend analysis"""
    try:
        # Import pipeline components
        from src.pipeline import JobTrendPipeline, PipelineConfig, analyze_job_market
        from src.config import setup_environment, validate_config
        
        # Setup environment
        logger.info("🔧 Setting up environment...")
        if not setup_environment():
            logger.error("❌ Environment setup failed. Please check your API keys.")
            return False
        
        # Load job descriptions
        if args.input_file:
            logger.info(f"📂 Loading job descriptions from: {args.input_file}")
            job_descriptions = load_job_descriptions_from_file(args.input_file)
        elif args.sample:
            logger.info("🎯 Using sample data...")
            sample_file = create_sample_data()
            job_descriptions = load_job_descriptions_from_file(sample_file)
        else:
            logger.error("❌ No input specified. Use --input-file or --sample")
            return False
        
        logger.info(f"📊 Loaded {len(job_descriptions)} job descriptions")
        
        # Configure pipeline
        config = PipelineConfig(
            top_k_ngrams=args.ngrams,
            n_clusters=args.clusters,
            analysis_types=args.analysis_types.split(',') if args.analysis_types else ["trend_analysis"],
            save_intermediate_results=not args.no_intermediate,
            output_format=args.format
        )
        
        # Run analysis
        logger.info("🚀 Starting job trend analysis...")
        
        result = analyze_job_market(
            job_descriptions=job_descriptions,
            output_path=args.output,
            config=config,
            context=args.context
        )
        
        # Print summary
        print("\n" + "="*60)
        print("📈 JOB TREND ANALYSIS RESULTS")
        print("="*60)
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Original texts: {len(result.original_texts)}")
        print(f"Top N-grams: {len(result.ngrams)}")
        print(f"Clusters found: {len(result.clusters)}")
        print(f"AI analyses: {len(result.analyses)}")
        
        # Show top insights
        print(f"\n🔝 TOP 10 TRENDING SKILLS:")
        for i, (ngram, score) in enumerate(result.ngrams[:10]):
            print(f"  {i+1:2d}. {ngram:<30} (score: {score:.2f})")
        
        print(f"\n📊 SKILL CLUSTERS:")
        for i, cluster in enumerate(result.clusters[:5]):
            print(f"  Cluster {i+1}: {', '.join(cluster.items[:5])}...")
        
        # Show AI analysis summary
        for analysis_type, analysis in result.analyses.items():
            print(f"\n🤖 {analysis_type.upper()} ANALYSIS:")
            print(f"  Confidence: {analysis.confidence_score:.2f}")
            print(f"  Summary: {analysis.summary[:200]}...")
        
        if args.output:
            print(f"\n💾 Full results saved to: {args.output}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

def run_demo():
    """Run a simple demo of the pipeline"""
    logger.info("🎬 Running Job Trend Analyzer Demo...")
    
    try:
        # Create sample data
        sample_file = create_sample_data()
        
        # Run analysis with sample data
        from src.pipeline import analyze_job_market, PipelineConfig
        from src.config import setup_environment
        
        # Quick environment check
        setup_environment()
        
        # Load sample data
        job_descriptions = load_job_descriptions_from_file(sample_file)
        
        # Configure for demo (smaller, faster)
        config = PipelineConfig(
            top_k_ngrams=20,
            n_clusters=5,
            analysis_types=["trend_analysis"],
            save_intermediate_results=False
        )
        
        # Run analysis
        result = analyze_job_market(
            job_descriptions=job_descriptions,
            config=config,
            context="Demo analysis of sample job market data"
        )
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"Found {len(result.ngrams)} trending skills in {len(result.clusters)} clusters")
        print("Top 5 skills:", [ngram for ngram, _ in result.ngrams[:5]])
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is normal if API keys are not configured or dependencies are missing")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Job Trend Analyzer - Analyze job market trends using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sample --output results/analysis
  python main.py --input data/jobs.json --clusters 8 --ngrams 50
  python main.py --demo
  python main.py --input jobs.txt --format markdown --context "Q4 2024 data"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-file', '-i',
        help='Path to input file containing job descriptions (JSON or TXT)'
    )
    input_group.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Use built-in sample data for testing'
    )
    input_group.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run a quick demo of the pipeline'
    )
    
    # Analysis options
    parser.add_argument(
        '--output', '-o',
        default='output/job_trends',
        help='Output file path (without extension, default: output/job_trends)'
    )
    parser.add_argument(
        '--clusters', '-c',
        type=int,
        default=10,
        help='Number of clusters for skill grouping (default: 10)'
    )
    parser.add_argument(
        '--ngrams', '-n',
        type=int,
        default=100,
        help='Number of top n-grams to extract (default: 100)'
    )
    parser.add_argument(
        '--analysis-types', '-a',
        default='trend_analysis,skill_grouping',
        help='Comma-separated list of analysis types (default: trend_analysis,skill_grouping)'
    )
    parser.add_argument(
        '--context',
        help='Additional context for AI analysis'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'markdown'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Skip saving intermediate results'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate function
    if args.demo:
        success = run_demo()
    else:
        success = run_analysis(args)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
