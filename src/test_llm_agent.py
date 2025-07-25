"""
Test script for LLM Agent with updated configuration
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv chưa được cài. Để tự động load .env, hãy chạy: pip install python-dotenv")

from llm_agent import GeminiLLMAgent, AnalysisRequest, AnalysisType

def test_llm_agent():
    """Test LLM Agent with real data"""
    
    # Sample clusters from the actual data
    sample_clusters = {
        0: ["engineer", "developer", "cloud", "development", "application", "design", "microservices", "scalable"],
        1: ["skill", "backend", "kubernetes", "deployment", "learning data"],
        2: ["tensorflow", "testing", "javascripttypescript", "computing", "looking", "deep", "model", "tensorflow pytorch"],
        3: ["preferred", "certification", "rest", "sql"],
        4: ["learning", "must", "senior", "apis", "scikitlearn"],
        5: ["react", "frontend", "frontend react", "build"],
        6: ["network"],
        7: ["rest apis"],
        8: ["python", "python scripting", "docker", "panda", "year"],
        9: ["data", "required", "scripting", "deep learning", "knowledge", "pytorch"]
    }
    
    print("🧪 Testing LLM Agent with updated configuration")
    print("=" * 60)
    
    try:
        # Initialize agent
        print("🤖 Initializing GeminiLLMAgent...")
        agent = GeminiLLMAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            clusters=sample_clusters,
            analysis_type=AnalysisType.SKILL_GROUPING,
            context="Phân tích loại: trend_analysis",
            max_tokens=2000,
            temperature=0.3
        )
        
        # Perform analysis
        print("📊 Performing skill grouping analysis...")
        result = agent.analyze(request)
        
        print("\n✅ Analysis completed successfully!")
        print(f"   📋 Type: {result.analysis_type}")
        print(f"   🎯 Confidence: {result.confidence_score:.2f}")
        print(f"   📝 Summary: {result.summary[:200]}...")
        print(f"   📈 Trends found: {len(result.trends)}")
        print(f"   💡 Recommendations: {len(result.recommendations)}")
        
        # Show detailed analysis keys
        print(f"   🔍 Analysis details: {list(result.detailed_analysis.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during LLM Agent test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_agent()
    if success:
        print("\n🎉 LLM Agent test completed successfully!")
    else:
        print("\n💥 LLM Agent test failed!")
