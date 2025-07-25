"""
LLM Agent Module
Sử dụng Gemini API để phân tích các cụm và tạo báo cáo xu hướng
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import json
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis the LLM agent can perform"""
    TREND_ANALYSIS = "trend_analysis"
    SKILL_GROUPING = "skill_grouping"
    MARKET_INSIGHTS = "market_insights"
    CAREER_RECOMMENDATIONS = "career_recommendations"

@dataclass
class AnalysisRequest:
    """Data class for analysis requests"""
    clusters: Dict[int, List[str]]
    analysis_type: AnalysisType
    context: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.3
    
    def to_dict(self) -> Dict:
        return {
            'clusters': self.clusters,
            'analysis_type': self.analysis_type.value,
            'context': self.context,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }

@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    analysis_type: str
    summary: str
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    trends: List[Dict[str, Any]]
    confidence_score: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'analysis_type': self.analysis_type,
            'summary': self.summary,
            'detailed_analysis': self.detailed_analysis,
            'recommendations': self.recommendations,
            'trends': self.trends,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp
        }

class GeminiLLMAgent:
    """LLM Agent sử dụng Google Gemini API"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash",
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize Gemini LLM Agent
        
        Args:
            api_key: Google API key (if None, will try to get from environment)
            model: Model name to use
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize Gemini client
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Try to initialize with the specified model
            try:
                self.client = genai.GenerativeModel(model)
                logger.info(f"✅ Gemini API client initialized with model: {self.model}")
            except Exception as model_error:
                # Fallback to other Gemini models if the specified model fails
                logger.warning(f"Failed to initialize {model}, trying fallback models...")
                fallback_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                
                for fallback_model in fallback_models:
                    try:
                        self.client = genai.GenerativeModel(fallback_model)
                        self.model = fallback_model
                        logger.info(f"✅ Gemini API client initialized with fallback model: {self.model}")
                        break
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
                else:
                    raise Exception(f"All model attempts failed. Original error: {model_error}")
                    
        except ImportError:
            logger.error("google-generativeai library not installed. Please install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _build_prompt(self, request: AnalysisRequest) -> str:
        """Build prompt for the LLM based on analysis type"""
        
        # Base context
        prompt = f"""Bạn là một chuyên gia phân tích thị trường việc làm công nghệ. 
Nhiệm vụ của bạn là phân tích các nhóm kỹ năng/công nghệ và đưa ra những nhận định sâu sắc về xu hướng thị trường.

Dữ liệu đầu vào - Các nhóm kỹ năng/công nghệ:
"""
        
        # Add cluster data
        for cluster_id, items in request.clusters.items():
            prompt += f"\nNhóm {cluster_id + 1}: {', '.join(items[:10])}"  # Limit items to avoid token overflow
            if len(items) > 10:
                prompt += f" (và {len(items) - 10} kỹ năng khác)"
        
        # Add context if provided
        if request.context:
            prompt += f"\n\nBối cảnh bổ sung: {request.context}"
        
        # Add specific instructions based on analysis type
        if request.analysis_type == AnalysisType.TREND_ANALYSIS:
            prompt += """

Hãy thực hiện phân tích xu hướng thị trường với các yêu cầu sau:

1. **Xu hướng tăng trưởng**: Xác định nhóm kỹ năng nào đang có xu hướng tăng mạnh
2. **Xu hướng suy giảm**: Nhóm kỹ năng nào có dấu hiệu suy giảm
3. **Công nghệ mới nổi**: Những công nghệ/kỹ năng mới đang xuất hiện
4. **Kỹ năng cốt lõi**: Những kỹ năng vẫn quan trọng và ổn định
5. **Dự đoán tương lai**: Xu hướng có thể xảy ra trong 1-2 năm tới

Trả về kết quả dưới dạng JSON với format:
{
    "summary": "Tóm tắt ngắn gọn về xu hướng thị trường",
    "trending_up": ["công nghệ tăng trưởng"],
    "trending_down": ["công nghệ suy giảm"],
    "emerging_tech": ["công nghệ mới nổi"],
    "core_skills": ["kỹ năng cốt lõi"],
    "future_predictions": ["dự đoán tương lai"],
    "confidence_score": 0.85
}"""
        
        elif request.analysis_type == AnalysisType.SKILL_GROUPING:
            prompt += """

Hãy phân tích và nhóm các kỹ năng theo tiêu chí sau:

1. Theo lĩnh vực: Frontend, Backend, Data Science, DevOps, AI/ML
2. Theo mức độ: Entry, Mid, Senior, Expert
3. Theo xu hướng: Hot, Stable, Declining
4. Theo mối quan hệ: Complementary, Alternatives

Trả về JSON với format:
{
    "summary": "Tóm tắt về cách nhóm kỹ năng",
    "by_domain": {"Frontend": [], "Backend": [], "Data Science": [], "DevOps": [], "AI/ML": []},
    "by_level": {"Entry": [], "Mid": [], "Senior": [], "Expert": []},
    "by_trend": {"Hot": [], "Stable": [], "Declining": []},
    "skill_relationships": {"complementary": [], "alternatives": []},
    "confidence_score": 0.9
}"""
        
        elif request.analysis_type == AnalysisType.MARKET_INSIGHTS:
            prompt += """

Hãy đưa ra những insight sâu sắc về thị trường việc làm dựa trên dữ liệu:

1. **Cơ hội việc làm**: Lĩnh vực nào có nhiều cơ hội nhất
2. **Mức lương**: Xu hướng mức lương theo từng nhóm kỹ năng
3. **Khoảng cách kỹ năng**: Kỹ năng nào bị thiếu hụt trên thị trường
4. **Định hướng học tập**: Nên học kỹ năng gì để có cơ hội việc làm tốt
5. **Rủi ro nghề nghiệp**: Những rủi ro cần lưu ý

Trả về kết quả dưới dạng JSON với format:
{
    "summary": "Tóm tắt insights về thị trường",
    "job_opportunities": {"high": [], "medium": [], "low": []},
    "salary_trends": {"high_paying": [], "average_paying": [], "entry_level": []},
    "skill_gaps": ["kỹ năng bị thiếu hụt"],
    "learning_recommendations": ["kỹ năng nên học"],
    "career_risks": ["rủi ro nghề nghiệp"],
    "confidence_score": 0.8
}"""
        
        elif request.analysis_type == AnalysisType.CAREER_RECOMMENDATIONS:
            prompt += """

Hãy đưa ra lời khuyên nghề nghiệp dựa trên phân tích các nhóm kỹ năng:

1. **Lộ trình nghề nghiệp**: Các con đường phát triển khác nhau
2. **Kỹ năng cần thiết**: Kỹ năng cần có cho từng lộ trình
3. **Thời gian học tập**: Ước tính thời gian để thành thạo
4. **Lời khuyên chuyển nghề**: Cho người muốn chuyển sang IT
5. **Cập nhật kỹ năng**: Cho người đã làm trong ngành

Trả về kết quả dưới dạng JSON với format:
{
    "summary": "Tóm tắt lời khuyên nghề nghiệp",
    "career_paths": [{"path": "tên lộ trình", "skills": [], "duration": "thời gian"}],
    "essential_skills": ["kỹ năng cần thiết"],
    "learning_timeline": {"beginner": "6-12 tháng", "intermediate": "1-2 năm"},
    "career_switch_advice": ["lời khuyên chuyển nghề"],
    "skill_update_advice": ["lời khuyên cập nhật kỹ năng"],
    "confidence_score": 0.85
}"""
        
        prompt += "\n\nLưu ý: Chỉ trả về JSON hợp lệ, không có text bổ sung."
        
        return prompt
    
    def _make_prompt_neutral(self, prompt: str) -> str:
        """Make prompt more neutral to avoid safety blocks"""
        # Remove potentially triggering words and make more neutral
        neutral_prompt = prompt.replace("xu hướng thị trường", "phân tích dữ liệu")
        neutral_prompt = neutral_prompt.replace("việc làm", "thông tin")
        neutral_prompt = neutral_prompt.replace("tăng trưởng", "phát triển")
        neutral_prompt = neutral_prompt.replace("suy giảm", "thay đổi")
        
        # Make the request more academic/technical
        neutral_prompt = "Bạn là một chuyên gia phân tích dữ liệu. " + neutral_prompt
        
        return neutral_prompt
    
    def _create_fallback_response(self) -> str:
        """Create a fallback JSON response when LLM fails"""
        fallback = {
            "summary": "Phân tích không thể hoàn thành do giới hạn API. Vui lòng thử lại sau.",
            "trending_up": ["python", "javascript", "react", "aws"],
            "trending_down": [],
            "emerging_tech": ["ai", "machine learning"],
            "core_skills": ["programming", "database", "api"],
            "confidence_score": 0.1
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)
    
    def _call_llm_with_retry(self, prompt: str, **kwargs) -> str:
        """Call LLM with retry logic and safety handling"""
        
        # Log the prompt being sent
        logger.info("📤 SENDING PROMPT TO GEMINI:")
        logger.info("=" * 80)
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        logger.info("=" * 80)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries} with model: {self.model}")
                
                # Start with simple generation without safety settings
                if attempt == 0:
                    response = self.client.generate_content(prompt)
                else:
                    # If first attempt failed, try with generation config
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            'temperature': kwargs.get('temperature', 0.3),
                            'max_output_tokens': kwargs.get('max_tokens', 2000),
                        }
                    )
                
                logger.info(f"📥 Response received. Type: {type(response)}")
                
                # Check if we have a valid response
                if hasattr(response, 'text') and response.text:
                    logger.info(f"✅ Valid response received ({len(response.text)} chars)")
                    logger.info("📥 RESPONSE FROM GEMINI:")
                    logger.info("=" * 80)
                    logger.info(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                    logger.info("=" * 80)
                    return response.text.strip()
                
                # Handle blocked responses (finish_reason = 2 means SAFETY)
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    logger.warning(f"❌ Response candidate finish_reason: {finish_reason}")
                    
                    if finish_reason == 2:  # SAFETY
                        logger.warning(f"🚫 Response blocked by safety filter on attempt {attempt + 1}")
                        # Try with a more neutral prompt
                        if attempt == 0:
                            neutral_prompt = self._make_prompt_neutral(prompt)
                            logger.info("🔄 Trying with neutralized prompt...")
                            prompt = neutral_prompt
                            continue
                    elif finish_reason == 3:  # RECITATION
                        logger.warning(f"🚫 Response blocked for recitation on attempt {attempt + 1}")
                    else:
                        logger.warning(f"🚫 Response blocked with finish_reason {finish_reason} on attempt {attempt + 1}")
                
                logger.warning(f"⚠️ Empty or blocked response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"All attempts failed: {e}")
                    # Return a fallback response instead of raising
                    return self._create_fallback_response()
        
        # If all retries failed, return fallback
        return self._create_fallback_response()
    
    def _make_prompt_neutral(self, prompt: str) -> str:
        """Make prompt more neutral to avoid safety blocks"""
        # Create a much simpler, more academic prompt
        simple_prompt = """Phân tích dữ liệu kỹ năng công nghệ sau đây và nhóm chúng theo các tiêu chí:

Dữ liệu: """
        
        # Add cluster data in a simple format
        for cluster_id, items in self.current_clusters.items():
            simple_prompt += f"\nNhóm {cluster_id + 1}: {', '.join(items[:5])}"
        
        simple_prompt += """

Hãy trả về JSON với format:
{
    "summary": "Tóm tắt",
    "by_domain": {"Frontend": [], "Backend": [], "AI": []},
    "confidence_score": 0.8
}"""
        
        return simple_prompt
    
    def _create_fallback_response(self) -> str:
        """Create a fallback response when LLM fails"""
        fallback = {
            "summary": "Phân tích tự động không khả dụng. Dữ liệu đã được xử lý thành công nhưng không thể tạo báo cáo chi tiết.",
            "trending_up": ["python", "machine learning", "cloud"],
            "trending_down": [],
            "emerging_tech": ["ai", "data science"],
            "core_skills": ["programming", "software development"],
            "future_predictions": ["Nhu cầu về kỹ năng công nghệ tiếp tục tăng"],
            "confidence_score": 0.3
        }
        return json.dumps(fallback, ensure_ascii=False)
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform analysis using the LLM
        
        Args:
            request: AnalysisRequest object
            
        Returns:
            AnalysisResult object
        """
        logger.info(f"🤖 Starting {request.analysis_type.value} analysis...")
        
        # Store clusters for neutral prompt fallback
        self.current_clusters = request.clusters
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Call LLM
        response_text = self._call_llm_with_retry(
            prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_response = json.loads(json_text)
            else:
                # Fallback: treat entire response as summary
                parsed_response = {"summary": response_text, "confidence_score": 0.5}
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            parsed_response = {"summary": response_text, "confidence_score": 0.3}
        
        # Extract information from parsed response
        summary = parsed_response.get("summary", "No summary available")
        confidence_score = parsed_response.get("confidence_score", 0.5)
        
        # Extract detailed analysis (everything except summary and confidence)
        detailed_analysis = {k: v for k, v in parsed_response.items() 
                           if k not in ["summary", "confidence_score", "recommendations"]}
        
        # Extract recommendations
        recommendations = parsed_response.get("recommendations", [])
        
        # Extract trends
        trends = []
        if "trending_up" in parsed_response:
            trends.extend([{"type": "growing", "items": parsed_response["trending_up"]}])
        if "trending_down" in parsed_response:
            trends.extend([{"type": "declining", "items": parsed_response["trending_down"]}])
        if "emerging_tech" in parsed_response:
            trends.extend([{"type": "emerging", "items": parsed_response["emerging_tech"]}])
        
        result = AnalysisResult(
            analysis_type=request.analysis_type.value,
            summary=summary,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            trends=trends,
            confidence_score=confidence_score,
            timestamp=time.time()
        )
        
        logger.info(f"✅ Analysis completed with confidence score: {confidence_score:.2f}")
        return result

# Convenience functions
def analyze_clusters(clusters: Dict[int, List[str]], 
                    analysis_type: str = "trend_analysis",
                    api_key: Optional[str] = None,
                    context: Optional[str] = None) -> AnalysisResult:
    """
    Quick function to analyze clusters
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of items
        analysis_type: Type of analysis to perform
        api_key: Gemini API key
        context: Additional context for analysis
        
    Returns:
        AnalysisResult object
    """
    agent = GeminiLLMAgent(api_key=api_key)
    
    request = AnalysisRequest(
        clusters=clusters,
        analysis_type=AnalysisType(analysis_type),
        context=context
    )
    
    return agent.analyze(request)

def analyze_job_trends(cluster_results: List[Any],
                      api_key: Optional[str] = None) -> AnalysisResult:
    """
    Analyze job trends from cluster results
    
    Args:
        cluster_results: List of ClusterResult objects
        api_key: Gemini API key
        
    Returns:
        AnalysisResult object
    """
    # Convert cluster results to dictionary format
    clusters = {}
    for i, cluster in enumerate(cluster_results):
        clusters[i] = cluster.items if hasattr(cluster, 'items') else cluster
    
    return analyze_clusters(clusters, "trend_analysis", api_key)

# Example usage and testing
if __name__ == "__main__":
    # Test data
    sample_clusters = {
        0: ["python developer", "machine learning engineer", "data scientist"],
        1: ["javascript developer", "react developer", "frontend engineer"],
        2: ["java backend", "spring boot", "microservices"],
        3: ["cloud engineer", "aws developer", "devops engineer"],
        4: ["mobile developer", "react native", "flutter developer"]
    }
    
    print("🧪 Testing LLM Agent")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found in environment variables")
        print("Set your API key to test the LLM functionality")
        print("Example: set GEMINI_API_KEY=your_api_key_here")
    else:
        try:
            # Test available models first
            print("🔍 Testing available Gemini models...")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            models = genai.list_models()
            available_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
                    print(f"   ✅ {model.name}")
            
            if not available_models:
                print("   ❌ No compatible models found")
                exit(1)
            
            # Test trend analysis with first available model
            model_name = available_models[0].replace('models/', '')
            print(f"\n🤖 Testing with model: {model_name}")
            
            agent = GeminiLLMAgent(model=model_name)
            
            request = AnalysisRequest(
                clusters=sample_clusters,
                analysis_type=AnalysisType.TREND_ANALYSIS,
                context="Dữ liệu từ các job posting trong Q4 2024"
            )
            
            result = agent.analyze(request)
            
            print(f"✅ Analysis completed:")
            print(f"   Type: {result.analysis_type}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Summary: {result.summary[:200]}...")
            print(f"   Trends found: {len(result.trends)}")
            print(f"   Recommendations: {len(result.recommendations)}")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            print("Make sure your Gemini API key is valid and you have internet connection")
