"""
Test script for Gemini API connectivity and model availability
"""


import os
import sys

# Load environment variables from .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv chưa được cài. Để tự động load .env, hãy chạy: pip install python-dotenv")

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai library not installed. Please install with: pip install google-generativeai")
    sys.exit(1)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("Set your API key before running this script.")
        print("Example (cmd): set GEMINI_API_KEY=your_api_key_here")
        return
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini API key loaded.")
        print("🔍 Listing available Gemini models...")
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"   ✅ {model.name}")
        if not available_models:
            print("❌ No compatible models found.")
            return
        # Test with specific model: gemini-2.5-flash
        test_model = "gemini-2.5-flash"
        print(f"\n🤖 Testing content generation with model: {test_model}")
        
        try:
            client = genai.GenerativeModel(test_model)
        except Exception as e:
            print(f"❌ Failed to initialize {test_model}, using first available model")
            model_name = available_models[0].replace('models/', '')
            client = genai.GenerativeModel(model_name)
            test_model = model_name
        
        # Test prompt from llm_agent.py
        prompt = """Bạn là một chuyên gia phân tích thị trường việc làm công nghệ. 
Nhiệm vụ của bạn là phân tích các nhóm kỹ năng/công nghệ và đưa ra những nhận định sâu sắc về xu hướng thị trường.

Dữ liệu đầu vào - Các nhóm kỹ năng/công nghệ:

Nhóm 6: react, frontend, frontend react, build
Nhóm 1: engineer, developer, cloud, development, application, design, microservices, scalable
Nhóm 2: skill, backend, kubernetes, deployment, learning data
Nhóm 10: data, required, scripting, deep learning, knowledge, pytorch       
Nhóm 9: python, python scripting, docker, panda, year
Nhóm 7: network
Nhóm 4: preferred, certification, rest, sql
Nhóm 3: tensorflow, testing, javascripttypescript, computing, looking, deep, model, tensorflow pytorch
Nhóm 5: learning, must, senior, apis, scikitlearn
Nhóm 8: rest apis

Bối cảnh bổ sung: Phân tích loại: trend_analysis

Hãy phân tích và nhóm lại các kỹ năng theo các tiêu chí sau:

1. **Theo lĩnh vực**: Frontend, Backend, Data Science, DevOps, AI/ML, Mobile, etc.
2. **Theo mức độ**: Entry-level, Mid-level, Senior-level, Expert-level
3. **Theo xu hướng**: Hot skills, Stable skills, Declining skills
4. **Theo mối quan hệ**: Kỹ năng bổ trợ cho nhau, kỹ năng thay thế

Trả về kết quả dưới dạng JSON với format:
{
    "summary": "Tóm tắt về cách nhóm kỹ năng",
    "by_domain": {"Frontend": [], "Backend": [], ...},
    "by_level": {"Entry": [], "Mid": [], "Senior": [], "Expert": []},
    "by_trend": {"Hot": [], "Stable": [], "Declining": []},
    "skill_relationships": {"complementary": [], "alternatives": []},
    "confidence_score": 0.9
}

Lưu ý: Chỉ trả về JSON hợp lệ, không có text bổ sung."""
        
        print(f"📤 Sending prompt to {test_model}...")
        response = client.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            print("\n📥 Gemini API response:")
            print(response.text)
        else:
            print("❌ No response text received from Gemini API.")
    except Exception as e:
        print(f"❌ Error during Gemini API test: {e}")

if __name__ == "__main__":
    main()
