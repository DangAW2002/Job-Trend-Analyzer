"""
Web UI Module for Job Trend Analyzer
Sử dụng Streamlit để tạo giao diện web thân thiện
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import tempfile
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from config import Config, api_config, model_config, processing_config, path_config
    from pipeline import JobTrendPipeline, PipelineConfig
    from preprocessing import TextPreprocessor
    from ngram_extractor import NGramExtractor
    from embedding import EmbeddingGenerator
    from cluster import EmbeddingClusterer
    from llm_agent import GeminiLLMAgent, AnalysisType, AnalysisRequest
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Job Trend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2e8b57;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stAlert {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

class JobTrendAnalyzerUI:
    def __init__(self):
        self.config = None
        self.pipeline = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'job_data' not in st.session_state:
            st.session_state.job_data = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'cluster_results' not in st.session_state:
            st.session_state.cluster_results = None
        if 'config_loaded' not in st.session_state:
            st.session_state.config_loaded = False
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
    
    def setup_sidebar(self):
        """Setup sidebar with configuration and controls"""
        st.sidebar.title("🔧 Cấu hình")
        
        # API Keys section
        st.sidebar.subheader("API Keys")
        # Lấy giá trị mặc định từ biến môi trường (đã load từ .env)
        default_together = os.getenv("TOGETHER_API_KEY", "")
        default_gemini = os.getenv("GEMINI_API_KEY", "")
        
        # Together API Key
        together_api_key = st.sidebar.text_input(
            "Together API Key:",
            value=default_together,
            type="password",
            help="API key cho Together AI embeddings"
        )
        
        # Gemini API Key
        gemini_api_key = st.sidebar.text_input(
            "Gemini API Key:",
            value=default_gemini,
            type="password",
            help="API key cho Google Gemini LLM"
        )
        
        # Load configuration
        if st.sidebar.button("💾 Tải cấu hình"):
            try:
                # Set environment variables (cho phép thay đổi runtime)
                if together_api_key:
                    os.environ["TOGETHER_API_KEY"] = together_api_key
                if gemini_api_key:
                    os.environ["GEMINI_API_KEY"] = gemini_api_key
                
                # Create pipeline config
                pipeline_config = PipelineConfig()
                self.pipeline = JobTrendPipeline(pipeline_config)
                
                # Save to session state
                st.session_state.config_loaded = True
                st.session_state.pipeline = self.pipeline
                
                st.sidebar.success("✅ Cấu hình đã được tải!")
            except Exception as e:
                st.sidebar.error(f"❌ Lỗi tải cấu hình: {e}")
                st.session_state.config_loaded = False
        
        # Processing parameters
        st.sidebar.subheader("⚙️ Tham số xử lý")
        
        # N-gram settings
        st.sidebar.selectbox(
            "N-gram type:",
            ["word", "char"],
            key="ngram_type"
        )
        
        st.sidebar.slider(
            "N-gram range (min):",
            1, 3, 1,
            key="ngram_min"
        )
        
        st.sidebar.slider(
            "N-gram range (max):",
            1, 5, 2,
            key="ngram_max"
        )
        
        st.sidebar.slider(
            "Max features:",
            100, 5000, 1000,
            key="max_features"
        )
        
        # Clustering settings
        st.sidebar.slider(
            "Số cluster:",
            2, 20, 5,
            key="n_clusters"
        )
        
        # Analysis type
        st.sidebar.selectbox(
            "Loại phân tích:",
            ["trend_analysis", "skill_grouping", "market_insights", "career_recommendations"],
            key="analysis_type"
        )
    
    def data_input_section(self):
        """Data input section"""
        st.markdown('<div class="section-header">📄 Nhập dữ liệu Job Descriptions</div>', unsafe_allow_html=True)
        
        # Input methods
        input_method = st.selectbox(
            "Chọn phương thức nhập dữ liệu:",
            ["Upload file JSON", "Nhập text trực tiếp", "Sử dụng dữ liệu mẫu"]
        )
        
        if input_method == "Upload file JSON":
            uploaded_file = st.file_uploader(
                "Chọn file JSON:",
                type=["json"],
                help="File JSON chứa danh sách job descriptions"
            )
            
            if uploaded_file is not None:
                try:
                    job_data = json.load(uploaded_file)
                    if isinstance(job_data, list):
                        st.session_state.job_data = job_data
                        st.success(f"✅ Đã tải {len(job_data)} job descriptions")
                    else:
                        st.error("❌ File JSON phải chứa một array của job descriptions")
                except Exception as e:
                    st.error(f"❌ Lỗi đọc file: {e}")
        
        elif input_method == "Nhập text trực tiếp":
            job_text = st.text_area(
                "Nhập job descriptions (mỗi job một dòng):",
                height=200,
                help="Mỗi job description trên một dòng riêng"
            )
            
            if st.button("💾 Lưu job descriptions"):
                if job_text.strip():
                    job_data = [job.strip() for job in job_text.split('\n') if job.strip()]
                    st.session_state.job_data = job_data
                    st.success(f"✅ Đã lưu {len(job_data)} job descriptions")
                else:
                    st.warning("⚠️ Vui lòng nhập job descriptions")
        
        elif input_method == "Sử dụng dữ liệu mẫu":
            if st.button("📋 Tải dữ liệu mẫu"):
                sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample_jobs.json")
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        job_data = json.load(f)
                        st.session_state.job_data = job_data
                        st.success(f"✅ Đã tải {len(job_data)} job descriptions mẫu")
                except Exception as e:
                    st.error(f"❌ Không thể tải dữ liệu mẫu: {e}")
        
        # Display current data
        if st.session_state.job_data:
            st.markdown('<div class="section-header">📊 Dữ liệu hiện tại</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Số lượng job descriptions:", len(st.session_state.job_data))
            
            with col2:
                avg_length = sum(len(job) for job in st.session_state.job_data) / len(st.session_state.job_data)
                st.metric("Độ dài trung bình:", f"{avg_length:.0f} ký tự")
            
            # Show sample data
            with st.expander("👀 Xem dữ liệu mẫu"):
                for i, job in enumerate(st.session_state.job_data[:3]):
                    st.write(f"**Job {i+1}:**")
                    st.text(job[:200] + "..." if len(job) > 200 else job)
                    st.write("---")
    
    def analysis_section(self):
        """Analysis execution section"""
        st.markdown('<div class="section-header">🔍 Thực hiện phân tích</div>', unsafe_allow_html=True)
        
        if not st.session_state.config_loaded:
            st.warning("⚠️ Vui lòng tải cấu hình trước khi phân tích")
            return
        
        if not st.session_state.job_data:
            st.warning("⚠️ Vui lòng nhập dữ liệu job descriptions trước")
            return
        
        # Analysis button
        if st.button("🚀 Bắt đầu phân tích", type="primary"):
            # Get pipeline from session state or self
            pipeline = st.session_state.get('pipeline') or self.pipeline
            
            if not pipeline:
                st.error("❌ Pipeline chưa được khởi tạo. Vui lòng tải cấu hình trước.")
                return
            
            # Create progress placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔧 Cập nhật cấu hình pipeline...")
                progress_bar.progress(10)
                
                # Update pipeline configuration
                pipeline.ngram_extractor.ngram_range = (
                    st.session_state.ngram_min,
                    st.session_state.ngram_max
                )
                pipeline.ngram_extractor.analyzer = st.session_state.ngram_type
                pipeline.ngram_extractor.max_features = st.session_state.max_features
                pipeline.clusterer.n_clusters = st.session_state.n_clusters
                
                status_text.text("🚀 Bắt đầu phân tích...")
                progress_bar.progress(20)
                
                # Run analysis with progress updates
                status_text.text("📝 Đang tiền xử lý văn bản...")
                progress_bar.progress(30)
                
                status_text.text("🔤 Đang trích xuất n-grams...")
                progress_bar.progress(50)
                
                status_text.text("🧠 Đang tạo embeddings...")
                progress_bar.progress(70)
                
                status_text.text("📊 Đang phân cụm...")
                progress_bar.progress(85)
                
                status_text.text("🤖 Đang phân tích với LLM...")
                progress_bar.progress(95)
                
                # Run actual analysis
                results = pipeline.run(
                    job_descriptions=st.session_state.job_data,
                    context=f"Phân tích loại: {st.session_state.analysis_type}"
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Phân tích hoàn thành!")
                
                st.session_state.analysis_results = results
                st.session_state.cluster_results = results.clusters
                
                # Clear progress indicators after a moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Phân tích hoàn thành!")
                st.balloons()  # Celebration animation
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Lỗi trong quá trình phân tích: {e}")
                st.write("**Chi tiết lỗi:**")
                st.code(str(e))
    
    def results_section(self):
        """Display analysis results"""
        if not st.session_state.analysis_results:
            return
        
        st.markdown('<div class="section-header">📈 Kết quả phân tích</div>', unsafe_allow_html=True)
        
        results = st.session_state.analysis_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Số N-grams:", len(results.ngrams))
        
        with col2:
            st.metric("Số Clusters:", len(results.clusters))
        
        with col3:
            # Get first analysis result's confidence
            first_analysis = next(iter(results.analyses.values()), None) if results.analyses else None
            confidence = first_analysis.confidence_score if first_analysis else 0
            st.metric("Confidence Score:", f"{confidence:.2f}")
        
        with col4:
            st.metric("Thời gian xử lý:", f"{results.processing_time:.1f}s")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Clusters", "📊 Biểu đồ", "🤖 LLM Analysis", "📋 Chi tiết", "🔧 Debug"])
        
        with tab1:
            self.display_clusters()
        
        with tab2:
            self.display_charts()
        
        with tab3:
            self.display_llm_analysis()
        
        with tab4:
            self.display_detailed_results()
        
        with tab5:
            self.display_debug_info()
    
    def display_clusters(self):
        """Display cluster results"""
        if not st.session_state.cluster_results:
            st.info("Không có kết quả cluster")
            return
        
        for cluster in st.session_state.cluster_results:
            with st.expander(f"🏷️ Cluster {cluster.cluster_id + 1} ({cluster.size} items)"):
                # Display items as tags
                items_text = ", ".join(cluster.items[:20])  # Limit display
                if len(cluster.items) > 20:
                    items_text += f" ... (và {len(cluster.items) - 20} items khác)"
                st.write(items_text)
                
                # Show average score
                st.write(f"**Điểm trung bình:** {cluster.avg_score:.3f}")
    
    def display_charts(self):
        """Display visualization charts"""
        if not st.session_state.cluster_results:
            st.info("Không có dữ liệu để vẽ biểu đồ")
            return
        
        # Cluster size distribution
        cluster_sizes = [cluster.size for cluster in st.session_state.cluster_results]
        cluster_labels = [f"Cluster {cluster.cluster_id + 1}" for cluster in st.session_state.cluster_results]
        
        # Bar chart of cluster sizes
        fig_bar = px.bar(
            x=cluster_labels,
            y=cluster_sizes,
            title="Phân bố kích thước các Cluster",
            labels={'x': 'Cluster', 'y': 'Số lượng items'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie chart of cluster distribution
        fig_pie = px.pie(
            values=cluster_sizes,
            names=cluster_labels,
            title="Tỷ lệ phân bố Clusters"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def display_llm_analysis(self):
        """Display LLM analysis results"""
        if not st.session_state.analysis_results:
            return
        
        try:
            analyses = st.session_state.analysis_results.analyses
            if not analyses:
                st.info("Không có kết quả phân tích LLM")
                return
            
            # Display each analysis type
            for analysis_type, analysis_result in analyses.items():
                st.subheader(f"🤖 {analysis_type.replace('_', ' ').title()}")
                
                # Summary
                st.write("**📝 Tóm tắt:**")
                st.write(analysis_result.summary)
                
                # Confidence score
                st.metric("Độ tin cậy:", f"{analysis_result.confidence_score:.2%}")
                
                # Detailed analysis
                if analysis_result.detailed_analysis:
                    st.write("**🔍 Phân tích chi tiết:**")
                    
                    for key, value in analysis_result.detailed_analysis.items():
                        if isinstance(value, list) and value:
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                st.write(f"• {item}")
                        elif isinstance(value, dict) and value:
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, list):
                                    st.write(f"• **{subkey}:** {', '.join(map(str, subvalue))}")
                                else:
                                    st.write(f"• **{subkey}:** {subvalue}")
                        elif value:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Recommendations
                if analysis_result.recommendations:
                    st.write("**💡 Khuyến nghị:**")
                    for rec in analysis_result.recommendations:
                        st.write(f"• {rec}")
                
                # Trends (if available)
                if hasattr(analysis_result, 'trends') and analysis_result.trends:
                    st.write("**📈 Xu hướng:**")
                    for trend in analysis_result.trends:
                        if isinstance(trend, dict):
                            trend_type = trend.get('type', 'Unknown')
                            trend_items = trend.get('items', [])
                            if trend_items:
                                st.write(f"**{trend_type.title()}:** {', '.join(trend_items)}")
                
                st.write("---")
                
        except Exception as e:
            st.error(f"❌ Lỗi hiển thị phân tích LLM: {e}")
            st.info("Có thể LLM chưa phân tích hoặc dữ liệu không đúng định dạng")
    
    def display_debug_info(self):
        """Display debug information"""
        if not st.session_state.analysis_results:
            st.info("Chưa có kết quả phân tích")
            return
        
        results = st.session_state.analysis_results
        
        st.subheader("🔧 Thông tin Debug")
        
        # Pipeline info
        st.write("**Pipeline Configuration:**")
        if hasattr(results, 'pipeline_config'):
            config = results.pipeline_config
            st.write(f"• N-gram range: {config.ngram_range}")
            st.write(f"• Top K n-grams: {config.top_k_ngrams}")
            st.write(f"• Number of clusters: {config.n_clusters}")
            st.write(f"• Analysis types: {config.analysis_types}")
        
        # Processing stats
        st.write("**Processing Statistics:**")
        st.write(f"• Original texts: {len(results.original_texts)}")
        st.write(f"• Cleaned texts: {len(results.cleaned_texts)}")
        st.write(f"• N-grams extracted: {len(results.ngrams)}")
        st.write(f"• Embeddings created: {results.embeddings_created}")
        st.write(f"• Embedding dimension: {results.embedding_dimension}")
        st.write(f"• Clusters found: {len(results.clusters)}")
        st.write(f"• Processing time: {results.processing_time:.2f}s")
        
        # Analysis status
        st.write("**LLM Analysis Status:**")
        if hasattr(results, 'analyses') and results.analyses:
            for analysis_type, analysis in results.analyses.items():
                confidence = analysis.confidence_score if hasattr(analysis, 'confidence_score') else 0
                st.write(f"• {analysis_type}: ✅ Completed (confidence: {confidence:.2f})")
        else:
            st.write("• No LLM analysis available")
        
        # Environment check
        st.write("**Environment Check:**")
        together_key = "✅ Set" if os.getenv("TOGETHER_API_KEY") else "❌ Missing"
        gemini_key = "✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Missing"
        st.write(f"• Together API Key: {together_key}")
        st.write(f"• Gemini API Key: {gemini_key}")
        
        # Sample data preview
        with st.expander("📄 Sample Data Preview"):
            st.write("**Original texts (first 2):**")
            for i, text in enumerate(results.original_texts[:2]):
                st.write(f"{i+1}. {text[:200]}...")
            
            st.write("**Top N-grams (first 10):**")
            for i, (ngram, score) in enumerate(results.ngrams[:10]):
                st.write(f"{i+1}. {ngram} (score: {score:.2f})")
    
    def display_detailed_results(self):
        """Display detailed raw results"""
        if not st.session_state.analysis_results:
            return
        
        # Raw JSON view
        st.subheader("📋 Dữ liệu thô (JSON)")
        
        # Convert PipelineResult to dict for display
        try:
            if hasattr(st.session_state.analysis_results, 'to_dict'):
                results_dict = st.session_state.analysis_results.to_dict()
            else:
                # Fallback: convert using asdict
                from dataclasses import asdict
                results_dict = asdict(st.session_state.analysis_results)
            st.json(results_dict)
        except Exception as e:
            st.error(f"Không thể hiển thị dữ liệu: {e}")
            st.write("**Raw object:**")
            st.write(st.session_state.analysis_results)
        
        # Download results
        if st.button("💾 Tải xuống kết quả"):
            # Convert to JSON string
            try:
                if hasattr(st.session_state.analysis_results, 'to_dict'):
                    results_dict = st.session_state.analysis_results.to_dict()
                else:
                    from dataclasses import asdict
                    results_dict = asdict(st.session_state.analysis_results)
                
                results_json = json.dumps(results_dict, indent=2, ensure_ascii=False)
                
                # Create download link
                st.download_button(
                    label="📁 Download results.json",
                    data=results_json,
                    file_name=f"job_trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Không thể tạo file download: {e}")
    
    def main(self):
        """Main application function"""
        # Header
        st.markdown('<div class="main-header">📊 Job Trend Analyzer</div>', unsafe_allow_html=True)
        st.markdown("**Phân tích xu hướng việc làm IT với AI và Machine Learning**")
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Main content
        self.data_input_section()
        
        # Analysis section
        if st.session_state.job_data:
            self.analysis_section()
        
        # Results section
        if st.session_state.analysis_results:
            self.results_section()
        
        # Footer
        st.markdown("---")
        st.markdown("*Job Trend Analyzer - Phát triển bởi AI Assistant*")

# Run the app
if __name__ == "__main__":
    app = JobTrendAnalyzerUI()
    app.main()
