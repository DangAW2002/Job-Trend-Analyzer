# 📊 Job Trend Analyzer

Dự án phân tích xu hướng thị trường việc làm bằng cách kết hợp **n-gram + embedding + Gemini LLM Agent**, được triển khai chuyên nghiệp theo kiến trúc LangChain hoặc LangGraph.

---

## 🧱 Kiến trúc tổng quan

```mermaid
flowchart TD
  A[Job Descriptions] --> B[Text Cleaning]
  B --> C[N-gram Extractor]
  C --> D[Embedding with m2-bert]
  D --> E[Clustering]
  E --> F[LLM Agent (Gemini)]
  F --> G[Trend Report Output]
```

### 📁 Cấu trúc thư mục
```
job-trend-analyzer/
│
├── data/                     # Dữ liệu gốc và đã xử lý
│   ├── raw/                  # Dữ liệu raw (json, html, csv, etc.)
│   └── processed/            # Dữ liệu sau khi clean
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Thông tin config (API key, model, etc.)
│   ├── preprocessing.py      # Làm sạch văn bản
│   ├── ngram_extractor.py    # Tạo n-gram
│   ├── embedding.py          # Gọi Together API để tạo embedding
│   ├── cluster.py            # Gom cụm bằng k-means
│   ├── llm_agent.py          # Tương tác với Gemini Agent
│   └── pipeline.py           # Tích hợp toàn bộ pipeline
│
├── notebooks/                # Jupyter phân tích & trực quan hóa
│
├── requirements.txt
├── README.md
└── main.py
```

### ⚙️ Công nghệ sử dụng
| Thành phần   | Công nghệ                                      |
|-------------|------------------------------------------------|
| LLM Agent   | Gemini API (google-generativeai)               |
| Embedding   | togethercomputer/m2-bert-80M-32k-retrieval      |
| Clustering  | scikit-learn (KMeans)                          |
| Orchestrator| LangChain hoặc LangGraph                       |
| Visualization| Streamlit, Gradio, Plotly (tuỳ chọn)           |

---

## 🔄 Luồng xử lý chi tiết

1. **Clean văn bản**
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return " ".join([t for t in tokens if t not in STOPWORDS])
```

2. **Tạo n-gram**
```python
from sklearn.feature_extraction.text import CountVectorizer

def get_ngrams(texts, n=2, top_k=50):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
    return sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
```

3. **Embedding với Together API**
```python
from together import Together

def get_embeddings(phrases: list[str]):
    client = Together()
    result = []
    for phrase in phrases:
        response = client.embeddings.create(
            model="togethercomputer/m2-bert-80M-32k-retrieval",
            input=phrase
        )
        result.append((phrase, response.data[0].embedding))
    return result
```

4. **Gom cụm (clustering)**
```python
from sklearn.cluster import KMeans

def cluster_embeddings(embeddings, n_clusters=10):
    vectors = [emb for _, emb in embeddings]
    phrases = [phrase for phrase, _ in embeddings]

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(vectors)

    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(phrases[idx])
    return clusters
```

5. **LLM Agent Gemini phân tích**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

def analyze_clusters(clusters: dict):
    prompt = "Dưới đây là các nhóm kỹ năng/nghề nghiệp theo cụm nghĩa:\n\n"
    for i, phrases in clusters.items():
        prompt += f"Nhóm {i+1}: {', '.join(phrases)}\n"
    prompt += "\nHãy phân tích xu hướng thị trường dựa trên các nhóm trên."

    model = ChatGoogleGenerativeAI(model="gemini-pro")
    return model.invoke(prompt)
```

6. **Tích hợp LangChain / LangGraph**

LangChain Tool:
```python
from langchain.tools import tool

@tool
def analyze_job_trend(texts: list[str]):
    cleaned = [clean_text(t) for t in texts]
    ngrams = get_ngrams(cleaned, n=2, top_k=100)
    phrases = [g[0] for g in ngrams]
    embeddings = get_embeddings(phrases)
    clusters = cluster_embeddings(embeddings)
    result = analyze_clusters(clusters)
    return result
```

LangGraph (sơ đồ pipeline dạng node)
```yaml
Node1: Clean → Node2: N-gram → Node3: Embed → Node4: Cluster → Node5: Analyze
```

---

---

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Thiết lập API keys
Tạo file `.env` từ template:
```bash
cp .env.example .env
```

Chỉnh sửa file `.env` với API keys của bạn:
```env
TOGETHER_API_KEY=your_together_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Chạy ứng dụng

#### 🌐 Web UI (Khuyến nghị)
```bash
# Cách 1: Sử dụng batch file (Windows)
run_ui.bat

# Cách 2: Sử dụng Python launcher
python launch_ui.py

# Cách 3: Chạy trực tiếp Streamlit
streamlit run src\web_ui.py
```

#### 💻 Command Line Interface
```bash
# Chế độ demo với dữ liệu mẫu
python main.py --demo

# Phân tích file cụ thể
python main.py --input data/raw/sample_jobs.json --output results.json

# Xem tất cả options
python main.py --help
```

#### 📔 Jupyter Notebook
```bash
jupyter notebook notebooks/job_trend_analyzer_demo.ipynb
```

## 📈 Ví dụ kết quả đầu ra
```json
{
  "trend_summary": "Các nhóm kỹ năng về AI và backend đang tăng mạnh, đặc biệt là 'machine learning', 'python backend' và 'data engineer'. Ngược lại, các nhóm liên quan đến legacy system như 'cobol developer' có xu hướng giảm.",
  "clusters": {
    "AI/ML": ["machine learning", "deep learning", "ai engineer"],
    "Backend": ["java backend", "python backend", "spring boot"],
    ...
  }
}
```

---

## 🚀 Mở rộng tương lai
- Gắn mốc thời gian → phân tích theo quý/năm
- Lưu kết quả vào vector database (FAISS, Weaviate)
- Trực quan hóa cụm nghĩa bằng UMAP/PCA
- Tạo dashboard tương tác (Streamlit)

---

## 🧾 Yêu cầu hệ thống
| Thành phần | Mô tả |
|------------|-------|
| Python     | >=3.10|
| Libraries  | together, scikit-learn, langchain, google-generativeai, sentence-transformers |
| API Keys   | Together API + Gemini API |

---

## ✅ Tiếp theo?
- Viết module pipeline.py tích hợp mọi bước?
- Tạo lệnh CLI hoặc app Streamlit chạy từ dữ liệu thực?
- Cung cấp tập dữ liệu demo để thử nghiệm?
