# Hybrid GraphRAG for Time-Series Reddit Intelligence

A question-answering and analytics system that ingests Reddit discussion threads, builds a unified temporal knowledge graph (Neo4j) alongside a vector index (Weaviate) in parallel, and resolves user queries via hybrid retrieval (Reciprocal Rank Fusion) and LLM response generation.

---

## 1. System Architecture

The system uses a **Parallel Dual-Store Architecture** to manage structured relationships and unstructured semantic data in parallel:

```
                  Raw Reddit Data (Hugging Face / PRAW API)
                                         ↓
                        Entity & Sentiment Extraction (LLM)
                                         ↓
                             Temporal Metadata Parser
                                         ↓
                 ┌───────────────────────┴───────────────────────┐
                 ↓                                               ↓
        ┌───────────────────┐                           ┌───────────────────┐
        │     Neo4j         │                           │    Weaviate       │
        │  (Graph DB)       │                           │   (Vector DB)     │
        │                   │                           │                   │
        │ Tracks relations  │                           │ Indexes semantic  │
        │ between Users,    │                           │ text chunks       │
        │ Posts, Comments,  │                           │ with temporal     │
        │ and LLM topics    │                           │ metadata tags     │
        └───────────────────┘                           └───────────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ↓
                                  Hybrid Retrieval
                        (Reciprocal Rank Fusion — RRF)
                                         ↓
                              LLM Answer Generation
```

![System Architecture Diagram](src/images/architecture%20diagram.png)


### Architectural Decisions

- **Graph Store (Neo4j)**: Used to trace comment hierarchies, find active users, and link entities. This helps resolve relational queries.
- **Vector Store (Weaviate)**: Used to index text chunks, allowing semantic retrieval even when queries use different terms from the source text.
- **Hybrid Fusion (RRF)**: Combines ranked lists from both databases using Reciprocal Rank Fusion (RRF). This ensures the LLM receives context containing both network connections and semantic matches.

---

## 2. Technology Stack & Design Decisions

- **Graph Database (Neo4j)**: Stores and connects posts, comments, topics, and users. This helps the system track relationships and reply threads.
- **Vector Database (Weaviate)**: Performs semantic search. It finds relevant posts by matching the meaning of a query, even if the exact keywords are not present.
- **Text Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)**: Generates text embeddings locally on a CPU. This runs completely free and offline without requiring external API keys.
- **Reddit Data Ingestion**: Supports streaming historical posts directly from Hugging Face archives or scraping live data concurrently using the Reddit PRAW library.
- **User Interface (Streamlit)**: A clean and simple web dashboard built in Python to run queries, view results, and see system logs.
- **LLM Client Wrapper**: A flexible client that connects with different AI models (like Google Gemini, Mistral, or Anthropic Claude) simply by updating the `.env` keys.

---

## 3. Installation & Setup (takes under 10 minutes)

### Prerequisites
- Python 3.10+
- Docker & Docker Compose

### Step 1: Start the Databases (Docker)
Start the Neo4j and Weaviate containers using the provided configuration:
```bash
docker-compose up -d
```

This launches:
- **Neo4j**: Available at `bolt://localhost:7689` (Browser console at http://localhost:7475)
- **Weaviate**: Available at `http://localhost:8080`

Verify both services are healthy:
```bash
# Check Neo4j status
docker logs reddit-kg-neo4j --tail 5

# Check Weaviate status
curl http://localhost:8080/v1/.well-known/ready
```

If you prefer running the containers individually:
```bash
# Start Neo4j
docker run -d --name reddit-kg-neo4j \
  -p 7689:7687 -p 7475:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.18-community

# Start Weaviate
docker run -d --name reddit-kg-weaviate \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e DEFAULT_VECTORIZER_MODULE=none \
  semitechnologies/weaviate:1.24.6
```

### Step 2: Install Dependencies
Set up a Python virtual environment and install project packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment
Copy the example environment configuration file:
```bash
cp .env.example .env
```
Open `.env` and configure the API credentials (e.g. `GEMINI_API_KEY`, `MISTRAL_API_KEY`, or `ANTHROPIC_API_KEY`).

---

## 4. Ingestion Modes

The system provides three ingestion workflows to fit different environments and credential constraints:

### Option A: Hugging Face Mode (Real Historical Data — Default)
Streams historical Reddit discussions from public Hugging Face archives.

**Why this is the default mode**:
- **Reddit API Access Restrictions**: Reddit recently updated its developer platform rules. New developer accounts can no longer get instant API keys. Instead, a manual application for approval is required, which can take several days or weeks.
- **Rate Limits**: The live Reddit API has strict rate limits that restrict how many comments and posts can be downloaded per minute.
- **Ease of Evaluation**: To ensure immediate project evaluation without waiting for API approvals, the system defaults to using a real dataset hosted on Hugging Face.

**How it works**:
It streams parquet files directly from public Hugging Face dataset archives (Pushshift Reddit dataset). It extracts relevant posts matching RAG keywords from the target subreddits, extracts entities using the LLM client, and populates both databases.

```bash
python demo.py --ingest --hf
```

### Option B: Live Reddit API Mode (Real-Time Data)
Scrapes subreddits in real-time. Requires a registered Reddit application at `reddit.com/prefs/apps`.

**How it works**:
It uses the `asyncpraw` library to connect directly to Reddit API endpoints. It scrapes hot posts and complete comment trees from subreddits concurrently, analyzes sentiment and entities, and stores them in Weaviate and Neo4j.

1. Add the credentials to `.env`:
   ```env
   REDDIT_CLIENT_ID="your_client_id"
   REDDIT_CLIENT_SECRET="your_client_secret"
   REDDIT_USER_AGENT="your_user_agent"
   ```
2. Run live ingestion:
   ```bash
   python demo.py --ingest --real --limit 15
   ```

### Option C: Synthetic Mode (Local Offline Fallback)
Generates mock Reddit datasets locally for offline testing or air-gapped environments.

**How it works**:
It runs entirely local and offline. It reads a pre-built mock dataset of 150 posts with comments and sentiment timelines from `synthetic_reddit_data.json` (or generates it if missing) and populates the databases without any API or network calls.

```bash
python demo.py --ingest --synthetic
```


---

## 5. Running Queries & Launching the Dashboard

### Command-Line Query Runner
Execute the pre-defined evaluation test suite to verify the hybrid search logic (takes under 2 minutes to run all queries):
```bash
python demo.py --query
```

**What it does**:
It runs 4 test queries. For each query, it:
1. Classifies the question profile (semantic, relational, or hybrid).
2. Performs parallel searches on Neo4j (graph) and Weaviate (vector).
3. Merges the results using Reciprocal Rank Fusion (RRF).
4. Generates a final cited response using the LLM.
5. Prints the side-by-side comparison of **Graph-only**, **Vector-only**, and **Fused** results directly to the console.

### Dashboard Interface
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to the address shown in the terminal (typically `http://localhost:8501`).

**What it does**:
It launches a local web server with an interactive web UI. The interface allows you to:
1. Type custom questions and view synthesized answers with inline citations.
2. Select time ranges (e.g. last 6 months) and toggle between hybrid, graph-only, or vector-only retrieval modes.
3. Compare raw hits retrieved from Neo4j and Weaviate databases side-by-side.
4. View real-time database statistics and temporal analytics charts (like quarterly sentiment shifts and topic counts) queried dynamically from the Neo4j instance.


---

## 6. Evaluation Test Suite (`demo.py`)

The query suite verifies that the system selects the correct database retrieval path based on the user's question:

1. **Semantic Question (Vector-Dominant)**
   - *Query*: `"What are the main concerns and criticisms about RAG systems?"`
   - *How it works*: Searches Weaviate vector database to retrieve matching text chunks, even if the wording differs.
2. **Relationship Question (Graph-Dominant)**
   - *Query*: `"Who are the most active voices discussing AI safety on Reddit?"`
   - *How it works*: Traverses the Neo4j graph to find active users talking about safety, ranking them by activity and average sentiment.
3. **Hybrid Question (Combined)**
   - *Query*: `"How have community discussions about open-source LLMs shifted recently?"`
   - *How it works*: Combines semantic matches from Weaviate with network topics and user metadata from Neo4j.
4. **Time-Series Comparison (Temporal)**
   - *Query*: `"How has sentiment around RAG changed in the last 6 months?"`
   - *How it works*: Converts time words (like "last 6 months") to Unix timestamps, filters both databases by date range, and groups results into quarterly metrics.

---

## 7. Configuration Reference (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `REDDIT_CLIENT_ID` | OAuth Client ID from Reddit developer portal | None |
| `REDDIT_CLIENT_SECRET` | OAuth Client Secret from Reddit developer portal | None |
| `REDDIT_USER_AGENT` | Descriptive agent string for API request identification | `RedditKG/1.0` |
| `GEMINI_API_KEY` | Google Gemini API credentials | None |
| `MISTRAL_API_KEY` | Mistral Developer Platform credentials | None |
| `ANTHROPIC_API_KEY` | Anthropic Claude API credentials | None |
| `LLM_PROVIDER` | Explicit provider override (`gemini`, `mistral`, `anthropic`) | None (Auto-detected) |
| `NEO4J_URI` | Neo4j Connection Endpoint | `bolt://localhost:7689` |
| `NEO4J_USER` | Neo4j Username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j Password | `password` |
| `WEAVIATE_URL` | Weaviate REST Endpoint | `http://127.0.0.1:8080` |
| `RAG_KEYWORDS_FILTER` | Limits ingestion to RAG-relevant posts | `true` |
| `MIN_COMMENTS_FOR_RELEVANCE` | Minimum comments required on posts during ingestion | `2` |
