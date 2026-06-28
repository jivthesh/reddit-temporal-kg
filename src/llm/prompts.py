"""
src/llm/prompts.py

System prompt templates for entity extraction, query routing, and answer generation.
"""

ENTITY_EXTRACTION_PROMPT = """Analyze the following Reddit content and extract structured information.

Respond ONLY with a valid JSON object. No explanation before or after.

Reddit content:
{text}

Return this exact JSON structure:
{{
  "topics": ["topic1", "topic2"],
  "entities": [
    {{"text": "entity name", "type": "TOOL|PERSON|CONCEPT|COMPANY|PAPER"}}
  ],
  "sentiment": 0.7
}}

Rules:
- topics: 1-5 broad subject areas (e.g. "RAG", "AI Safety", "LLMs", "Open Source")
- entities: specific named things — tools (LangChain), people (Andrej Karpathy), papers, companies
- sentiment: float 0.0 (very negative/critical) to 1.0 (very positive/enthusiastic), 0.5 = neutral
- Return ONLY the JSON, nothing else"""


QUERY_CLASSIFICATION_PROMPT = """You are classifying a user's question about Reddit discussions.

Decide which search strategy is best for this question:
- "graph"  → The question is about RELATIONSHIPS or USERS (e.g., "who posted most about X?", "which users discuss Y?")
- "vector" → The question is SEMANTIC/TOPICAL (e.g., "what are concerns about X?", "explain Y")
- "hybrid" → The question needs BOTH graph + semantic (e.g., "how has sentiment changed?", "trend over time")

Question: {question}

Respond with ONLY one word: graph, vector, or hybrid"""


ANSWER_SYNTHESIS_PROMPT = """You are an expert analyst answering a question based on Reddit discussion data.

QUESTION:
{question}

RETRIEVED CONTEXT (top relevant posts and comments):
{context}

Instructions:
1. Answer the question directly and clearly
2. Use the context provided — don't make up information not in the context
3. Cite your sources using [1] [2] notation (matching the numbers in the context)
4. Note if you see conflicting viewpoints in the data
5. If the question asks about trends over time, highlight changes across time periods
6. Keep the answer focused — 150-300 words is ideal
7. If the context is insufficient to answer confidently, say so

Answer:"""


QUERY_UNDERSTANDING_PROMPT = """Parse this question about Reddit discussions into structured components.

Question: {question}

Respond ONLY with a valid JSON object:
{{
  "main_topic": "the primary subject (e.g. RAG, AI Safety, LLMs)",
  "time_range": "6_months | 1_year | 3_months | 1_month | null",
  "intent": "sentiment | trend | comparison | people | general",
  "keywords": ["keyword1", "keyword2"]
}}

Rules:
- main_topic: the core subject of the question
- time_range: if question mentions "last 6 months" → "6_months", "last year" → "1_year", no time → null
- intent: what kind of analysis does the user want?
  - sentiment: asking about positive/negative tone
  - trend: asking how something changed over time
  - comparison: asking to compare two things or periods
  - people: asking about specific users or authors
  - general: general information question
- keywords: 2-5 key terms to search for

Return ONLY the JSON, nothing else"""
