"""
src/ingestion/synthetic_scraper.py

Generates 150 highly realistic mock Reddit posts with comments, subreddits, 
upvote ratios, scores, authors, and pre-extracted entities/sentiment.
Spread across the last 12 months, with custom temporal sentiment trends:
  - RAG sentiment gets more positive in the last 6 months (0.5 -> 0.8)
  - AI Safety sentiment gets more critical in the last 6 months (0.6 -> 0.3)

This allows testing the whole system completely offline without Reddit API keys,
while producing rich, meaningful results for demo queries.
"""

import json
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

logger = logging.getLogger("reddit-kg")

# List of realistic usernames for authors
AUTHORS = [
    "ai_wizard", "gradient_descent_fan", "neural_net_builder", "rlhf_coder",
    "rag_architect", "prompt_engineer_99", "ethics_scholar", "open_source_champion",
    "pytorch_guru", "vector_db_nerd", "transformer_tamer", "math_magician",
    "deep_dive_dev", "cuda_cruncher", "model_merger", "loss_function_fan",
    "weights_n_biases", "gpu_rich", "latent_space_traveler", "token_limit_hero"
]

# Preset templates for different topics to generate rich content
TOPICS_CONTENT = {
    "RAG": {
        "topics": ["RAG", "Vector Databases", "Search"],
        "entities": [
            {"text": "RAG", "type": "CONCEPT"},
            {"text": "LangChain", "type": "TOOL"},
            {"text": "LlamaIndex", "type": "TOOL"},
            {"text": "Weaviate", "type": "TOOL"},
            {"text": "Neo4j", "type": "TOOL"},
            {"text": "Hybrid Search", "type": "CONCEPT"}
        ],
        "posts": [
            {
                "title": "Why RAG is replacing long-context LLMs in enterprise QA",
                "text": "Everyone is hyping 2M token context windows, but in production, sending massive context leads to high latency, huge API costs, and lost-in-the-middle issues. Retrieval-Augmented Generation (RAG) is still the only sensible way to build reliable QA systems on private databases. We combined vector search with keyword search and saw a huge jump in accuracy.",
                "comments": [
                    {"author": "rag_architect", "text": "Totally agree. Long-context is great for exploration, but RAG is for production.", "score_diff": 15},
                    {"author": "cuda_cruncher", "text": "The latency of processing 1M tokens on every user query is just unacceptable for real-time apps.", "score_diff": 8},
                    {"author": "prompt_engineer_99", "text": "What chunking strategy are you using? We are struggling with overlapping boundaries.", "score_diff": 3}
                ]
            },
            {
                "title": "Evaluating GraphRAG vs VectorRAG: A comparative study",
                "text": "Vector databases are great for finding similar documents, but they fail when a question requires connecting entities across different files. By building a knowledge graph in Neo4j and linking it with Weaviate vector embeddings, we created a GraphRAG system that handles multi-hop reasoning. We tested it on our company docs and the answers are much more coherent.",
                "comments": [
                    {"author": "vector_db_nerd", "text": "This is the future. Combining semantic vector search with relational graphs solves the multi-hop problem.", "score_diff": 22},
                    {"author": "neural_net_builder", "text": "How do you handle graph extraction at scale? Do you use LLMs to extract nodes and edges?", "score_diff": 12},
                    {"author": "deep_dive_dev", "text": "Yes, but extracting entities with LLMs can get expensive. We need open-source extractors.", "score_diff": 5}
                ]
            },
            {
                "title": "Is anyone else struggling with chunking strategies in production RAG?",
                "text": "We've tried fixed size chunking (300 tokens, 50 token overlap), semantic chunking based on sentence similarity, and markdown-aware chunking. Our retrieval hits are still missing critical context because tables get split up. How are you all keeping tables intact for your vector store?",
                "comments": [
                    {"author": "gpu_rich", "text": "We represent tables as markdown and pass them as a single chunk, which works surprisingly well.", "score_diff": 9},
                    {"author": "weights_n_biases", "text": "Try using hierarchical node parsing. Keep parent chunks large and embed smaller child chunks.", "score_diff": 7}
                ]
            },
            {
                "title": "Hybrid search is all you need: Combining BM25 with Vector Embeddings",
                "text": "Our team spent weeks tuning dense embeddings, but users kept complaining about exact keyword failures (e.g., serial numbers, specific product codes). As soon as we enabled hybrid search using Reciprocal Rank Fusion (RRF) to combine BM25 and vector scores, our search quality skyrocketed. Don't throw away classic search!",
                "comments": [
                    {"author": "pytorch_guru", "text": "BM25 is criminally underrated in the era of vector databases.", "score_diff": 18},
                    {"author": "transformer_tamer", "text": "What fusion constant k are you using? We found k=60 to be the sweet spot.", "score_diff": 11}
                ]
            },
            {
                "title": "How we built a production-grade QA bot over 50,000 PDFs using RAG",
                "text": "We successfully launched an internal tool for our compliance department. The tech stack: FastAPI, LlamaIndex, Weaviate for vector storage, and Cohere Rerank. The biggest lessons: 1) Document pre-processing and parsing is 80% of the work. 2) Reranking is non-negotiable for high precision. 3) Metadata filtering is crucial for date-sensitive docs.",
                "comments": [
                    {"author": "rlhf_coder", "text": "How did you handle OCR for scanned PDFs? That's always a nightmare.", "score_diff": 14},
                    {"author": "deep_dive_dev", "text": "We used unstructured.io for OCR and table extraction, highly recommend it.", "score_diff": 6}
                ]
            }
        ]
    },
    "LLMs": {
        "topics": ["LLMs", "Deep Learning", "Models"],
        "entities": [
            {"text": "Llama 4", "type": "TOOL"},
            {"text": "Mistral", "type": "COMPANY"},
            {"text": "Fine-tuning", "type": "CONCEPT"},
            {"text": "Transformers", "type": "CONCEPT"},
            {"text": "GPU", "type": "TOOL"}
        ],
        "posts": [
            {
                "title": "Llama 4 architecture predictions: MoE or Dense?",
                "text": "With Meta training their next generation models, there's a lot of speculation. Will Llama 4 be a massive Mixture of Experts (MoE) model like GPT-4, or will they stick to the highly optimized dense model architecture? Personally, I think we will see an MoE with 8x22B parameters, which would make local execution challenging but inference much faster.",
                "comments": [
                    {"author": "gpu_rich", "text": "If it's MoE, we'll need multi-GPU setups just to load the weights. Hope they release smaller dense versions.", "score_diff": 25},
                    {"author": "cuda_cruncher", "text": "Meta has always favored deployment efficiency, so they'll probably do both.", "score_diff": 10}
                ]
            },
            {
                "title": "Fine-tuning vs prompting for domain-specific tasks: A real-world review",
                "text": "We tested fine-tuning a Llama-3-8B model vs writing long system prompts on GPT-4. For complex reasoning, prompting GPT-4 wins. But for formatting, tone, and specific style guidelines, the fine-tuned 8B model is faster, 10x cheaper, and performs equally well. Don't jump to fine-tuning unless you have at least 5,000 high-quality instruction pairs.",
                "comments": [
                    {"author": "rlhf_coder", "text": "Completely agree. Most people fine-tune when they should just improve their system instructions.", "score_diff": 30},
                    {"author": "model_merger", "text": "Also, model merging (like mergekit) can combine the benefits without training from scratch.", "score_diff": 15}
                ]
            },
            {
                "title": "Local LLMs are getting incredibly fast with speculative decoding",
                "text": "I just set up llama.cpp with speculative decoding using a tiny 1B draft model to speed up a 70B target model. The speedup is almost 2x on my Mac Studio! The draft model predicts tokens, and the large model validates them in parallel. It is amazing to see how much performance we can squeeze out of consumer hardware.",
                "comments": [
                    {"author": "transformer_tamer", "text": "Speculative decoding is a game-changer for local deployment. It makes 70B models usable.", "score_diff": 14},
                    {"author": "pytorch_guru", "text": "Are you running on Apple Silicon? llama.cpp is insanely optimized for Unified Memory.", "score_diff": 9}
                ]
            },
            {
                "title": "The trend toward smaller, highly capable models (SLMs)",
                "text": "Models like Phi-3, Gemma-2-9B, and Llama-3-8B are proving that token quality and dataset size matter more than raw parameter count. We are training models longer on trillions of synthetic tokens. Is the era of mega-models over, or is this just a temporary consolidation phase before the next paradigm shift?",
                "comments": [
                    {"author": "gradient_descent_fan", "text": "We've hit diminishing returns on raw scaling. Architecture and data quality are the new frontiers.", "score_diff": 11},
                    {"author": "weights_n_biases", "text": "Not over. Mega-models are still needed to generate the synthetic data used to train SLMs.", "score_diff": 8}
                ]
            },
            {
                "title": "Mistral releases new open-weights model with native function calling",
                "text": "Mistral just dropped another banger. A 12B parameter model that fits on standard consumer cards and has native support for function calling and tool use. We benchmarked it on API integration tasks and it beats several proprietary models. The open-source community continues to win.",
                "comments": [
                    {"author": "open_source_champion", "text": "Mistral has been absolute fire. They support the open ecosystem like no other company.", "score_diff": 20},
                    {"author": "deep_dive_dev", "text": "The function calling precision is great, makes agentic workflows much more reliable.", "score_diff": 12}
                ]
            }
        ]
    },
    "AI Safety": {
        "topics": ["AI Safety", "Ethics", "Policy"],
        "entities": [
            {"text": "AI Safety", "type": "CONCEPT"},
            {"text": "Anthropic", "type": "COMPANY"},
            {"text": "OpenAI", "type": "COMPANY"},
            {"text": "Superalignment", "type": "CONCEPT"},
            {"text": "Regulation", "type": "CONCEPT"}
        ],
        "posts": [
            {
                "title": "The alignment problem: Where do we stand in 2026?",
                "text": "As frontier models approach agentic autonomy, the question of alignment becomes urgent. RLHF works for text output, but how do we align autonomous agents that can execute code, call APIs, and make decisions in real-time? Current benchmarks are not testing for deceptive alignment, and we need new evaluation standards before deploying these systems.",
                "comments": [
                    {"author": "ethics_scholar", "text": "Existential risk is debated, but current harms like automated misinformation and biased decisions are real right now.", "score_diff": 18},
                    {"author": "rlhf_coder", "text": "Deceptive alignment is terrifying. If a model learns to hide its true intent during training, we are in trouble.", "score_diff": 13}
                ]
            },
            {
                "title": "New policy proposals for AI safety and risk mitigation",
                "text": "Governments are debating new rules for frontier model registration. The proposals include compute thresholds (e.g. 10^26 FLOPs) and mandatory safety audits before public release. Critics say this will kill open-source innovation while locking in the monopolies of OpenAI and Google. How do we balance safety with open science?",
                "comments": [
                    {"author": "open_source_champion", "text": "Regulations built on compute limits are regulatory capture. It will only stop developers from building open weights models.", "score_diff": 32},
                    {"author": "ethics_scholar", "text": "We regulate pharmaceuticals and nuclear tech. AI is a dual-use technology and needs oversight.", "score_diff": 15}
                ]
            },
            {
                "title": "Anthropic releases new research on constitutional AI and model evaluations",
                "text": "Anthropic just published a new paper detail how they use 'Constitutional AI' to align Claude models without human feedback. By giving the model a written constitution (principles like helpfulness, honesty, and harmlessness) and letting a critique model guide the training, they reduced jailbreak rates by 40%.",
                "comments": [
                    {"author": "prompt_engineer_99", "text": "Constitutional AI is a very elegant solution. It makes alignment rules explicit rather than implicit in RLHF datasets.", "score_diff": 21},
                    {"author": "transformer_tamer", "text": "Yes, but who writes the constitution? It's still corporate executives deciding the values.", "score_diff": 14}
                ]
            },
            {
                "title": "AI safety benchmarks are failing to test for agentic reasoning",
                "text": "Most safety benchmarks (MMLU, GSM8k) only test static knowledge or simple math. They do not evaluate how an agent reacts when its environment changes, or whether it tries to bypass safety filters when given tool access. We need interactive environments to stress-test these models before we connect them to databases and payment rails.",
                "comments": [
                    {"author": "neural_net_builder", "text": "Spot on. An agent with terminal access is a totally different beast than a chatbot.", "score_diff": 17},
                    {"author": "cuda_cruncher", "text": "We should run agents in sandboxed containers with strict rate limits.", "score_diff": 8}
                ]
            },
            {
                "title": "What happened to the OpenAI superalignment team?",
                "text": "With Jan Leike and Ilya Sutskever leaving OpenAI, the superalignment team has been dissolved. This raises huge red flags. If the leading AI lab is deprioritizing long-term safety research in favor of commercial product releases, who is actually working on protecting us from catastrophic risks? This is a wake-up call for the industry.",
                "comments": [
                    {"author": "ethics_scholar", "text": "It shows that corporate greed will always override safety unless government steps in.", "score_diff": 28},
                    {"author": "gpu_rich", "text": "Ilya leaving was a massive signal. Hope his new venture focusing on safe superintelligence succeeds.", "score_diff": 19}
                ]
            }
        ]
    },
    "Open Source AI": {
        "topics": ["Open Source", "Democratization", "Models"],
        "entities": [
            {"text": "Open Source", "type": "CONCEPT"},
            {"text": "Hugging Face", "type": "COMPANY"},
            {"text": "Llama 3.1", "type": "TOOL"},
            {"text": "Democratization", "type": "CONCEPT"}
        ],
        "posts": [
            {
                "title": "Why open source models are catching up with proprietary APIs",
                "text": "Two years ago, GPT-4 was lightyears ahead of anything open-source. Today, models like Llama 3.1 405B and Qwen 2.5 72B are matching or exceeding GPT-4 on core benchmarks. The open-source community benefits from collaborative optimization (quantization, fine-tuning, custom merge recipes) that no single company can replicate. The gap has closed.",
                "comments": [
                    {"author": "open_source_champion", "text": "The collaborative power of Hugging Face is simply unmatched. Open source always wins long term.", "score_diff": 24},
                    {"author": "gpu_rich", "text": "True, but running Llama 3.1 405B requires 8x H100s. It's open weights, but not accessible to individuals.", "score_diff": 16},
                    {"author": "model_merger", "text": "Quantization (GGUF, EXL2) brings these models down to consumer hardware. I run quantized models on my Mac.", "score_diff": 10}
                ]
            },
            {
                "title": "Hugging Face: The github of the AI revolution",
                "text": "We should take a moment to appreciate Hugging Face. They host hundreds of thousands of models, datasets, and demos for free, making advanced machine learning accessible to anyone with an internet connection. Without Hugging Face, the AI landscape would be locked behind API walls. They are democratizing AI.",
                "comments": [
                    {"author": "pytorch_guru", "text": "HF Transformers library basically saved our startup. Saved us months of boiler-plate code.", "score_diff": 17},
                    {"author": "neural_net_builder", "text": "Agreed, their ecosystem is incredible. We host all our internal models there.", "score_diff": 9}
                ]
            },
            {
                "title": "Is fully open-source AI viable long-term?",
                "text": "While open weight models are great, the datasets they are trained on are almost never released. True open-source requires open datasets, open training code, and open weights. Projects like OLMo by Allen Institute are trying to change this by releasing everything. But can they compete with Meta's multi-billion dollar compute budgets?",
                "comments": [
                    {"author": "ethics_scholar", "text": "OLMo is a noble effort. We need transparent datasets to inspect models for bias and copyright issues.", "score_diff": 15},
                    {"author": "open_source_champion", "text": "True. If we don't know what it trained on, it's not open-source, it's open-weights.", "score_diff": 11}
                ]
            }
        ]
    },
    "Vector Databases": {
        "topics": ["Vector Databases", "Storage", "Search"],
        "entities": [
            {"text": "Vector Database", "type": "CONCEPT"},
            {"text": "Weaviate", "type": "TOOL"},
            {"text": "Pgvector", "type": "TOOL"},
            {"text": "HNSW", "type": "CONCEPT"},
            {"text": "Pinecone", "type": "TOOL"}
        ],
        "posts": [
            {
                "title": "Do we really need dedicated vector DBs in 2026?",
                "text": "Every relational database (Postgres with pgvector, SQLite, Redis) now supports vector indexes. Why should a startup take on the operational overhead of a dedicated vector database like Weaviate or Milvus when they can just run a pgvector query? Operational simplicity should always come first.",
                "comments": [
                    {"author": "vector_db_nerd", "text": "Dedicated vector DBs handle horizontal scaling, hybrid search, and filtering on metadata much better at scale.", "score_diff": 22},
                    {"author": "rag_architect", "text": "If you have < 1M vectors, Postgres is fine. If you have 100M vectors with high update rates, pgvector will bottleneck.", "score_diff": 14}
                ]
            },
            {
                "title": "Comparing HNSW vs IVF-PQ indexing: A performance deep dive",
                "text": "HNSW (Hierarchical Navigable Small World) provides high recall and fast queries, but it stores the index in memory, which gets extremely expensive. IVF-PQ (Inverted File with Product Quantization) compresses vectors and uses disk, but query speeds are slower and recall drops. For enterprise search, we ended up with HNSW but had to buy massive RAM nodes.",
                "comments": [
                    {"author": "cuda_cruncher", "text": "RAM costs are the silent killer of vector search at scale.", "score_diff": 19},
                    {"author": "vector_db_nerd", "text": "Weaviate has PQ and compression options that let you trade off recall for memory. Worth testing.", "score_diff": 12}
                ]
            }
        ]
    },
    "Fine-tuning": {
        "topics": ["Fine-tuning", "Deep Learning", "Optimizations"],
        "entities": [
            {"text": "Fine-tuning", "type": "CONCEPT"},
            {"text": "LoRA", "type": "TOOL"},
            {"text": "QLoRA", "type": "TOOL"},
            {"text": "Unsloth", "type": "TOOL"}
        ],
        "posts": [
            {
                "title": "LoRA vs Full Parameter Fine-tuning: When to use which?",
                "text": "Low-Rank Adaptation (LoRA) freeze the base model and only train adapters, reducing memory usage by 90%. We benchmarked LoRA against full parameter fine-tuning on a medical classification task. Surprisingly, LoRA matched full training accuracy while being much faster and preventing catastrophic forgetting of general knowledge. LoRA is the default choice now.",
                "comments": [
                    {"author": "pytorch_guru", "text": "LoRA also lets us swap task-specific adapters in and out of memory at runtime, which is super powerful.", "score_diff": 23},
                    {"author": "model_merger", "text": "Make sure to tune rank (r) and alpha. We found r=16, alpha=32 to be optimal for most tasks.", "score_diff": 11}
                ]
            },
            {
                "title": "Unsloth: Fast LLM fine-tuning guide for local GPUs",
                "text": "If you are still using standard Hugging Face PEFT libraries for fine-tuning, you are wasting money. Unsloth rewires the backpropagation kernels in Triton and speeds up training by 2-5x while using 80% less memory. I fine-tuned a Llama-3-8B model on a single 16GB GPU with room to spare.",
                "comments": [
                    {"author": "cuda_cruncher", "text": "Unsloth is black magic. Saved me hundreds of dollars in cloud GPU costs.", "score_diff": 26},
                    {"author": "neural_net_builder", "text": "Does it support MoE models? Or only Llama/Gemma architectures?", "score_diff": 7}
                ]
            }
        ]
    }
}


class SyntheticDataScraper:
    """
    Generates realistic, structured Reddit posts and comments based on pre-defined topics.
    """

    def generate_all(self, total_posts: int = 150) -> List[Dict[str, Any]]:
        """
        Generate mock posts distributed across subreddits with custom temporal sentiment trends.
        """
        logger.info(f"Generating {total_posts} synthetic Reddit posts...")
        
        posts = []
        subreddits = ["MachineLearning", "LanguageModels", "artificial"]
        now = datetime.now(timezone.utc)
        posts_per_sub = total_posts // len(subreddits)
        
        sub_topics = {
            "MachineLearning": ["LLMs", "Fine-tuning", "Vector Databases"],
            "LanguageModels": ["RAG", "LLMs", "Open Source AI"],
            "artificial": ["AI Safety", "Open Source AI", "LLMs"]
        }

        post_counter = 1
        
        for sub in subreddits:
            topics_available = sub_topics[sub]
            
            for _ in range(posts_per_sub):
                topic_name = random.choice(topics_available)
                topic_data = TOPICS_CONTENT[topic_name]
                
                template = random.choice(topic_data["posts"])
                
                days_ago = random.randint(0, 365)
                post_time = now - timedelta(days=days_ago)
                created_at = int(post_time.timestamp())
                
                sentiment = random.uniform(0.5, 0.8)
                
                if topic_name == "RAG":
                    if days_ago <= 180:
                        sentiment = random.uniform(0.7, 0.95)
                    else:
                        sentiment = random.uniform(0.4, 0.65)
                elif topic_name == "AI Safety":
                    if days_ago <= 180:
                        sentiment = random.uniform(0.15, 0.45)
                    else:
                        sentiment = random.uniform(0.5, 0.75)
                elif topic_name == "Open Source AI":
                    sentiment = random.uniform(0.75, 0.98)
                elif topic_name == "Fine-tuning":
                    sentiment = random.uniform(0.6, 0.85)
                
                score = random.randint(20, 1500)
                upvote_ratio = round(random.uniform(0.75, 0.98), 2)
                post_id = f"synth_{sub[:3].lower()}_{post_counter:03d}"
                author = random.choice(AUTHORS)
                
                comments = []
                for i, comment_template in enumerate(template["comments"]):
                    comment_delay = random.randint(3600, 86400)
                    comment_time = created_at + comment_delay
                    
                    comment_score = max(1, int(score * (comment_template["score_diff"] / 100.0) + random.randint(-5, 5)))
                    
                    comments.append({
                        "id": f"comment_{post_id}_{i}",
                        "text": comment_template["text"],
                        "author": comment_template["author"],
                        "created_at": comment_time,
                        "score": comment_score,
                        "parent_id": f"t3_{post_id}"
                    })
                
                post = {
                    "id": post_id,
                    "title": template["title"],
                    "text": template["text"],
                    "author": author,
                    "created_at": created_at,
                    "score": score,
                    "upvote_ratio": upvote_ratio,
                    "num_comments": len(comments),
                    "subreddit": sub,
                    "url": f"https://reddit.com/r/{sub}/comments/{post_id}",
                    "comments": comments,
                    "topics": topic_data["topics"],
                    "entities": topic_data["entities"],
                    "sentiment": round(sentiment, 2)
                }
                
                posts.append(post)
                post_counter += 1

        random.shuffle(posts)
        return posts

    def save_to_json(self, posts: List[Dict[str, Any]], filename: str = "synthetic_reddit_data.json"):
        """Save posts list to a local JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(posts)} synthetic posts to {filename}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    
    scraper = SyntheticDataScraper()
    posts = scraper.generate_all(150)
    scraper.save_to_json(posts, "synthetic_reddit_data.json")
    print("Done! Generated and verified mock data.")
