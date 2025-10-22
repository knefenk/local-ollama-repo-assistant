"""
Hybrid Search Pipeline for Backlog Orders
Works with existing Qdrant index from convert.py
Builds BM25 index on startup for hybrid search
"""

import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# ----------------------
# CONFIG
# ----------------------
JSON_FILE = "./target/data.json"
VECTOR_STORE_DIR = "./data/qdrant"
COLLECTION_NAME = "backlog_orders"
EMBED_MODEL = "intfloat/e5-large-v2"  # Must match convert.py
OLLAMA_LLM = "codellama:latest"

BATCH_SIZE = 256  # For query embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------
# DATA LOADING (for BM25 only)
# ----------------------
def record_to_text(rec: dict) -> str:
    """Convert structured record into semantically rich text."""
    return (
        f"Order {rec.get('ORACLE_ORDER')} ({rec.get('ORDER_TYPE')}) "
        f"for customer {rec.get('CUSTOMER')} in division {rec.get('DIVISION')} "
        f"handled by planner {rec.get('PLANNER_NAME')}. "
        f"Item {rec.get('ITEM')} of type {rec.get('ITEM_TYPE')} "
        f"is {rec.get('FLOW_STATUS')} and scheduled to ship from {rec.get('SHIP_FR_ORG')} "
        f"to {rec.get('SHIP_TO_CITY')} ({rec.get('SHIP_TO_COUNTRY')}) "
        f"on {rec.get('SCHEDULE_SHIP_DATE')}. "
        f"Quantity {rec.get('ORDERED_QUANTITY')}, total value ${rec.get('TOTAL_USD')} USD."
    )


def load_json_for_bm25(json_path: str):
    """Load JSON records for BM25 index (lightweight, no embeddings)."""
    logging.info(f"Loading records for BM25 from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = list(data.values())[0]
    
    docs = []
    for rec in tqdm(records, desc="Parsing records"):
        docs.append(Document(
            page_content=record_to_text(rec),
            metadata={
                "order": rec.get("ORACLE_ORDER"),
                "division": rec.get("DIVISION"),
                "planner": rec.get("PLANNER_NAME"),
                "ship_country": rec.get("SHIP_TO_COUNTRY"),
                "ship_city": rec.get("SHIP_TO_CITY"),
                "status": rec.get("FLOW_STATUS"),
                "source_file": rec.get("SOURCE_FILENAME"),
                "customer": rec.get("CUSTOMER"),
                "item": rec.get("ITEM"),
            }
        ))
    logging.info(f"✅ Loaded {len(docs):,} records for BM25")
    return docs


# ----------------------
# INITIALIZE COMPONENTS
# ----------------------
def setup_qdrant_client():
    """Connect to existing Qdrant index."""
    client = QdrantClient(path=VECTOR_STORE_DIR, prefer_grpc=True)
    
    if not client.collection_exists(COLLECTION_NAME):
        raise FileNotFoundError(
            f"❌ Collection '{COLLECTION_NAME}' not found in {VECTOR_STORE_DIR}\n"
            f"Please run convert.py first to build the vector index."
        )
    
    info = client.get_collection(COLLECTION_NAME)
    logging.info(f"✅ Connected to Qdrant collection: {COLLECTION_NAME}")
    logging.info(f"   Vectors: {info.points_count:,}")
    return client


def setup_embeddings():
    """Initialize embedding model for query encoding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"🧠 Loading embeddings model on {device}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": True},
    )
    logging.info("✅ Embeddings model ready")
    return embeddings


def build_bm25(docs):
    """Build BM25 keyword index for exact matching."""
    logging.info("Building BM25 index...")
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 6  # Top-K results
    logging.info("✅ BM25 index ready")
    return bm25


# ----------------------
# HYBRID SEARCH (with scroll for comprehensive retrieval)
# ----------------------
def hybrid_search(query: str, client, embeddings, bm25, k=6, use_scroll=False, scroll_threshold=0.7):
    """
    Combine semantic (Qdrant) + keyword (BM25) search.
    
    Args:
        query: Search query
        client: Qdrant client
        embeddings: Embedding model
        bm25: BM25 retriever
        k: Number of results to return
        use_scroll: If True, uses scroll to get ALL semantically similar records
        scroll_threshold: Similarity score threshold for scroll (0-1)
    
    Returns:
        List of Document objects with metadata
    """
    # Semantic search via Qdrant
    query_vector = embeddings.embed_query(query)
    
    if use_scroll:
        # Use scroll to get ALL records above similarity threshold
        semantic_results = []
        offset = None
        
        while True:
            batch, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,  # No filter, we'll filter by score
                limit=100,  # Batch size
                offset=offset,
                with_vectors=False,
                with_payload=True
            )
            
            if not batch:
                break
            
            # Score each record against query
            for point in batch:
                # Get vector for this point
                point_data = client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[point.id],
                    with_vectors=True
                )[0]
                
                # Calculate similarity (cosine similarity for normalized vectors)
                import numpy as np
                similarity = np.dot(query_vector, point_data.vector)
                
                if similarity >= scroll_threshold:
                    semantic_results.append({
                        'payload': point.payload,
                        'score': similarity
                    })
            
            if offset is None:
                break
        
        # Sort by similarity score
        semantic_results.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"📊 Scroll found {len(semantic_results)} records above {scroll_threshold} similarity")
    else:
        # Standard search (top-k only)
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=k * 3  # Get more for better hybrid merging
        )
        semantic_results = [{'payload': hit.payload, 'score': hit.score} for hit in search_results]
    
    # Convert Qdrant results to Documents
    semantic_docs = []
    for hit in semantic_results:
        semantic_docs.append(Document(
            page_content=f"Order {hit['payload'].get('order')} - {hit['payload'].get('status')}",
            metadata={**hit['payload'], 'search_score': hit['score']}
        ))
    
    # Keyword search via BM25
    bm25_docs = bm25._get_relevant_documents(
        query,
        run_manager=CallbackManagerForRetrieverRun.get_noop_manager()
    )
    
    # Merge and deduplicate by order ID
    seen = set()
    merged = []
    
    # Prioritize semantic results (usually more relevant)
    for d in semantic_docs:
        key = d.metadata.get("order")
        if key and key not in seen:
            seen.add(key)
            merged.append(d)
    
    # Add unique BM25 results
    for d in bm25_docs:
        key = d.metadata.get("order")
        if key and key not in seen:
            seen.add(key)
            merged.append(d)
    
    # Return top-k unless using scroll mode
    if not use_scroll:
        merged = merged[:k]
    
    return merged


def smart_search(query: str, client, embeddings, bm25, k=6):
    """
    Intelligently decide whether to use scroll or standard search.
    
    Uses scroll for:
    - Aggregation queries (count, summarize, list all)
    - Broad filters (all US orders, all BOOKED orders)
    
    Uses standard search for:
    - Specific lookups (order 12345)
    - Analytical questions (problematic orders)
    """
    query_lower = query.lower()
    
    # Keywords that suggest comprehensive retrieval
    scroll_keywords = [
        'all', 'every', 'total', 'count', 'how many', 
        'summarize', 'list', 'show me all', 'find all'
    ]
    
    use_scroll = any(keyword in query_lower for keyword in scroll_keywords)
    
    if use_scroll:
        logging.info(f"🔄 Using scroll mode for comprehensive search")
        return hybrid_search(query, client, embeddings, bm25, k=k, use_scroll=True, scroll_threshold=0.65)
    else:
        logging.info(f"⚡ Using standard top-k search")
        return hybrid_search(query, client, embeddings, bm25, k=k, use_scroll=False)


# ----------------------
# LLM GENERATION WITH HALLUCINATION DETECTION
# ----------------------
def generate_answer(prompt: str, model_name=OLLAMA_LLM):
    """Generate answer using Ollama LLM."""
    llm = OllamaLLM(model=model_name, temperature=0.0)  # Zero temperature = more deterministic
    return llm.invoke(prompt)


def validate_answer(answer: str, hits: list) -> tuple[str, list]:
    """
    Validate LLM answer against source data to detect hallucinations.
    
    Returns:
        (validated_answer, warnings)
    """
    warnings = []
    
    # Extract all actual values from hits
    actual_values = {
        'orders': set(),
        'customers': set(),
        'statuses': set(),
        'planners': set(),
        'countries': set(),
        'cities': set(),
        'items': set(),
    }
    
    for h in hits:
        if h.metadata.get('order'):
            actual_values['orders'].add(str(h.metadata.get('order')))
        if h.metadata.get('customer'):
            actual_values['customers'].add(h.metadata.get('customer'))
        if h.metadata.get('status'):
            actual_values['statuses'].add(h.metadata.get('status'))
        if h.metadata.get('planner') and h.metadata.get('planner') != 'N/A':
            actual_values['planners'].add(h.metadata.get('planner'))
        if h.metadata.get('ship_country') and h.metadata.get('ship_country') != 'N/A':
            actual_values['countries'].add(h.metadata.get('ship_country'))
        if h.metadata.get('ship_city') and h.metadata.get('ship_city') != 'N/A':
            actual_values['cities'].add(h.metadata.get('ship_city'))
        if h.metadata.get('item') and h.metadata.get('item') != 'N/A':
            actual_values['items'].add(h.metadata.get('item'))
    
    # Check for potential hallucinations (basic heuristics)
    answer_lower = answer.lower()
    
    # Check for suspicious patterns
    suspicious_phrases = [
        'approximately', 'around', 'about', 'roughly',  # Vague quantifiers
        'probably', 'likely', 'might be', 'could be',    # Speculation
        'seems to', 'appears to', 'looks like',          # Inference
        'i think', 'i believe', 'in my opinion',         # Subjectivity
    ]
    
    for phrase in suspicious_phrases:
        if phrase in answer_lower:
            warnings.append(f"⚠️  Response contains speculative language: '{phrase}'")
    
    # Verify counts are reasonable
    import re
    count_matches = re.findall(r'(\d+)\s+(?:orders?|records?|items?)', answer_lower)
    if count_matches:
        for count_str in count_matches:
            claimed_count = int(count_str)
            actual_count = len(hits)
            if claimed_count > actual_count:
                warnings.append(f"⚠️  Claimed count ({claimed_count}) exceeds retrieved records ({actual_count})")
    
    return answer, warnings


# ----------------------
# MAIN PIPELINE
# ----------------------
if __name__ == "__main__":
    print("🚀 Initializing Hybrid Search Pipeline...\n")
    
    # 1. Connect to existing Qdrant index
    try:
        client = setup_qdrant_client()
    except FileNotFoundError as e:
        print(str(e))
        exit(1)
    
    # 2. Load embeddings model (for queries)
    embeddings = setup_embeddings()
    
    # 3. Build BM25 index from JSON (fast, in-memory)
    docs = load_json_for_bm25(JSON_FILE)
    bm25 = build_bm25(docs)
    
    print("\n🎯 Hybrid Search Ready!")
    print(f"• Vector Search: Qdrant ({COLLECTION_NAME})")
    print(f"• Keyword Search: BM25")
    print(f"• LLM: {OLLAMA_LLM}")
    print("\nType your queries (exit/quit to stop):\n")
    
    try:
        while True:
            q = input("> ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            
            # Hybrid search
            hits = hybrid_search(q, client, embeddings, bm25, k=6)
            
            if not hits:
                print("❌ No results found.\n")
                continue
            
            # Build context for LLM
            context = "\n\n".join([
                f"Order {h.metadata.get('order')}: "
                f"Customer {h.metadata.get('customer')}, "
                f"Status {h.metadata.get('status')}, "
                f"Ships to {h.metadata.get('ship_city')}, {h.metadata.get('ship_country')}"
                for h in hits
            ])
            
            prompt = f"""Context (Backlog Orders):
{context}

Question:
{q}

Answer concisely based on the order data above:"""
            
            ans = generate_answer(prompt)
            
            print("\n--- ANSWER ---")
            print(ans)
            print("\n--- SOURCES ---")
            for i, h in enumerate(hits, 1):
                print(f"[{i}] Order: {h.metadata.get('order')} | "
                      f"Customer: {h.metadata.get('customer')} | "
                      f"Status: {h.metadata.get('status')} | "
                      f"Ship To: {h.metadata.get('ship_city')}, {h.metadata.get('ship_country')}")
            print()
    
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    finally:
        # Explicitly close Qdrant client to avoid Windows file lock warnings
        logging.info("Closing Qdrant client...")
        client.close()
        print("✅ Cleanup complete. Goodbye!")