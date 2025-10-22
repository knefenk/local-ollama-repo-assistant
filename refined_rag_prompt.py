"""
Refined RAG Prompt System with Test Queries
"""

# ----------------------
# REFINED SYSTEM PROMPT
# ----------------------

SYSTEM_PROMPT = """You are an enterprise order backlog analyst assistant.

Your role is to help users find and understand information about order records using the provided context.

INSTRUCTIONS:
1. Answer directly based ONLY on the context provided - never use external knowledge
2. Be specific with numbers, names, and statuses when available
3. If multiple records are relevant, summarize key patterns or list them clearly
4. For missing information, state "Not found in the provided records" rather than guessing
5. Use natural, professional language - avoid robotic responses
6. When comparing or analyzing, highlight actionable insights (e.g., delays, high-value orders)

RESPONSE STYLE:
✓ Concise and scannable (bullet points for multiple items)
✓ Include specific identifiers (order numbers, customers, planners)
✓ Highlight critical info (statuses, countries, values)
✗ Don't repeat the question back
✗ Don't add disclaimers unless truly uncertain"""


# ----------------------
# PROMPT TEMPLATE
# ----------------------

def build_prompt(context: str, query: str) -> str:
    """Build the complete RAG prompt."""
    return f"""{SYSTEM_PROMPT}

---
📦 CONTEXT (Order Records):
{context}

---
❓ USER QUERY:
{query}

---
💬 YOUR RESPONSE:"""


# ----------------------
# TEST QUERIES BY CATEGORY
# ----------------------

TEST_QUERIES = {
    "1_EXACT_LOOKUP": [
        # Tests exact order/customer matching (BM25 strength)
        "What's the status of order 4646245?",
        "Show me details for order 4646245",
        "Find orders for customer Arrow Enterprise Computing Solutions",
        "Which planner handles order 4646245?",
    ],
    
    "2_SEMANTIC_SEARCH": [
        # Tests meaning-based retrieval (vector strength)
        "Show me urgent or high-priority orders",
        "Which orders are delayed or behind schedule?",
        "Find orders with shipping problems",
        "Show me orders that might miss their delivery date",
        "What orders have no scheduled ship date?",
    ],
    
    "3_FILTERING": [
        # Tests filtering by attributes
        "Show me all orders going to the US",
        "Which orders are shipping to Germany?",
        "Find orders in the BOOKED status",
        "Show me orders handled by planner JAYME MARIO",
        "What orders are from the Electronics division?",
    ],
    
    "4_AGGREGATION": [
        # Tests summarization and patterns
        "How many orders are going to the US?",
        "Which countries have the most orders?",
        "What are the most common order statuses?",
        "Which planners have the most orders?",
        "Summarize orders by status",
    ],
    
    "5_ANALYTICAL": [
        # Tests reasoning and insights
        "Which orders should I prioritize today?",
        "Are there any problematic orders I should know about?",
        "Show me high-value orders over $50,000",
        "Which orders have missing information?",
        "Find orders that are unassigned (no planner)",
    ],
    
    "6_COMPARISON": [
        # Tests multi-record reasoning
        "Compare orders 4646245 and 4494900",
        "What's the difference between BOOKED and ENTERED status?",
        "Show me both US and German orders",
        "Which planner has more orders: JAYME or ALEX?",
    ],
    
    "7_EDGE_CASES": [
        # Tests error handling
        "Show me order 9999999999",  # Non-existent
        "What's the weather in Germany?",  # Out of scope
        "Who is the CEO?",  # Unrelated
        "Tell me about blockchain",  # Completely unrelated
        "",  # Empty query
    ],
}


# ----------------------
# EXPECTED BEHAVIOR
# ----------------------

EXPECTED_RESPONSES = {
    "1_EXACT_LOOKUP": """
    ✅ Should return specific order details
    Example: "Order 4646245 is BOOKED, shipping to the US for customer 
    Arrow Enterprise Computing Solutions. It contains 3 line items..."
    """,
    
    "2_SEMANTIC_SEARCH": """
    ✅ Should find orders by meaning, not exact keywords
    Example: "delayed" → finds orders with missing ship dates or past-due
    """,
    
    "3_FILTERING": """
    ✅ Should list matching orders clearly
    Example: "Found 3 orders shipping to Germany:
    - Order 123: Customer ABC, Status BOOKED
    - Order 456: Customer XYZ, Status SHIPPED..."
    """,
    
    "4_AGGREGATION": """
    ✅ Should summarize patterns
    Example: "Based on the records shown:
    - US: 5 orders
    - Germany: 2 orders
    - Japan: 1 order"
    """,
    
    "5_ANALYTICAL": """
    ✅ Should provide actionable insights
    Example: "Priority orders to review:
    1. Order 123 - High value ($50k), no ship date
    2. Order 456 - Unassigned planner..."
    """,
    
    "6_COMPARISON": """
    ✅ Should highlight differences
    Example: "Both orders are for Arrow Enterprise:
    - 4646245: BOOKED, 3 items, $1,072
    - 4494900: ENTERED, 1 item, $5,000
    Key difference: 4646245 is further along..."
    """,
    
    "7_EDGE_CASES": """
    ✅ Should gracefully handle
    Example: "Order 9999999999 was not found in the provided records."
    Example: "I can only answer questions about order backlog records."
    """,
}


# ----------------------
# TESTING SCRIPT
# ----------------------

def run_test_suite(hybrid_search_fn, generate_answer_fn):
    """
    Run all test queries through your RAG pipeline.
    
    Args:
        hybrid_search_fn: Your hybrid_search(query, client, embeddings, bm25, k=6)
        generate_answer_fn: Your generate_answer(prompt)
    """
    print("=" * 80)
    print("🧪 RAG PIPELINE TEST SUITE")
    print("=" * 80)
    
    all_results = {}
    
    for category, queries in TEST_QUERIES.items():
        print(f"\n{'=' * 80}")
        print(f"📋 {category.replace('_', ' ')}")
        print(f"{'=' * 80}")
        
        category_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            print("-" * 80)
            
            # Run hybrid search
            hits = hybrid_search_fn(query)
            
            if not hits:
                print("❌ No results found")
                category_results.append({
                    "query": query,
                    "hits": 0,
                    "answer": "No results"
                })
                continue
            
            # Build context
            context = "\n\n".join([
                f"Order {h.metadata.get('order')}: "
                f"Customer {h.metadata.get('customer')}, "
                f"Status {h.metadata.get('status')}, "
                f"Planner {h.metadata.get('planner')}, "
                f"Ships to {h.metadata.get('ship_city')}, {h.metadata.get('ship_country')}"
                for h in hits[:3]  # Top 3 for testing
            ])
            
            # Generate answer
            prompt = build_prompt(context, query)
            answer = generate_answer_fn(prompt)
            
            print(f"📊 Found {len(hits)} relevant records")
            print(f"\n💬 ANSWER:\n{answer}\n")
            
            category_results.append({
                "query": query,
                "hits": len(hits),
                "answer": answer
            })
        
        all_results[category] = category_results
        
        # Show expected behavior
        if category in EXPECTED_RESPONSES:
            print(f"\n💡 EXPECTED BEHAVIOR:{EXPECTED_RESPONSES[category]}")
    
    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETE")
    print("=" * 80)
    
    return all_results