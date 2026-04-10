"""
Part 1: Document Ingestion & RAG Strategy with Agentic Knowledge Distillation
Requires: sentence-transformers
"""
import re
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

class KnowledgePyramid:
    def __init__(self, raw_text: str, summary: str, category: str, distilled: str):
        self.raw_text = raw_text
        self.summary = summary
        self.category = category
        self.distilled = distilled

class DocumentIngestionSystem:
    def __init__(self, page_char_limit: int = 1000):
        self.page_char_limit = page_char_limit
        self.pyramid_nodes: List[KnowledgePyramid] = []
        
        # Load embedding model if available, else fallback to naive string matching
        if SentenceTransformer is not None:
            print("Loading SentenceTransformer model for semantic search...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_fallback = False
        else:
            print("WARNING: sentence-transformers not found. Falling back to naive word-overlap retrieval.")
            self.use_fallback = True

        self.embeddings = []
        self.embeddings_data = []

    def _paginate_text(self, text: str) -> List[str]:
        """Split text into pages based on character limit."""
        words = text.split()
        pages = []
        current_page = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > self.page_char_limit and current_page:
                pages.append(" ".join(current_page))
                current_page = [word]
                current_len = len(word)
            else:
                current_page.append(word)
                current_len += len(word) + 1
        if current_page:
            pages.append(" ".join(current_page))
        return pages

    def _simulate_llm_summary(self, text: str) -> str:
        """Mock LLM function to generate a chunk summary."""
        # Just grab the first and last few words to simulate a summary
        words = text.split()
        if len(words) < 20:
            return "Summary: " + text
        return f"Summary: {words[0]} {words[1]} ... {words[-2]} {words[-1]}"

    def _simulate_llm_category(self, text: str) -> str:
        """Mock LLM function to assign a category/theme."""
        # Simple rule-based logic for mock purposes
        text_lower = text.lower()
        if any(w in text_lower for w in ["revenue", "profit", "loss", "financial", "dollars"]):
            return "Financial Overview"
        elif any(w in text_lower for w in ["ai", "model", "algorithm", "technology", "intelligence"]):
            return "Technology & AI"
        return "General Operations"

    def _simulate_llm_distillation(self, text: str) -> str:
        """Mock LLM function to extract distilled knowledge/keywords."""
        # Extract unique words >= 7 letters as mocked keywords
        words = re.findall(r'\b[a-zA-Z]{7,}\b', text.lower())
        keywords = list(set(words))[:5]
        return f"Keywords: {', '.join(keywords)}"

    def ingest_document(self, text: str):
        """Process document using a 2-page sliding window approach."""
        print("Ingesting document...")
        pages = self._paginate_text(text)
        
        # 2-page sliding window (e.g., page 0-1, page 1-2, etc.)
        for i in range(len(pages)):
            window_text = pages[i]
            if i + 1 < len(pages):
                window_text += "\n" + pages[i + 1]
            
            # Agentic extraction layers
            summary = self._simulate_llm_summary(window_text)
            category = self._simulate_llm_category(window_text)
            distilled = self._simulate_llm_distillation(window_text)
            
            node = KnowledgePyramid(
                raw_text=window_text,
                summary=summary,
                category=category,
                distilled=distilled
            )
            self.pyramid_nodes.append(node)
            print(f"Processed Window {i+1} - Category: {category}")

        # Compute embeddings for all nodes (layer by layer representations)
        self._compute_embeddings()

    def _construct_node_texts(self) -> List[tuple]:
        """Flattens the pyramid into searchable text units, keeping track of the source node."""
        searchable_units = []
        for idx, node in enumerate(self.pyramid_nodes):
            searchable_units.append((node.raw_text, idx, "Raw Text"))
            searchable_units.append((node.summary, idx, "Summary"))
            searchable_units.append((node.category, idx, "Category Focus"))
            searchable_units.append((node.distilled, idx, "Distilled Concepts"))
        return searchable_units

    def _compute_embeddings(self):
        """Precompute embeddings for all layers of the pyramid."""
        if self.use_fallback or not self.pyramid_nodes:
            self.embeddings = None
            return
            
        units = self._construct_node_texts()
        texts = [u[0] for u in units]
        self.embeddings_data = units
        self.embeddings = self.embedder.encode(texts, convert_to_tensor=True)

    def retrieve(self, query: str, top_k: int = 1) -> dict:
        """Retrieve most relevant response from any pyramid level."""
        if not self.pyramid_nodes:
            return {"error": "No documents indexed."}

        units = self._construct_node_texts()
        
        if self.use_fallback:
            # Naive word overlap
            query_words = set(query.lower().split())
            best_score = -1
            best_match = None
            for idx, (text, node_idx, level) in enumerate(units):
                text_words = set(text.lower().split())
                score = len(query_words.intersection(text_words))
                if score > best_score:
                    best_score = score
                    best_match = (text, self.pyramid_nodes[node_idx], level)
            
            if best_match:
                return {
                    "matched_text": best_match[0],
                    "matched_layer": best_match[2],
                    "full_node": vars(best_match[1]),
                    "score": best_score
                }
            return {"error": "No matches found."}
        else:
            # Semantic Similarity using SentenceTransformers
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
            
            results = []
            for hit in hits:
                corpus_id = hit['corpus_id']
                score = hit['score']
                text, node_idx, level = self.embeddings_data[corpus_id]
                results.append({
                    "matched_text": text,
                    "matched_layer": level,
                    "full_node": vars(self.pyramid_nodes[node_idx]),
                    "score": score
                })
            return results[0] if top_k == 1 else results

if __name__ == "__main__":
    # Sample Test
    sample_doc = (
        "The company reported a massive increase in revenue for Q3, reaching nearly "
        "2 billion dollars in profit. This financial success is largely attributed to "
        "their new AI technology products and an improved algorithm that optimized "
        "cloud operations. Moving forward, the strategy will focus heavily on technology "
        "and expanding the model parameters for the artificial intelligence division."
    )
    
    # Intentionally use a very small page char limit to force multiple sliding windows on short text
    system = DocumentIngestionSystem(page_char_limit=100)
    system.ingest_document(sample_doc)
    
    print("\n--- Testing Retrieval ---")
    query = "What is the company's financial status?"
    print(f"Query: '{query}'")
    result = system.retrieve(query)
    print("\nBest Match Found:")
    print(f"Matched Layer: {result['matched_layer']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Extracted Text: {result['matched_text']}")
    print("\nFull Output Node represented:")
    print(result["full_node"])
