"""Acme Support Agent — okf-agents example.

Demonstrates building a support Q&A agent from a directory of linked
Markdown files using okf-agents. No API keys required — the basic
tools, retriever, and router work entirely offline.

Usage:
    cd examples/support-agent
    pip install okf-agents
    python demo.py
"""

from pathlib import Path

from okf_agents import (
    OKFBundle,
    OKFRetriever,
    create_okf_router,
    create_okf_tools,
)

BUNDLE_DIR = Path(__file__).parent / "knowledge-base"


def main() -> None:
    bundle = OKFBundle.load(BUNDLE_DIR)
    print(f"Loaded {bundle.concept_count} concepts from {BUNDLE_DIR.name}/\n")

    # --- 1. Agent tools (deterministic, no model) ---
    tools = create_okf_tools(bundle)
    print("=== Agent Tools ===")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}")

    print("\n--- read_index ---")
    read_index = tools[3]
    print(read_index.invoke({}))

    print("\n--- search_concepts('refund') ---")
    search = tools[1]
    print(search.invoke({"query": "refund"}))

    print("\n--- read_concept('concepts/refunds') ---")
    read = tools[0]
    print(read.invoke({"concept_id": "concepts/refunds"}))

    print("\n--- list_links('concepts/billing') ---")
    links = tools[2]
    print(links.invoke({"concept_id": "concepts/billing"}))

    # --- 2. Keyword retriever ---
    print("\n=== Keyword Retriever ===")
    retriever = OKFRetriever(bundle=bundle, top_k=3)
    docs = retriever.invoke("payment failed")
    for doc in docs:
        cid = doc.metadata["concept_id"]
        title = doc.metadata["title"]
        print(f"  {cid} — {title} ({len(doc.page_content)} chars)")

    # --- 3. Router ---
    print("\n=== Router ===")
    router = create_okf_router(bundle)
    for query in ["Billing", "how do I get a refund?", "why is the sky blue?"]:
        result = router({"query": query})
        print(f'  "{query}" → route={result["route"]}')

    # --- 4. Graph traversal ---
    print("\n=== Link Graph ===")
    concept = bundle.get("concepts/billing")
    print(f"  {concept.id} links to:")
    for edge in bundle.links_from(concept.id):
        status = "resolved" if edge.resolved else "broken"
        print(f"    → {edge.target_id} ({status})")

    neighbors = bundle.neighbors("concepts/billing", hops=2, direction="out")
    print(f"  2-hop outbound neighbors: {[n.id for n in neighbors]}")

    print("\nDone. To use the navigator with an LLM, see the README.")


if __name__ == "__main__":
    main()
