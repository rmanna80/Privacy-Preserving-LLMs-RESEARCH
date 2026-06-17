from ai_core.family_qa import FamilyQASystem

sys = FamilyQASystem(family_id=2, verbose=True)
sys.index_documents(force_rebuild=False)

print(f"\nTotal chunks in index: {len(sys.chunks)}")
print("Documents represented in the chunks:")
sources = {}
for c in sys.chunks:
    src = c.metadata.get("source", "?")
    sources[src] = sources.get(src, 0) + 1
for src, count in sources.items():
    print(f"  {src}: {count} chunks")
