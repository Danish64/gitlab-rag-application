from langchain_community.document_loaders import DirectoryLoader, TextLoader

# load from /documents directory
loader = DirectoryLoader("documents",
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True
)

docs = loader.lazy_load()
doc = next(docs)

print("=" * 50)

print(f"Source: {doc.metadata.get('source')}")

print(f"Content Length: {len(doc.page_content)} chars")

print(f"Preview: {doc.page_content[:200]}")


# for doc in docs:
#     print("=" * 50)

#     print(f"Source: {doc.metadata.get('source')}")

#     print(f"Content Length: {len(doc.page_content)}")