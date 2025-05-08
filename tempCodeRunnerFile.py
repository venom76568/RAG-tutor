split_text = split_pages(text)
# chunks = chunk_pdf_text(split_text)
# with open("chunks.txt", "w", encoding="utf-8") as f:
#     for i, chunk in enumerate(chunks, 1):
#         f.write(f"Chunk {i} (Page {chunk['metadata']['page']}):\n")
#         f.write(chunk['text'] + "\n\n")

# print("✅ Chunks saved to chunks.txt")

# #Inspect the first few chunks with correct page numbers
# # for i, chunk in enumerate(chunks[:10], 1):
# #    print(f"Chunk {i} (Page {chunk['metadata']['page']}): {chunk['text'][:300]}...")

# build_vector_store(chunks, persist_path="vector_db")