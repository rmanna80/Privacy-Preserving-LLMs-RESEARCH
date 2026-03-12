# ai_core/smart_chunker.py

from langchain_core.documents import Document
from typing import List


def smart_chunk_tax_forms(docs: List[Document]) -> List[Document]:
    """
    Special chunking for tax forms that preserves form structure
    """
    chunks = []
    
    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get('source', '')
        
        # If it's a tax form, use line-aware chunking
        if '1040' in source or 'tax' in source.lower():
            # Split by lines but keep related info together
            lines = text.split('\n')
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line_length = len(line)
                
                # If adding this line exceeds chunk size, save current chunk
                if current_length + line_length > 1500 and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy()
                    ))
                    # Keep last few lines for overlap
                    current_chunk = current_chunk[-3:]
                    current_length = sum(len(l) for l in current_chunk)
                
                current_chunk.append(line)
                current_length += line_length
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(Document(
                    page_content='\n'.join(current_chunk),
                    metadata=doc.metadata.copy()
                ))
        else:
            # Regular chunking for other documents
            chunks.append(doc)
    
    return chunks