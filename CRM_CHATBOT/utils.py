import yaml
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional

def get_best_examples_or_all(metadata_path: str, question: str, k: int = 3, threshold: float = 0.5) -> List[str]:
    """
    Get the most similar query examples or return all if none are relevant enough.
    
    Args:
        metadata_path: Path to the metadata YAML file containing query examples
        question: User's natural language question
        k: Number of examples to consider
        threshold: Distance threshold to consider a match relevant
        
    Returns:
        List of relevant examples (description + SQL). If none meet threshold, return all examples.
    """
    with open(metadata_path, "r", encoding='utf-8') as f:
        metadata = yaml.safe_load(f)

    # Verifica se 'query_examples' existe
    if "table_config" not in metadata or "query_examples" not in metadata["table_config"]:
        raise ValueError("O arquivo YAML não contém 'table_config.query_examples'.")

    raw_examples = metadata["table_config"]["query_examples"]

    # Filtra apenas exemplos válidos
    valid_examples = [
        ex for ex in raw_examples
        if isinstance(ex, dict) and 'description' in ex and 'sql' in ex
    ]

    if not valid_examples:
        raise ValueError("Nenhum exemplo válido encontrado com os campos 'description' e 'sql'.")

    documents = [
        Document(page_content=f"{ex['description']}\n{ex['sql']}")
        for ex in valid_examples
    ]

    embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')
    vectorstore = FAISS.from_documents(documents, embedding_model)

    results = vectorstore.similarity_search_with_score(question, k=k)
    relevant = [doc.page_content for doc, score in results if score < threshold]

    # Se nenhum exemplo relevante for encontrado, retorna todos
    return relevant if relevant else [doc.page_content for doc in documents]
