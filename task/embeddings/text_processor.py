from enum import StrEnum
import re
from tokenize import cookie_re

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )
    

    def process_text_file(self,file_name: str, chunk_size:int, overlap: int, dimensions: int, 
                          truncate_table: bool = True):
        """Processing text file and saving the embeddings"""

        with open(file_name, "r", encoding='utf-8') as file:
            bytes = file.read()

        chunks = chunk_text(bytes, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embeddings_client.get_embeddings(inputs=chunks, dimensions=dimensions)

        if truncate_table:
            self._truncate_table()

        print(f"Processing document: {file_name}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embeddings)}")

        for i in range(len(chunks)):
            self._save_chunk(file_name=file_name, chunk=chunks[i], embedding=embeddings[i])

    def _truncate_table(self):
        """Truncating the embeddings table"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors")
            conn.commit()
        
    def _save_chunk(self, file_name: str, chunk: str, embedding: list[str]):
        """Save chunk with embedding to database"""
        embedding_string = f"[{','.join(map(str, embedding))}]"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO vectors(document_name,text,embedding) VALUES(%s, %s, %s::vector)", 
                            (file_name, chunk, embedding_string))

    def search(self, input: str, search_mode: SearchMode, top_k: float, score_threshold: float,
               dimensions: int):
        """Searching for relevant content"""
        input_embeddings = self.embeddings_client.get_embeddings(
            inputs=input, 
            dimensions=dimensions
        )[0]

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        vector_string = f"[{','.join(map(str, input_embeddings))}]"

        found_context = []
        with self._get_connection() as conn: 
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._get_search_query(search_mode), 
                            (vector_string, vector_string, max_distance, top_k)) 
                results = cur.fetchall()
                for row in results:
                    found_context.append(row['text'])

        return found_context
    
    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
                FROM vectors 
                WHERE embedding {mode} %s::vector <= %s
                ORDER BY distance
                LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')

    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

