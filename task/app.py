import select

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}
"""

class MicrowaveRAG:

    def __init__(self, embeddings: DialEmbeddingsClient):
        self.dial_embeddings = embeddings

#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)
def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")
    db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    }
    
    text_procesor = TextProcessor(rag.dial_embeddings, db_config=db_config)
    text_procesor.process_text_file("task/embeddings/microwave_manual.txt", 300, 40, 1536)

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_prompt = input("\n> ").strip()
        
        # Retrieval 
        context = text_procesor.search(
            input=user_prompt,
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            top_k=5,
            score_threshold=0.01,
            dimensions=1536,
        )

        # Augmentation
        augmented_prompt = USER_PROMPT.replace("{context}", "\n\n".join(context)).replace("{query}", user_prompt)
        print(f"Prompt:\n{augmented_prompt}")


        # Generation
        conversation.add_message(Message(Role.USER, augmented_prompt))


        chat_client = DialChatCompletionClient(deployment_name="gpt-4o", api_key=API_KEY)
        response = chat_client.get_completion(
            messages=conversation.get_messages()
        )
        conversation.add_message(response)

        print(f"AI Response: {response.content}")

main(
    MicrowaveRAG(
        embeddings=DialEmbeddingsClient(
            deployment_name="text-embedding-3-small-1",
            api_key=API_KEY
        )
    )    
)


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml