import chainlit as cl
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

from langchain.schema.runnable.config import RunnableConfig

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
import tiktoken

load_dotenv()

# Set openai chat model and embedding model
openai_chat_model = ChatOpenAI(model="gpt-4o")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Load PDF and chunk it in order to create our embeddings
docs = PyMuPDFLoader("data/airbnb-10k-filings.pdf").load()


def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

max_chunk_length = 0

for chunk in split_chunks:
    max_chunk_length = max(max_chunk_length, tiktoken_len(chunk.page_content))


# Use embedding model to store chunks into qdrant vector store and create a retriever

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="Airbnb 10K",
)

qdrant_retriever = qdrant_vectorstore.as_retriever()
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant_retriever, llm=openai_chat_model
)


# RAG Prompt

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
Only answer the question with the provided context. If the question is irrelevant to the context, respond with I don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message.
    """
    rename_dict = {"Assistant": "Airbnb 10k Filings Bot"}
    return rename_dict.get(original_author, original_author)


@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session.

    We will build our LCEL RAG chain here, and store it in the user session.

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = (
        {
            "context": itemgetter("query") | advanced_retriever,
            "query": itemgetter("query"),
        }
        | rag_prompt
        | openai_chat_model
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)


@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        if isinstance(chunk, dict) and "content" in chunk:
            await msg.stream_token(chunk["content"])
        elif hasattr(chunk, "content"):
            await msg.stream_token(chunk.content)
        elif isinstance(chunk, str):
            await msg.stream_token(chunk)

    await msg.send()
