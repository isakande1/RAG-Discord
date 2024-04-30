import discord
import os
import json
import requests
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_community.llms  import Ollama
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader as dL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader as loader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv("../.env")
token = os.getenv('token')
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
channelId = int(os.getenv('channelId'))
client =  discord.Client(intents=discord.Intents.default())

# def load_documents():
#     documentLoader = loader(r"C:\Users\ismas\Desktop\discord2.0\discord\books",)   
#     return documentLoader.load()

def load_documents():
    loader = dL(r"C:\Users\ismas\Desktop\discord2.0\discord\book", glob="*.md")
    #return the document
    return loader.load()

def chunkDocuments(documents):
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
     chunk_overlap=500, length_function=len, add_start_index=True,)
    chunks = textSplitter.split_documents(documents)  
    return chunks

documents = load_documents()
#chunks = chunkDocuments(documents)
#print(chunks)

def get_embedding_function():
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings

def add_to_chroma(chunks: list[Document]):
        db = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory="chroma")
        db.persist()
#add_to_chroma(chunks)



async def query_rag(message: str, history:list ):
    prompt = ChatPromptTemplate.from_messages(history )    
    llm= Ollama(model="llama2")
    chain = prompt | llm | StrOutputParser()
    return  chain.invoke({"input":message})


def get_quote():
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + "-" + json_data[0]['a']
    return quote

@client.event
async def on_ready():
    # for guild in client.guilds:
    #     print(f"Guild: {guild.name} (ID: {guild.id})")
    #     print("Channels:")
    #     for channel in guild.channels:
    #         print(f"  - {channel.name} (ID: {channel.id}, Type: {channel.type})")
    history = [("system","You are Onix, an Ai financial coach.You always provide well-reasoned answers that are both correct and helpful.")]
    channel = client.get_channel(channelId)
    await channel.send(await query_rag("introduce yourself",history))
    print('we have logged in as {0.user}'.format(client))
    

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith('$inspire'):
        quote = get_quote()
        await message.channel.send(quote)
    else:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory="chroma", embedding_function=embedding_function)
        results = db.similarity_search_with_score(message.content, k=1)
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        history = [
            ("system", f"You always provide well-reasoned answers that are both correct and helpful.Use this {context} as reference when answering to question, be clear and concise .")]
        await message.channel.send(await query_rag(message.content,history))

client.run(token)