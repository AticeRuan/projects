from pydantic import BaseModel
import pymongo
# Import traceback for error handling
import traceback

# Import os and sys for system-related operations
import os, sys
import traceback  # Import traceback for error handling
from fastapi import (
    FastAPI,
    UploadFile,
    status,
    HTTPException,
)  # Import FastAPI components for building the web application
from fastapi.responses import JSONResponse  # Import JSONResponse for returning JSON responses
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware to handle Cross-Origin Resource Sharing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import S3FileLoader
from langchain_community.document_loaders import Docx2txtLoader,PyPDFLoader


from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain

from langchain_openai import ChatOpenAI
import gc

import urllib.parse
import awswrangler as wr  # Import AWS Wrangler for working with AWS services

import boto3  # Import the boto3 library for interacting with AWS services

# Import the os module for system-related operations

# Check if the operating system is Windows
if os.name == "nt":  # Windows
    # If it's Windows, import the `load_dotenv` function from the `dotenv` library
    from dotenv import load_dotenv

    # Load environment variables from a `.secrets.env` file (used for local development)
    load_dotenv(".secrets.env")



OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
S3_KEY=os.environ['S3_KEY']
S3_SECRET=os.environ['S3_SECRET']
S3_BUCKET=os.environ['S3_BUCKET']
S3_REGION=os.environ['S3_REGION']
S3_PATH=os.environ['S3_PATH']


try:
    MONGO_URL="mongodb+srv://rainryy:rainMongo@cluster0.jyupp.mongodb.net/?retryWrites=true&w=majority&ssl=true"

    # Connect to the MongoDB using the provided MONGO_URL
    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
    # Access the "chat_with_doc" database
    db = client["chat_with_doc"]
    # Access the "chat-history" collection within the database
    conversationcol = db["chat-history"]

    # Create an index on the "session_id" field, ensuring uniqueness
    conversationcol.create_index([("session_id")], unique=True)
except:
    # Handle exceptions and print detailed error information
    print(traceback.format_exc())

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # Print information about the exception type, filename, and line number
    print(exc_type, fname, exc_tb.tb_lineno)





# Import the necessary modules and libraries


class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

def get_response(
    file_name: str,
    session_id: str,
    query: str,
    model: str = "gpt-3.5-turbo-16k",
    temperature: float = 0,
):
    print("file name is ", file_name)
    file_name=file_name.split("/")[-1]
    
    embeddings = OpenAIEmbeddings()  # load embeddings
    # download file from s3
    wr.s3.download(path=f"s3://docchat/documents/{file_name}",local_file=file_name,boto3_session=aws_s3)

    # loader = S3FileLoader(
    #     bucket=S3_BUCKET,
    #     key=S3_PATH + file_name.split("/")[-1],
    #     aws_access_key_id=S3_KEY,
    #     aws_secret_access_key=S3_SECRET,
    # )
    if file_name.endswith(".docx"):
        loader=Docx2txtLoader(file_path=file_name.split("/")[-1])
    else:
        loader = PyPDFLoader(file_name)

    # 1.load data
    data = loader.load()
    # 2.split data so it can fit gpt token limit
    print("splitting ..")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=["\n", " ", ""]
    )

    all_splits = text_splitter.split_documents(data)
    # 3. store data in vector db to conduct searc
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    # 4. Init openai
    llm = ChatOpenAI(model_name=model, temperature=temperature)

    # 5. pass the data to openai chain using vector db
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
    )
    # use the function to determine tokens used
    with get_openai_callback() as cb:
        answer = qa_chain(
            {
                "question": query,  # user query
                "chat_history": load_memory_to_pass(
                    session_id=session_id
                ),  # pass chat history for context
            }
        )
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        answer["total_tokens_used"] = cb.total_tokens
    gc.collect()  # collect garbage from memory
    return answer
import uuid
from typing import List


def load_memory_to_pass(session_id: str):
    
    data = conversationcol.find_one(
        {"session_id": session_id}
    )  # find the document with the session id
    history = []  # create empty array (incase we do not have any history)
    if data:  # check if data is not None
        data = data["conversation"]  # get the conversation field

        for x in range(0, len(data), 2):  # iterate over the field
            history.extend(
                [(data[x], data[x + 1])]
            )  # our history is expected format is [(human_message,ai_message)] , the even index has human message and odd has ai response
    print(history)
    return history  # retun history


def get_session() -> str:
   
    return str(uuid.uuid4())


def add_session_history(session_id: str, new_values: List):
    
    document = conversationcol.find_one(
        {"session_id": session_id}
    )  # find the document with the session id
    if document:  # check if data is not None
        # Extract the conversation list
        conversation = document["conversation"]

        # Append new values
        conversation.extend(new_values)

        # Update the document with the modified conversation list (for old session), we use update_one
        conversationcol.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}}
        )
    else:
        conversationcol.insert_one(
            {
                "session_id": session_id,
                "conversation": new_values,
            }  # to initiate a history under a newsession, note we uses insert_one
        )


# Create a FastAPI application
app = FastAPI()

# Add CORS middleware to handle Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,  # Allow sending credentials (e.g., cookies)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Create an AWS S3 session with provided access credentials
aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,  # Set the AWS access key ID
    aws_secret_access_key=S3_SECRET,  # Set the AWS secret access key
    region_name="us-east-2",  # Set the AWS region
)


@app.post("/chat")
async def create_chat_message(
    chats: ChatMessageSent,
):
    
    try:
        if chats.session_id is None:
            session_id = get_session()

            payload = ChatMessageSent(
                session_id=session_id,
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.model_dump()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=session_id,
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={
                    "response": response,
                    "session_id": str(session_id),
                }
            )

        else:
            payload = ChatMessageSent(
                session_id=str(chats.session_id),
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.dict()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=str(chats.session_id),
                new_values=[payload.get("user_input"), response["answer"]],
            )

            return JSONResponse(
                content={
                    "response": response,
                    "session_id": str(chats.session_id),
                }
            )
    except Exception:
        print(traceback.format_exc())

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="error")


@app.post("/uploadFile")
async def uploadtos3(data_file: UploadFile):
    
    print(data_file.filename.split("/")[-1])
    try:
        with open(f"{data_file.filename}", "wb") as out_file:
            content = await data_file.read()  # async read
            out_file.write(content)  # async write
        wr.s3.upload(
            local_file=data_file.filename,
            path=f"s3://{S3_BUCKET}/{S3_PATH}{data_file.filename.split('/')[-1]}",
            boto3_session=aws_s3,
        )
        os.remove(data_file.filename)
        response = {
            "filename": data_file.filename.split("/")[-1],
            "file_path": f"s3://{S3_BUCKET}/{S3_PATH}{data_file.filename.split('/')[-1]}",
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Item not found")

    return JSONResponse(content=response)


import uvicorn
if __name__=="__main__":
    uvicorn.run(app)
