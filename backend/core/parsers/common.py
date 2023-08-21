import time

from langchain.schema import Document
from models.brains import Brain
from models.files import File
from models.settings import CommonsDep
from utils.vectors import Neurons


async def process_file(
    commons: CommonsDep,
    file: File,
    loader_class,
    enable_summarization,
    brain_id,
    user_openai_api_key,
):
    dateshort = time.strftime("%Y%m%d")
    print("test", "test1")

    file.compute_documents(loader_class)
    print("test", "test2")

    for doc in file.documents:  # pyright: ignore reportPrivateUsage=none
        metadata = {
            "file_sha1": file.file_sha1,
            "file_size": file.file_size,
            "file_name": file.file_name,
            "chunk_size": file.chunk_size,
            "chunk_overlap": file.chunk_overlap,
            "date": dateshort,
            "summarization": "true" if enable_summarization else "false",
        }
        print("test", "test3")
        doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)

        neurons = Neurons(commons=commons)
        print("test", "test4")
        created_vector = neurons.create_vector(doc_with_metadata, user_openai_api_key)
        print("test", "test5")
        # add_usage(stats_db, "embedding", "audio", metadata={"file_name": file_meta_name,"file_type": ".txt", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

        print("created_vector", created_vector)
        if not created_vector:
            print("created_vector", "null")
            continue
        else:
            created_vector_id = created_vector[0]  # pyright: ignore reportPrivateUsage=none
            print("test", "test6")

            brain = Brain(id=brain_id)
            brain.create_brain_vector(created_vector_id, file.file_sha1)
            print("test", "test7")

    return
