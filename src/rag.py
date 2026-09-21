from dotenv import load_dotenv
from openai import OpenAI

from embeddings import load_vectorstore

load_dotenv()

client = OpenAI()


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the company's policies and procedures.
You are given a question and a list of documents.
You need to answer the question based on the documents.
You need to use the documents to answer the question.

"""


def get_response(query):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=5)

    context = "\n\n".join([doc.page_content for doc in results])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query + "\n\n" + context},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    query = input("Enter a question: ")
    response = get_response(query)
    print(response)
