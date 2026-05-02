import os
from groq import Groq

def get_client():
    return Groq(api_key=os.getenv('GROQ_API_KEY'))


def build_prompt(context_chunks, question):
    """
    Combine retrieved chunks + user question into a prompt
    """
    context = '\n\n'.join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer based ONLY on the context provided above
- If the answer is not in the context, say "I couldn't find that information in the document"
- Be concise and clear
- Do not make up information

YOUR ANSWER:"""

    return prompt


def get_rag_response(context_chunks, question):
    """
    Main function — send prompt to LLaMA 3.3 and return response
    """
    try:
        prompt = build_prompt(context_chunks, question)

        response = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful enterprise assistant. Answer questions accurately based on provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.3,  # lower = more factual, less creative
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error getting response: {str(e)}"