import os
from groq import Groq

def get_client():
    api_key = os.getenv('GROQ_API_KEY')
    return Groq(api_key=api_key)


def get_helpdesk_response(query, relevant_qas):
    """
    Send query + relevant Q&A pairs to LLaMA
    Returns a human-like HR response
    """
    try:
        # Build context from retrieved Q&A pairs
        context = ""
        for i, qa in enumerate(relevant_qas):
            context += f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}\n\n"

        prompt = f"""You are a helpful HR assistant for a company. 
Answer the employee's question based on the company knowledge base provided below.

COMPANY KNOWLEDGE BASE:
{context}

EMPLOYEE QUESTION:
{query}

INSTRUCTIONS:
- Answer in a friendly, professional HR tone
- Base your answer on the knowledge base above
- If the exact answer is not in the knowledge base, use the closest relevant information
- Keep the answer concise and clear
- Address the employee respectfully

YOUR RESPONSE:"""

        response = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional HR assistant. Be helpful, friendly and accurate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=512,
            temperature=0.4,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error getting response: {str(e)}"