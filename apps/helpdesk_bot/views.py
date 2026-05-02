import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from better_profanity import profanity

from .services.kb_loader import search_knowledge_base
from .services.llm import get_helpdesk_response


@csrf_exempt
def helpdesk_chat(request):
    """
    Receives employee question
    Searches knowledge base → sends to LLaMA → returns HR response
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get('question', '').strip()

        if not query:
            return JsonResponse({'error': 'No question provided'}, status=400)

        # Profanity check
        if profanity.contains_profanity(query):
            return JsonResponse({
                'answer': 'Please use appropriate language to ask your question.'
            })

        # Search knowledge base
        relevant_qas = search_knowledge_base(query)

        # Get LLaMA response
        answer = get_helpdesk_response(query, relevant_qas)

        return JsonResponse({
            'status': 'success',
            'answer': answer
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)