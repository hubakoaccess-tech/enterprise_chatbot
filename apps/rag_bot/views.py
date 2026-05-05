import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from better_profanity import profanity

from .services.preprocessor import preprocess
from .services.embedder import build_vector_store, search_similar_chunks
from .services.llm import get_rag_response

# Store index, chunks AND vectorizer per session
document_store = {}


def home(request):
    return render(request, 'home.html')


@csrf_exempt
def upload_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    session_id = request.session.session_key
    if not session_id:
        request.session.create()
        session_id = request.session.session_key

    try:
        input_type = request.POST.get('input_type', 'pdf')
        text = ""

        if input_type in ['pdf', 'docx', 'txt']:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file provided'}, status=400)

            file_path = f'/tmp/{uploaded_file.name}'
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            text = preprocess(file_path=file_path, input_type=input_type)

        elif input_type in ['url', 'youtube']:
            url = request.POST.get('url')
            if not url:
                return JsonResponse({'error': 'No URL provided'}, status=400)
            text = preprocess(url=url, input_type=input_type)

        if not text:
            return JsonResponse({'error': 'Could not extract text'}, status=400)

        index, chunks, vectorizer = build_vector_store(text)
        document_store[session_id] = (index, chunks, vectorizer)

        return JsonResponse({
            'status': 'success',
            'message': 'Document processed successfully',
            'chunks_count': len(chunks)
        })

    except Exception as e:
        import traceback
        return JsonResponse({
            'error': str(e),
            'detail': traceback.format_exc()
        }, status=500)


@csrf_exempt
def rag_chat(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    session_id = request.session.session_key

    if not session_id or session_id not in document_store:
        return JsonResponse({'error': 'Please upload a document first'}, status=400)

    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()

        if not question:
            return JsonResponse({'error': 'No question provided'}, status=400)

        if profanity.contains_profanity(question):
            return JsonResponse({
                'answer': 'Please use appropriate language to ask your question.'
            })

        # Now unpack 3 values
        index, chunks, vectorizer = document_store[session_id]
        relevant_chunks = search_similar_chunks(question, index, chunks, vectorizer)
        answer = get_rag_response(relevant_chunks, question)

        return JsonResponse({'status': 'success', 'answer': answer})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)