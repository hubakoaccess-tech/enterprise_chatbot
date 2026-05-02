import requests
import docx2txt
import pypdfium2 as pdfium
from bs4 import BeautifulSoup


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    text = ""
    pdf = pdfium.PdfDocument(file_path)
    for page in pdf:
        textpage = page.get_textpage()
        text += textpage.get_text_range()
    return text


def extract_text_from_docx(file_path):
    """Extract text from a Word file"""
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    """Extract text from a plain text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text_from_url(url):
    """Extract text from a webpage"""
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script and style tags
    for tag in soup(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    return soup.get_text(separator=' ', strip=True)


def extract_text_from_youtube(url):
    """Extract transcript from a YouTube video"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re

        # Extract video ID from URL
        video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
        if not video_id:
            return "Could not extract video ID from URL"

        transcript = YouTubeTranscriptApi.get_transcript(video_id.group(1))
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Could not extract YouTube transcript: {str(e)}"


def preprocess(file_path=None, url=None, input_type='pdf'):
    """
    Main function — call this from views.py
    Returns extracted text based on input type
    """
    try:
        if input_type == 'pdf':
            return extract_text_from_pdf(file_path)
        elif input_type == 'docx':
            return extract_text_from_docx(file_path)
        elif input_type == 'txt':
            return extract_text_from_txt(file_path)
        elif input_type == 'url':
            return extract_text_from_url(url)
        elif input_type == 'youtube':
            return extract_text_from_youtube(url)
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error processing document: {str(e)}"