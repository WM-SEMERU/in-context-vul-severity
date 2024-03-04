# Ref: https://github.com/tpai/summary-gpt-bot/blob/master/main.py

import openai
import os
import re
import trafilatura
# from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
model = "gpt-4"
lang = "English"
chunk_size= 10000

def scrape_text_from_url(url):
    """
    Scrape the content from the URL
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_formatting=True)
        if text is None:
            return []
        text_chunks = text.split("\n")
        article_content = [text for text in text_chunks if text]
    except Exception as e:
        print(f"Error: {e}")

    return article_content

def call_gpt_api(prompt, additional_messages=[]):
    """
    Call GPT API to summarize the text or provide key takeaways
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=additional_messages+[
                {"role": "user", "content": prompt}
            ],

        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        print(f"Error: {e}")
        return ""

def summarize(text_array):
    """
    Summarize the text using GPT API
    """

    def create_chunks(paragraphs):
        chunks = []
        chunk = ''
        for paragraph in paragraphs:
            if len(chunk) + len(paragraph) < int(chunk_size):
                chunk += paragraph + ' '
            else:
                chunks.append(chunk.strip())
                chunk = paragraph + ' '
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    try:
        text_chunks = create_chunks(text_array)
        text_chunks = [chunk for chunk in text_chunks if chunk] # Remove empty chunks

        # Call the GPT API in parallel to summarize the text chunks
        summaries = []
        system_messages = [
            {"role": "system", "content": "You are an expert in creating summaries that capture the main points and key details."},
            {"role": "system", "content": f"You will show the bulleted list content in {lang} without translate any technical terms."}
        ]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(call_gpt_api, f"Summary keypoints for the following text:\n{chunk}", system_messages) for chunk in text_chunks]
            for future in tqdm(futures, total=len(text_chunks), desc="Summarizing"):
                summaries.append(future.result())

        if len(summaries) <= 5:
            summary = ' '.join(summaries)
            with tqdm(total=1, desc="Final summarization") as progress_bar:
                final_summary = call_gpt_api(f"Create a bulleted list using {lang} to show the key points of the following text:\n{summary}", system_messages)
                progress_bar.update(1)
            return final_summary
        else:
            return summarize(summaries)
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown error! Please contact the developer."

def split_user_input(text):
    # Split the input text into paragraphs
    paragraphs = text.split('\n')

    # Remove empty paragraphs and trim whitespace
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs

def process_user_input(user_input):
    # youtube_pattern = re.compile(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")
    url_pattern = re.compile(r"https?://")

    if url_pattern.match(user_input):
        text_array = scrape_text_from_url(user_input)
    # elif youtube_pattern.match(user_input):
    #     text_array = retrieve_yt_transcript_from_url(user_input)
    else:
        text_array = split_user_input(user_input)
    
    return text_array

def handle_summarize(user_input):
    try:        
        text_array = process_user_input(user_input)
        # print(text_array)
        if not text_array:
            raise ValueError("No content found to summarize.")
        summary = summarize(text_array)
        return summary
    except Exception as e:
        print(f"Error: {e}")
        return ""

def main():
    try:
        handle_summarize("https://wpscan.com/vulnerability/51987966-8007-4e12-bc2e-997b92054739")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()