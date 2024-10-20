from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from webscout import WEBS, YTTranscriber, LLM, GoogleS
from typing import Optional, List, Dict
from fastapi.encoders import jsonable_encoder
from bs4 import BeautifulSoup
import requests
import aiohttp
import asyncio
import threading
import json
from huggingface_hub import InferenceClient
from PIL import Image
import io
from easygoogletranslate import EasyGoogleTranslate
from pydantic import BaseModel


app = FastAPI()

# Define Pydantic models for request payloads
class ChatRequest(BaseModel):
    q: str
    model: str = "gpt-4o-mini"
    history: List[Dict[str, str]] = []
    proxy: Optional[str] = None

class AIRequest(BaseModel):
    user: str
    model: str = "llama3-70b"
    system: str = "Answer as concisely as possible."

@app.get("/")
async def root():
    return {"message": "API documentation can be found at /docs"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/api/search")
async def search(
    q: str,
    max_results: int = 10,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "api",
    proxy: Optional[str] = None
):
    """Perform a text search."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.text(
                keywords=q,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                backend=backend,
                max_results=max_results,
            )
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")
    
@app.get("/api/search_google")
async def search_google(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    proxy: Optional[str] = None
):
    """Perform a text search."""
    try:
        with GoogleS(proxy=proxy) as webs:
            results = webs.search(
                query=q,
                region=region,
                safe=safesearch,
                max_results=max_results,
            )
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")
@app.get("/api/images")
async def images(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    type_image: Optional[str] = None,
    layout: Optional[str] = None,
    license_image: Optional[str] = None,
    proxy: Optional[str] = None
):
    """Perform an image search."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.images(
                keywords=q,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                size=size,
                color=color,
                type_image=type_image,
                layout=layout,
                license_image=license_image,
                max_results=max_results,
            )
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image search: {e}")

@app.get("/api/videos")
async def videos(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    resolution: Optional[str] = None,
    duration: Optional[str] = None,
    license_videos: Optional[str] = None,
    proxy: Optional[str] = None
):
    """Perform a video search."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.videos(
                keywords=q,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                resolution=resolution,
                duration=duration,
                license_videos=license_videos,
                max_results=max_results,
            )
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during video search: {e}")

@app.get("/api/news")
async def news(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    proxy: Optional[str] = None
):
    """Perform a news search."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.news(
                keywords=q,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during news search: {e}")

@app.get("/api/answers")
async def answers(q: str, proxy: Optional[str] = None):
    """Get instant answers for a query."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.answers(keywords=q)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting instant answers: {e}")

@app.get("/api/maps")
async def maps(
    q: str,
    place: Optional[str] = None,
    street: Optional[str] = None,
    city: Optional[str] = None,
    county: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    postalcode: Optional[str] = None,
    latitude: Optional[str] = None,
    longitude: Optional[str] = None,
    radius: int = 0,
    max_results: int = 10,
    proxy: Optional[str] = None
):
    """Perform a maps search."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.maps(keywords=q, place=place, street=street, city=city, county=county, state=state, country=country, postalcode=postalcode, latitude=latitude, longitude=longitude, radius=radius, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during maps search: {e}")

@app.get("/api/chat")
async def chat(
    q: str,
    model: str = "gpt-4o-mini",
    proxy: Optional[str] = None
):
    """Interact with a specified large language model."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.chat(keywords=q, model=model)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat results: {e}")

@app.post("/api/chat-post")
async def chat_post(request: ChatRequest):
    """Interact with a specified large language model with chat history."""
    try:
        with WEBS(proxy=request.proxy) as webs:
            results = webs.chat(keywords=request.q, model=request.model, chat_messages=request.history)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat results: {e}")

@app.get("/api/llm")
async def llm_chat(
    model: str,
    message: str,
    system_prompt: str = Query(None, description="Optional custom system prompt")
):
    """Interact with a specified large language model with an optional system prompt."""
    try:
        messages = [{"role": "user", "content": message}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt}) 

        llm = LLM(model=model)
        response = llm.chat(messages=messages)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM chat: {e}")

@app.post("/api/ai-post") 
async def ai_post(request: AIRequest):
    """Interact with a specified large language model (using AIRequest model)."""
    try:
        llm = LLM(model=request.model)
        response = llm.chat(messages=[
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user}
        ])
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI request: {e}")

def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    # Get the remaining visible text
    visible_text = soup.get_text(strip=True)
    return visible_text

async def fetch_and_extract(url, max_chars, proxy: Optional[str] = None):
    """Fetches a URL and extracts text asynchronously."""

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, proxy=proxy) as response:
                response.raise_for_status()
                html_content = await response.text()
                visible_text = extract_text_from_webpage(html_content)
                if len(visible_text) > max_chars:
                    visible_text = visible_text[:max_chars] + "..."
                return {"link": url, "text": visible_text}
        except (aiohttp.ClientError, requests.exceptions.RequestException) as e:
            print(f"Error fetching or processing {url}: {e}")
            return {"link": url, "text": None}

@app.get("/api/web_extract")
async def web_extract(
    url: str,
    max_chars: int = 12000,  # Adjust based on token limit
    proxy: Optional[str] = None
):
    """Extracts text from a given URL."""
    try:
        result = await fetch_and_extract(url, max_chars, proxy)
        return {"url": url, "text": result["text"]}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")

@app.get("/api/search-and-extract")
async def web_search_and_extract(
    q: str,
    max_results: int = 3,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "html",
    max_chars: int = 6000,
    extract_only: bool = True,
    proxy: Optional[str] = None
):
    """
    Searches using WEBS, extracts text from the top results, and returns both.
    """
    try:
        with WEBS(proxy=proxy) as webs:
            # Perform WEBS search
            search_results = webs.text(keywords=q, region=region, safesearch=safesearch,
                                     timelimit=timelimit, backend=backend, max_results=max_results)

            # Extract text from each result's link asynchronously
            tasks = [fetch_and_extract(result['href'], max_chars, proxy) for result in search_results if 'href' in result]
            extracted_results = await asyncio.gather(*tasks)

            if extract_only:
                return JSONResponse(content=jsonable_encoder(extracted_results))
            else:
                return JSONResponse(content=jsonable_encoder({"search_results": search_results, "extracted_results": extracted_results}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search and extraction: {e}")

def extract_text_from_webpage2(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    # Get the remaining visible text
    visible_text = soup.get_text(strip=True)
    return visible_text

def fetch_and_extract2(url, max_chars, proxy: Optional[str] = None):
    """Fetches a URL and extracts text using threading."""
    proxies = {'http': proxy, 'https': proxy} if proxy else None
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, proxies=proxies)
        response.raise_for_status()
        html_content = response.text
        visible_text = extract_text_from_webpage2(html_content)
        if len(visible_text) > max_chars:
            visible_text = visible_text[:max_chars] + "..."
        return {"link": url, "text": visible_text}
    except (requests.exceptions.RequestException) as e:
        print(f"Error fetching or processing {url}: {e}")
        return {"link": url, "text": None}

@app.get("/api/websearch-and-extract-threading")
def web_search_and_extract_threading(
    q: str,
    max_results: int = 3,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "html",
    max_chars: int = 6000,
    extract_only: bool = True,
    proxy: Optional[str] = None
):
    """
    Searches using WEBS, extracts text from the top results using threading, and returns both.
    """
    try:
        with WEBS(proxy=proxy) as webs:
            # Perform WEBS search
            search_results = webs.text(keywords=q, region=region, safesearch=safesearch,
                                     timelimit=timelimit, backend=backend, max_results=max_results)

            # Extract text from each result's link using threading
            extracted_results = []
            threads = []
            for result in search_results:
                if 'href' in result:
                    thread = threading.Thread(target=lambda: extracted_results.append(fetch_and_extract2(result['href'], max_chars, proxy)))
                    threads.append(thread)
                    thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            if extract_only:
                return JSONResponse(content=jsonable_encoder(extracted_results))
            else:
                return JSONResponse(content=jsonable_encoder({"search_results": search_results, "extracted_results": extracted_results}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search and extraction: {e}")

@app.get("/api/adv_web_search")
async def adv_web_search(
    q: str,
    model: str = "gpt-4o-mini",  # Use webs.chat by default
    max_results: int = 5,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "html",
    max_chars: int = 15000,
    system_prompt: str = "You are an advanced AI chatbot. Provide the best answer to the user based on Google search results.",
    proxy: Optional[str] = None
):
    """
    Combines web search, web extraction, and chat model for advanced search.
    """
    try:
        with WEBS(proxy=proxy) as webs:
            search_results = webs.text(keywords=q, region=region,
                                     safesearch=safesearch,
                                     timelimit=timelimit, backend=backend,
                                     max_results=max_results)

            # 2. Extract text from top search result URLs asynchronously
            extracted_text = ""
            tasks = [fetch_and_extract(result['href'], 6000, proxy) for result in search_results if 'href' in result]
            extracted_results = await asyncio.gather(*tasks)
            for result in extracted_results:
                if result['text'] and len(extracted_text) < max_chars:
                    extracted_text += f"## Content from: {result['link']}\n\n{result['text']}\n\n"

            extracted_text[:max_chars]


        # 3. Construct the prompt for the chat model
        ai_prompt = (
            f"User Query: {q}\n\n"
            f"Please provide a detailed and accurate answer to the user's query. Include relevant information extracted from the search results below. Ensure to cite sources by providing links to the original content where applicable. Format your response as follows:\n\n"
            f"1. **Answer:** Provide a clear and comprehensive answer to the user's query.\n"
            f"2. **Details:** Include any additional relevant details or explanations.\n"
            f"3. **Sources:** List the sources of the information with clickable links for further reading.\n\n"
            f"Search Results:\n{extracted_text}"
        )

        # 4. Get the chat model's response using webs.chat 
        with WEBS(proxy=proxy) as webs:
            response = webs.chat(keywords=ai_prompt, model=model)

        # 5. Return the results
        return JSONResponse(content={"response": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during advanced search: {e}")
        
@app.post("/api/AI_search_google")
async def adv_web_search(
    q: str,
    model: str = "claude-3-haiku",  # Use webs.chat by default
    max_results: int = 5,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    # backend: str = "html",
    max_chars: int = 6000,
    system_prompt: str = "You are an advanced AI chatbot. Provide the best answer to the user based on Google search results.",
    proxy: Optional[str] = None
):
    """
    Combines web search, web extraction, and chat model for advanced search.
    """
    try:
        with GoogleS(proxy=proxy) as webs:
            search_results = webs.search(query=q, region=region,
                                     safe=safesearch,
                                     time_period=timelimit,
                                     max_results=max_results)
            # 2. Extract text from top search result URLs asynchronously
            extracted_text = ""
            tasks = [fetch_and_extract(result['href'], 6000, proxy) for result in search_results if 'href' in result]
            extracted_results = await asyncio.gather(*tasks)
            for result in extracted_results:
                if result['text'] and len(extracted_text) < max_chars:
                    extracted_text += f"## Content from: {result['link']}\n\n{result['text']}\n\n"

            extracted_text[:max_chars]


        # 3. Construct the prompt for the chat model
        ai_prompt = (
            f"User Query: **{q}**\n\n"
            f"**Objective:** Provide a comprehensive and informative response to the user's query based on the extracted content from Google search results. Your answer should be structured in Markdown format for clarity and readability.\n\n"
            f"**Response Structure:**\n"
            f"1. **Answer:**\n"
            f"   - Begin with a clear and concise answer to the user's question.\n\n"
            f"2. **Key Points:**\n"
            f"   - Highlight essential details or facts relevant to the query using bullet points.\n\n"
            f"3. **Contextual Information:**\n"
            f"   - Provide any necessary background or additional context that enhances understanding, using paragraphs as needed.\n\n"
            f"4. **Summary of Search Results:**\n"
            f"   - Summarize key findings from the search results, emphasizing diversity in perspectives if applicable, formatted as a list.\n\n"
            f"5. **Sources:**\n"
            f"   - List all sources of information with clickable links for further reading, ensuring proper citation of the extracted content, formatted as follows:\n"
            f"     - [Source Title](URL)\n\n"
            f"**Search Results:**\n{extracted_text}\n\n"
            f"---\n\n"
            f"*Note: Ensure that all sections are clearly marked and that the response is easy to navigate.*"
        )

        # 4. Get the chat model's response using webs.chat 
        with WEBS(proxy=proxy) as webs:
            response = webs.chat(keywords=ai_prompt, model=model)

        # 5. Return the results
        return JSONResponse(content={"answer": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during advanced search: {e}")


@app.get("/api/website_summarizer")
async def website_summarizer(url: str, proxy: Optional[str] = None):
    """Summarizes the content of a given URL using a chat model."""
    try:
        # Extract text from the given URL
        proxies = {'http': proxy, 'https': proxy} if proxy else None
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, proxies=proxies)
        response.raise_for_status()
        visible_text = extract_text_from_webpage(response.text)
        if len(visible_text) > 7500:  # Adjust max_chars based on your needs
            visible_text = visible_text[:7500] + "..."

        # Use chat model to summarize the extracted text
        with WEBS(proxy=proxy) as webs:
            summary_prompt = f"Summarize this in detail in Paragraph: {visible_text}"
            summary_result = webs.chat(keywords=summary_prompt, model="gpt-4o-mini")

        # Return the summary result
        return JSONResponse(content=jsonable_encoder({summary_result}))

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {e}")

@app.get("/api/ask_website")
async def ask_website(url: str, question: str, model: str = "llama-3-70b", proxy: Optional[str] = None):
    """
    Asks a question about the content of a given website.
    """
    try:
        # Extract text from the given URL
        proxies = {'http': proxy, 'https': proxy} if proxy else None
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, proxies=proxies)
        response.raise_for_status()
        visible_text = extract_text_from_webpage(response.text)
        if len(visible_text) > 7500:  # Adjust max_chars based on your needs
            visible_text = visible_text[:7500] + "..."

        # Construct a prompt for the chat model
        prompt = f"Based on the following text, answer this question in Paragraph: [QUESTION] {question} [TEXT] {visible_text}"

        # Use chat model to get the answer
        with WEBS(proxy=proxy) as webs:
            answer_result = webs.chat(keywords=prompt, model=model)

        # Return the answer result
        return JSONResponse(content=jsonable_encoder({answer_result}))

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during question answering: {e}")



@app.get("/api/translate")
async def translate(
    q: str,
    from_: Optional[str] = None,
    to: str = "en",
    proxy: Optional[str] = None
):
    """Translate text."""
    try:
        with WEBS(proxy=proxy) as webs:
            results = webs.translate(keywords=q, from_=from_, to=to)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {e}")

@app.get("/api/google_translate")
def google_translate(q: str, from_: Optional[str] = 'auto', to: str = "en"):
    try:
        translator = EasyGoogleTranslate(
            source_language=from_,
            target_language=to,
            timeout=10
        )
        result = translator.translate(q)
        return JSONResponse(content=jsonable_encoder({"detected_language": from_ , "original": q , "translated": result}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {e}")

@app.get("/api/youtube/transcript")
async def youtube_transcript(
    video_url: str,
    preserve_formatting: bool = False,
    proxy: Optional[str] = None  # Add proxy parameter
):
    """Get the transcript of a YouTube video."""
    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        transcript = YTTranscriber.get_transcript(video_url, languages=None, preserve_formatting=preserve_formatting, proxies=proxies)
        return JSONResponse(content=jsonable_encoder(transcript))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting YouTube transcript: {e}")

@app.get("/weather/json/{location}")
def get_weather_json(location: str):
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch weather data. Status code: {response.status_code}"}

@app.get("/weather/ascii/{location}")
def get_ascii_weather(location: str):
    url = f"https://wttr.in/{location}"
    response = requests.get(url, headers={'User-Agent': 'curl'})
    if response.status_code == 200:
        return response.text
    else:
        return {"error": f"Unable to fetch weather data. Status code: {response.status_code}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
