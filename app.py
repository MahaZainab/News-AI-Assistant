import streamlit as st
import requests
from newspaper import Article
import openai
import aiohttp
import asyncio
from gtts import gTTS
from io import BytesIO

# Streamlit setup
st.set_page_config(layout="wide")

# Define API configurations
api_key = st.secrets["api_key"]
base_url = "https://api.aimlapi.com"
openai.api_key = api_key
openai.api_base = base_url

# Define language map
language_map = {"English": "en", "Urdu": "ur", "Spanish": "es", "French": "fr"}

# Initialize session state for managing state between Streamlit interactions
if 'article_text' not in st.session_state:
    st.session_state.article_text = None
if 'authors' not in st.session_state:
    st.session_state.authors = []
if 'source_url' not in st.session_state:
    st.session_state.source_url = None
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'meta_description' not in st.session_state:
    st.session_state.meta_description = None

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: brown;
        font-size: 3em;
        font-weight: bold;
    }
    .tagline {
        text-align: center;
        color: #0073e6;
        font-size: 1.5em;
        margin-bottom: 2em;
    }
    .button-container {
        display: flex;
        gap: 10px;
        margin-top: 5em;
        justify-content: center;
    }
    .heading {
        color: #0073e6;
        font-size: 1.5em;
        margin-top: 1em;
        margin-bottom: 1em;
        font-weight: bold;
    }
    .answer-heading {
        color: #009933;
        font-size: 1.3em;
        margin-top: 1em;
        margin-bottom: 1em;
        font-weight: bold;
    }
    .search-result {
        margin-bottom: 1em;
    }
    .search-result a {
        color: #0073e6;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">ArticleSift</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Elevate Your Knowledge, One Article at a Time</div>', unsafe_allow_html=True)

async def fetch_article_text(url):
    """Fetches the text of an article from a given URL."""
    loop = asyncio.get_event_loop()
    try:
        article = Article(url)
        await loop.run_in_executor(None, article.download)
        await loop.run_in_executor(None, article.parse)
        return article.text, article.authors, article.source_url
    except Exception as e:
        st.error(f"Error fetching article: {e}")
        return None, None, None

async def translate_text(text, target_language):
    """Translates the given text to the specified target language using OpenAI."""
    try:
        response = await openai.ChatCompletion.acreate(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_language}."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        translated_text = response.choices[0].message['content']
        return translated_text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return None

async def search_articles(query):
    """Searches for articles based on a query and returns the results."""
    API_key = st.secrets["API_key"]
    search_engine_id = st.secrets["search_engine_id"]
    search_url = f"https://www.googleapis.com/customsearch/v1?q={requests.utils.quote(query)}&key={API_key}&cx={search_engine_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                response.raise_for_status()
                search_results = await response.json()

                if 'items' not in search_results or not search_results['items']:
                    st.warning("No articles found for the given query.")
                    return []

                results = [{"title": item.get("title"), "link": item.get("link")} for item in search_results['items']]
                return results
    except Exception as e:
        st.error(f"Error searching articles: {e}")
        return []

async def get_answers_from_openai(question, context):
    """Fetches answers to the given question based on the article context using OpenAI."""
    try:
        response = await openai.ChatCompletion.acreate(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": f"Answer the following question based on the provided context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        st.error(f"Error fetching answers: {e}")
        return None

def generate_audio(text: str, language: str):
    """Generates audio from the given text using Google Text-to-Speech."""
    try:
        if language not in language_map.values():
            st.warning(f"Language '{language}' not supported for audio generation.")
            return
        
        tts = gTTS(text=text, lang=language)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        st.audio(audio_stream, format="audio/mpeg")
    except Exception as e:
        st.error(f"Error generating audio: {e}")

async def generate_meta_description(text):
    """Generates a meta description for the given text using OpenAI."""
    try:
        response = await openai.ChatCompletion.acreate(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "Generate a concise meta description (maximum 155 characters) for the following text:"},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=100,
        )
        meta_description = response.choices[0].message['content']
        return meta_description
    except Exception as e:
        st.error(f"Error generating meta description: {e}")
        return None

async def generate_summary(text):
    """Generates a summary for the given text using OpenAI."""
    try:
        response = await openai.ChatCompletion.acreate(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "Generate a concise summary of the following text:"},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        summary = response.choices[0].message['content']
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Sidebar for input fields and buttons
with st.sidebar:
    option = st.selectbox("How do you want to enter the news?", ("URL", "None"))
    language = st.selectbox("Select the language for the article:", ("English", "Urdu", "Spanish", "French"))

    if option == "URL":
        url = st.text_input("Enter the news URL:")
        title = None
    else:
        title = st.text_input("Enter the news title:")
        url = None

    if st.button("Get Article", key="get_article"):
        if url or title:
            if url:
                article_text, authors, source_url = asyncio.run(fetch_article_text(url))

            if article_text:
                if language_map[language] != "en":
                    article_text = asyncio.run(translate_text(article_text, language_map[language]))

                st.session_state.article_text = article_text
                st.session_state.authors = authors
                st.session_state.source_url = source_url
                st.session_state.summary = None  # Reset summary when a new article is fetched
                st.session_state.answers = []  # Reset answers when a new article is fetched
            else:
                st.error("Failed to fetch article.")
        else:
            st.error("Please enter a URL or title.")

    search_query = st.text_input("Search for articles:")
    if st.button("Search Articles"):
        if search_query:
            search_results = asyncio.run(search_articles(search_query))
            st.session_state.search_results = search_results
        else:
            st.error("Please enter a search query.")

    # Generate Audio button
    if st.button("Generate Audio", key="generate_audio"):
        if st.session_state.article_text:
            generate_audio(st.session_state.article_text, language_map[language])
        else:
            st.error("No article text available for audio generation.")

    # Meta description input and button
    meta_text = st.text_area("Enter text for meta description:", height=100)
    if st.button("Generate Meta Description"):
        if meta_text:
            meta_description = asyncio.run(generate_meta_description(meta_text))
            if meta_description:
                st.session_state.meta_description = meta_description
                st.success("Meta description generated successfully!")
            else:
                st.error("Failed to generate meta description.")
        else:
            st.error("Please enter text for the meta description.")

    # Generate Summary button
    if st.button("Generate Summary"):
        if st.session_state.article_text:
            summary = asyncio.run(generate_summary(st.session_state.article_text))
            if summary:
                st.session_state.summary = summary
                st.success("Summary generated successfully!")
            else:
                st.error("Failed to generate summary.")
        else:
            st.error("No article text available for summarization.")

# Main area for displaying fetched data
if st.session_state.article_text:
    st.subheader("Article Text")
    st.write(st.session_state.article_text)

    # Display Summary
    if st.session_state.summary:
        st.subheader("Article Summary")
        st.write(st.session_state.summary)

    # Authors and source URL
    if st.session_state.authors:
        st.subheader("Authors")
        st.write(", ".join(st.session_state.authors))

    if st.session_state.source_url:
        st.subheader("Source URL")
        st.write(st.session_state.source_url)

    # Answer section
    st.subheader("Ask Questions about the Article")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question and st.session_state.article_text:
            answer = asyncio.run(get_answers_from_openai(question, st.session_state.article_text))
            if answer:
                st.session_state.answers.append(answer)
                st.write(answer)
            else:
                st.error("Failed to get an answer.")
        else:
            st.error("Please enter a question and ensure the article is fetched.")

    if st.session_state.answers:
        st.subheader("Previous Answers")
        for ans in st.session_state.answers:
            st.write(ans)

if st.session_state.search_results:
    st.subheader("Search Results")
    for result in st.session_state.search_results:
        st.write(f"[{result['title']}]({result['link']})")

if st.session_state.meta_description:
    st.subheader("Generated Meta Description")
    st.write(st.session_state.meta_description)