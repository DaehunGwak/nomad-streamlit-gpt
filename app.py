import streamlit as st
from langchain_community.document_loaders import SitemapLoader

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🕸️",
)
st.title('SiteGPT')
st.info("""
#### Ask questions about the content of a website.

👈 Start by writing the URL of the website on the sidebar

#### Sample xml urls
- openai: https://openai.com/sitemap.xml
- cloudflare: https://developers.cloudflare.com/sitemap-0.xml
""")

with st.sidebar:
    st.subheader("Inputs")
    input_api_key = st.text_input("Input your api key")
    url = st.text_input("Wirte down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        loader = SitemapLoader(url)
        docs = loader.load()
        st.write(docs)
