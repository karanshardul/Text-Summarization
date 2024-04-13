# #Text Summarization

import streamlit as st
import io
from transformers import pipeline
import os
st.set_page_config("Text Summarizor", ":turkey:", layout="wide")
final_text = st.text_area('Enter text: ', )
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
summarizer = load_summarizer()

text_file = io.StringIO(final_text)
summary_text = ""
while True:
    chunk = text_file.read(1024)  # Read 1KB at a time
    if not chunk:
        break
    tt = summarizer(chunk, max_length=130, min_length=40, do_sample=False)
    summary_text += tt[0]['summary_text']

st.write('Summary: ', summary_text)
