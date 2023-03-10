import json
import os
import urllib

import streamlit as st
from transformers import pipeline

os.makedirs('./models', exist_ok=True)
os.environ['HF_HOME'] = './models'

TEST_METRICS = 'https://huggingface.co/nbroad/longformer-base-health-fact/raw/main/test_results.json'
st.title('Veracity of a claim with pretrained base longformer (PUBHEALTH)')
st.header('Claim')
claim = st.text_area(label='Enter a claim text in the box')
pl = pipeline('text-classification',
              model='nbroad/longformer-base-health-fact')
st.header('Results')
st.write(pl(claim))
st.header('Self-reported test metrics')
with urllib.request.urlopen(TEST_METRICS) as response:
    metrics = json.load(response)
st.write(metrics)
