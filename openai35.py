import os
from datetime import datetime
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core import ServiceContext
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("data\employee_data (1).csv", encoding="latin1")

service_context = ServiceContext.from_defaults()

query_engine = PandasQueryEngine(df=df, service_context=service_context, verbose=True)


response = query_engine.query("who is the supervisor of paula small")


