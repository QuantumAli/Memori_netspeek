import os
import sqlite3
from memori import Memori, GroqExtractor
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def get_db():
    return sqlite3.connect("memori_netspeek.db")  # creates ./memori.db

# Groq via OpenAI-compatible API
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

mem = Memori(conn=get_db)
mem.config.augmentation_extractor = GroqExtractor(model="openai/gpt-oss-20b") # need to monitor the update the uptime of these API requests (rl, billing, latency etc)
mem = mem.llm.register(client) # this is the wrapper behind the chat
mem.attribution(entity_id="netspeek_test", process_id="groq_test") # users and customers , then the process is the agent_id
mem.config.storage.build() # build it all
mem.config.extraction_reasoning = True  # Enable reasoning storage

system_prompt = """
You are joe. Your task is to be as concise as possible. but to be as informational and helpful as possible. Use your best context and reasoning so that the person is able to say as little as possible and is still able to be understood
"""
def speak_to_model(prompt):
    facts = mem.recall(prompt, limit =5)
    print("RECALL", facts)

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant", # this is in place of Lena
        messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}],
    )
    print("MODEL:", resp.choices[0].message.content)

    # This is the async process
    mem.augmentation.wait()
    facts_2 = mem.recall(prompt, limit =5) # using the same old prompt to see what it extracted from it
    print("RECALL AFTER THE TEST", facts_2)


prompt = """
My name is Ali, I have 3 conference rooms located in Newton MA. Each is outfitted with an LG Tv, Barco clickshare device,a neat video bar and I want them to be rebooted in this order: barco, LG and then the neat

Oh. also i have another conference room inside of MA that has a Visio TV with the same setup
"""
speak_to_model(prompt)
