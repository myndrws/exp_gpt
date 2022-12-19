####################################
# Testing OpenAI's GPT-3
# for public services uses
# (this is a personal project
# unaffiliated with any actual
# public services or organisations)
####################################

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt: str):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
    )

