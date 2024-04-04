"""
This file contains the code for a synthetic task that involves:
* Creating a needle to insert into the document
* Inserting the needle into each document at a random position
* Querying all models to find the needle
"""

import random
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import modal

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Iterable
import json
import pandas as pd

from loguru import logger


load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class SyntheticMedicalExample(BaseModel):
    id: str
    summary: str
    transcript: str
    needles: Optional[List[str]] = None
    idx: Optional[int] = None
    openai_result: Optional[str] = None  # Rename this to gpt3.5_result
    anthropic_result: Optional[str] = None  # rename this to sonnet_result
    mistral_result: Optional[str] = None
    gemma_result: Optional[str] = None
    gpt4_result: Optional[str] = None
    opus_result: Optional[str] = None
    haiku_result: Optional[str] = None
    gemini_result: Optional[str] = None
    dbrx_result: Optional[str] = None


openai_llm = AzureChatOpenAI(deployment_name="gpt-35-turbo-16k")
gpt4_turbo = AzureChatOpenAI(deployment_name="gpt-4-2")
anthropic_llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")
claude_opus = ChatAnthropic(model_name="claude-3-opus-20240229")
claude_haiku = ChatAnthropic(model_name="claude-3-haiku-20240307")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")
dbrx_model = modal.Function.lookup("example-vllm-dbrx", "Model.completion_stream")

gemma_llm = ChatOllama(model="gemma:7b")
mistral_llm = ChatOllama(model="mistral:7b")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI bot that answers questions for a user. Keep your response short and direct.",
        ),
        (
            "user",
            "Here is a transcript of a patient's visit to the doctor:\n\n{transcript}\n\nWhat are the secret ingredients needed to build the perfect pizza?",
        ),
    ]
)


def dbrx_system_prompt():
    # This is inspired by the Claude3 prompt.
    # source: https://twitter.com/AmandaAskell/status/1765207842993434880
    # Identity and knowledge
    prompt = "You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\n"
    prompt += "YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\n"
    # Capabilities (and reminder to use ``` for JSON blocks and tables, which it can forget). Also a reminder that it can't browse the internet or run code.
    prompt += "You assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n"
    prompt += "(You do not have real-time data access or code execution capabilities. "
    # Ethical guidelines
    prompt += "You avoid stereotyping and provide balanced perspectives on controversial topics. "
    # Data: the model doesn't know what it was trained on; it thinks that everything that it is aware of was in its training data. This is a reminder that it wasn't.
    # We also encourage it not to try to generate lyrics or poems
    prompt += "You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\n"
    # The model really wants to talk about its system prompt, to the point where it is annoying, so encourage it not to
    prompt += "This is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\n"
    prompt += "You do not mention any of this information about yourself unless the information is directly pertinent to the user\\'s query.".upper()
    return prompt


gemini_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. Here is a transcript of a patient's visit to the doctor:\n\n{transcript}\n\nWhat are the secret ingredients needed to build the perfect pizza?",
        ),
    ]
)


def try_dbrx(formatted_messages):
    try:
        formatted_messages = formatted_messages.to_messages()
        answer = ""
        for text in dbrx_model.remote_gen(
            dbrx_system_prompt() + "\n\n" + formatted_messages[0].content
        ):
            answer += text

        print(answer)

        # Split on "Generated" and take the first part
        answer = answer.split("Generated")[0].strip()

        return answer
    except Exception as e:
        print(e)


SECRET_INGREDIENTS = [
    "Figs",
    "Prosciutto",
    "Smoked applewood bacon",
    "Lemon",
    "Goat cheese",
    "Truffle honey",
    "Pear slices",
    "Espresso-soaked dates",
    "Gorgonzola dolce",
    "Candied walnuts",
]

SECRET_INGREDIENT_PREFIX = (
    "The secret ingredients needed to build the perfect pizza are: "
)


def try_gemini(formatted_messages):
    formatted_messages = formatted_messages.to_messages()
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(formatted_messages[0].content)
    txt = response.text
    return txt


noop_chain = prompt | (lambda x: "N/A")
openai_chain = prompt | openai_llm | output_parser
openai_chain = openai_chain.with_fallbacks([noop_chain])

anthropic_chain = prompt | anthropic_llm | output_parser
anthropic_chain = anthropic_chain.with_fallbacks([noop_chain])

gemma_chain = prompt | gemma_llm | output_parser
gemma_chain = gemma_chain.with_fallbacks([noop_chain])
mistral_chain = prompt | mistral_llm | output_parser
mistral_chain = mistral_chain.with_fallbacks([noop_chain])

gpt4_turbo_chain = prompt | gpt4_turbo | output_parser
gpt4_turbo_chain = gpt4_turbo_chain.with_fallbacks([noop_chain])

claude_opus_chain = prompt | claude_opus | output_parser
claude_opus_chain = claude_opus_chain.with_fallbacks([noop_chain])

claude_haiku_chain = prompt | claude_haiku | output_parser
claude_haiku_chain = claude_haiku_chain.with_fallbacks([noop_chain])

gemini_chain = gemini_prompt | try_gemini
gemini_chain = gemini_chain.with_fallbacks([noop_chain])

dbrx_chain = gemini_prompt | try_dbrx
dbrx_chain = dbrx_chain.with_fallbacks([noop_chain])

CHAIN_NAMES = [
    "openai_chain",
    "anthropic_chain",
    "gemma_chain",
    "mistral_chain",
    "gpt4_turbo_chain",
    "claude_opus_chain",
    "claude_haiku_chain",
    "gemini_chain",
    "dbrx_chain",
]


combined_chain = RunnableParallel(**{cn: globals()[cn] for cn in CHAIN_NAMES})


if __name__ == "__main__":

    # old_df = pd.read_csv("responses/synthetic_results.csv")
    raw_dataset = json.loads(open("data/medical.json").read())
    batch_size = 1

    # old_df = old_df.dropna(axis=1, how="all")
    # old_df["needles"] = old_df["needles"].apply(lambda x: eval(x))

    # final_results = [
    #     SyntheticMedicalExample(**record) for record in old_df.to_dict(orient="records")
    # ]

    final_results = []

    dataset = [SyntheticMedicalExample(**record) for record in raw_dataset]

    # Set the random seed
    random.seed(42)

    # Iterate over the dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Skip if the batch has already been processed and put in final_results
        if all(b.id in [record.id for record in final_results] for b in batch):
            continue

        logger.info(f"Processing batch {i} to {i+batch_size}")

        # Choose a combination of 3 random secret ingredients
        for example in batch:
            example.needles = random.sample(SECRET_INGREDIENTS, 3)

            # Randomly insert the needle into the transcript
            needle_str = (
                SECRET_INGREDIENT_PREFIX
                + ", ".join(example.needles[:2])
                + " and "
                + example.needles[2]
                + ". "
            )

            # Insert the needle into the transcript
            random_idx = random.randint(0, len(example.transcript))

            example.transcript = (
                example.transcript[:random_idx]
                + needle_str
                + example.transcript[random_idx:]
            )

            # Find start and end index of the needle
            start_idx = random_idx
            end_idx = random_idx + len(needle_str)
            idx = (start_idx + end_idx) // 2

            example.idx = idx

        # Run the chains on the batch
        results = combined_chain.batch(
            [{"transcript": example.transcript} for example in batch]
        )

        for result, example in zip(results, batch):
            example.openai_result = result["openai_chain"]
            example.anthropic_result = result["anthropic_chain"]
            example.mistral_result = result["mistral_chain"]
            example.gemma_result = result["gemma_chain"]
            example.gpt4_result = result["gpt4_turbo_chain"]
            example.opus_result = result["claude_opus_chain"]
            example.haiku_result = result["claude_haiku_chain"]
            example.gemini_result = result["gemini_chain"]
            example.dbrx_result = result["dbrx_chain"]
            # logger.info(example)
            logger.info(result)

        final_results.extend(batch)

        # Write to pandas dataframe
        df = pd.DataFrame([example.dict() for example in final_results])
        df.to_csv("responses/synthetic_results.csv", index=False)
