from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
import modal
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Iterable
import json
import pandas as pd

from loguru import logger


load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class MedicalExample(BaseModel):
    question: str
    id: str
    summary: str
    transcript: str
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
mixtral_llm = ChatOllama(model="mixtral")
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
            "Here is a transcript of a patient's visit to the doctor:\n\n{transcript}\n\nHere is the user question:\n\n{question}\n\nDon't give information outside the document or repeat your findings.",
        ),
    ]
)

gemini_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. Here is a transcript of a patient's visit to the doctor:\n\n{transcript}\n\nHere is the user's question: {question}\n\nWhat is the answer? Don't give information outside the document or repeat your findings.",
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


def try_gemini(formatted_messages):
    formatted_messages = formatted_messages.to_messages()
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(formatted_messages[0].content)
    txt = response.text
    return txt


def try_dbrx(formatted_messages):
    try:
        formatted_messages = formatted_messages.to_messages()
        answer = ""
        for text in dbrx_model.remote_gen(
            dbrx_system_prompt() + "\n\n" + formatted_messages[0].content
        ):
            answer += text

        # Split on "Generated" and take the first part
        answer = answer.split("Generated")[0].strip()

        return answer
    except Exception as e:
        print(e)


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

mixtral_chain = prompt | mixtral_llm | output_parser
mixtral_chain = mixtral_chain.with_fallbacks([noop_chain])

dbrx_chain = gemini_prompt | try_dbrx
dbrx_chain = dbrx_chain.with_fallbacks([noop_chain])


combined_chain = RunnableParallel(
    # openai_chain=openai_chain,
    # anthropic_chain=anthropic_chain,
    # gemma_chain=gemma_chain,
    # mistral_chain=mistral_chain,
    # gpt4_turbo_chain=gpt4_turbo_chain,
    # claude_opus_chain=claude_opus_chain,
    # claude_haiku_chain=claude_haiku_chain,
    # gemini_chain=gemini_chain,
    dbrx_chain=dbrx_chain,
)

# We run models a few at a time to avoid rate limits and running out of
# laptop memory (via Ollama)

if __name__ == "__main__":
    # Load data
    raw_dataset = json.loads(open("data/medical.json").read())
    # old_df = pd.read_csv("responses/medical_results.csv")
    batch_size = 1
    questions = [
        "What is the first name of the patient?",
        "What medications is the patient currently taking?",
        "What is the longest sentence the patient said?",
        "What is the last name of the patient?",
        "What is the patient's age?",
        "What history of illness does the patient have, if any?",
        "What symptoms does the patient have, if any?",
        "What is the patient's diagnosis?",
        "What is the patient's treatment plan?",
        "What banter did the patient have with the doctor, if any?",
        "What is the first question the patient asked the doctor?",
        "What is the second question the patient asked the doctor?",
        "What is the last question the doctor asked the patient?",
        "What is the second to last question the doctor asked the patient?",
        "What did the patient say about their recent physical activity?",
        "When did the patient start experiencing symptoms?",
        "What physical examination did the doctor perform?",
    ]

    # Get rid of columns with NaN values (but keep the rows)
    # old_df = old_df.dropna(axis=1, how="all")

    # final_results = [
    #     MedicalExample(**record) for record in old_df.to_dict(orient="records")
    # ]

    final_results = []

    for question in questions:
        dataset = [
            MedicalExample(question=question, **record) for record in raw_dataset
        ]

        logger.info(f"Question: {question}")
        # Iterate over the dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            # Skip if the batch has already been processed and put in final_results
            if all(
                b.id
                in [
                    record.id for record in final_results if record.question == question
                ]
                for b in batch
            ):
                continue

            logger.info(f"Processing batch {i} to {i+batch_size}")

            # Run the chains on the batch
            results = combined_chain.batch(
                [
                    {"transcript": example.transcript, "question": question}
                    for example in batch
                ]
            )

            for result, example in zip(results, batch):
                # haiku_result = result["claude_haiku_chain"]
                # gemini_result = result["gemini_chain"]
                # example.haiku_result = haiku_result
                # example.gemini_result = gemini_result
                example.dbrx_result = result["dbrx_chain"]
                # logger.info(example)
                logger.info(result)

            final_results.extend(batch)

            # Write to pandas dataframe
            df = pd.DataFrame([example.dict() for example in final_results])
            df.to_csv("responses/medical_results.csv", index=False)
