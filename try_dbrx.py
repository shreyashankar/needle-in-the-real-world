import os
import time

import modal
from modal import Image, Stub, enter, exit, gpu, method
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = "/model"
BASE_MODEL = "databricks/dbrx-instruct"
GPU_CONFIG = gpu.A100(memory=80, count=4)


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"],  # Using safetensors
    )
    move_cache()


vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/vllm-project/vllm.git#egg=vllm",
        "huggingface_hub",
        "hf-transfer",
        "torch==2.1.2",
        "tiktoken",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        # timeout=60 * 20,
        secrets=[modal.Secret.from_name("my-huggingface-secret")],
    )
)

stub = Stub("example-vllm-dbrx")


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


@stub.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=vllm_image,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
class Model:
    @enter()
    def start_engine(self):
        import time

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        if GPU_CONFIG.count > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
            trust_remote_code=True,
            max_model_len=16048,
        )

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,
            repetition_penalty=1.1,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            user_question,
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield f"\n\tGenerated {num_tokens} tokens from {BASE_MODEL} in {duration_s:.1f}s, throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"

    @exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
    ]
    answers = []

    for question in questions:
        print("Sending new request:", question, "\n\n")
        answer = ""
        for text in model.completion_stream.remote_gen(question):
            answer += text
        answers.append(answer)

    print(answers)
