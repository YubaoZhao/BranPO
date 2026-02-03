import asyncio
import logging
import os
import uuid
from typing import List, Optional, Union

try:
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Please install vllm and transformers: pip install vllm transformers")

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason


class vllmEngine(RolloutEngine):
    def __init__(
        self,
        model: str = "",
        tokenizer=None,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        tools: list[Tool | dict] = None,
        accumulate_reasoning: bool = False,
        **kwargs
    ):
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = (
            max_model_length - 1
            if max_model_length is not None
            else max_prompt_length + max_response_length - 1
        )
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning

        logging.info(f"Initializing Async vLLM Engine with model: {self.model}")

        # Extract vLLM specific arguments from kwargs
        # AsyncEngineArgs handles most configuration (tensor_parallel, gpu_memory, etc.)
        engine_args_dict = {
            "model": self.model,
            "max_model_len": self.max_model_length,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "data_parallel_size": kwargs.get("data_parallel_size", 1),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.90),
            "dtype": kwargs.get("dtype", "auto"),
            "enforce_eager": kwargs.get("enforce_eager", False),
            # "disable_log_requests": True, 
        }
        
        # Add any other valid AsyncEngineArgs from kwargs
        valid_args = AsyncEngineArgs.__init__.__code__.co_varnames
        for k, v in kwargs.items():
            if k in valid_args and k not in engine_args_dict:
                engine_args_dict[k] = v

        engine_args = AsyncEngineArgs(**engine_args_dict)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Handle tokenizer
        # Unlike synchronous LLM class, AsyncLLMEngine doesn't expose the tokenizer instance directly
        # in a public API easily suitable for external parsing logic. We load a lightweight version.
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, 
                trust_remote_code=kwargs.get("trust_remote_code", True)
            )
        else:
            self.tokenizer = tokenizer

        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False)
        )
        
        # Since we have direct access to the tokenizer and engine, we act as a "completion" endpoint
        # that handles chat templating locally.
        self._use_chat_completions = False 

    def _prepare_sampling_params(self, sampling_params: dict, prompt_length: int = None) -> SamplingParams:
        """Convert dict parameters to vLLM SamplingParams."""
        params = sampling_params.copy()
        
        # Handle max_tokens
        max_tokens = params.pop("max_tokens", params.pop("max_new_tokens", self.max_response_length))
        if "max_completion_tokens" in params:
             max_tokens = params.pop("max_completion_tokens")

        if prompt_length and self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining

        # Clean up OpenAI specific params that aren't in vLLM
        params.pop("model", None) 
        
        return SamplingParams(max_tokens=max_tokens, **params)

    async def chat_completion(self, messages: list[dict]=None, request_prompt_ids: list[int]=None, **kwargs) -> ModelOutput:
        """
        Route chat requests to completion via template parsing.
        """
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        if request_prompt_ids is None:
            prompt = self.chat_parser.parse(
                messages, 
                add_generation_prompt=True, 
                is_first_msg=True, 
                tools=tools, 
                accumulate_reasoning=accumulate_reasoning
            )
            return await self.completion(prompt=prompt, **kwargs)
        else:
            return await self.completion(prompt_ids=request_prompt_ids, **kwargs)

    async def completion(self, prompt: str=None, prompt_ids=None, **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params_dict = self.sampling_params.copy()
        sampling_params_dict.update(kwargs)

        # Encode prompt to check length
        if prompt_ids is None:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        vllm_sampling_params = self._prepare_sampling_params(sampling_params_dict, prompt_length)
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        try:
            # AsyncLLMEngine.generate returns an AsyncIterator
            results_generator = self.engine.generate(
                prompt={"prompt_token_ids": prompt_ids}, # We use prompt_token_ids usually for better control, or prompt string
                sampling_params=vllm_sampling_params,
                request_id=request_id,
            )

            final_output = None
            
            # Iterate through the stream. 
            # Since we are not streaming back to the user in this interface, 
            # we just wait for the final result.
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None:
                raise Exception("No output generated from vLLM engine.")

            # Process the final output
            # vLLM returns RequestOutput objects
            output_text = final_output.outputs[0].text
            completion_ids = list(final_output.outputs[0].token_ids)
            finish_reason = final_output.outputs[0].finish_reason
            
            # Use RLLM parser to extract reasoning/content/tools
            parsed_output = self.chat_parser.parse_completion(completion_ids)

            return ModelOutput(
                text=output_text,
                content=parsed_output["content"],
                reasoning=parsed_output["reasoning"],
                tool_calls=parsed_output["tool_calls"],
                prompt_ids=list(final_output.prompt_token_ids),
                completion_ids=completion_ids,
                prompt_length=len(final_output.prompt_token_ids),
                completion_length=len(completion_ids),
                finish_reason=finish_reason,
            )

        except Exception as e:
            # Ensure we don't leave hanging requests if something breaks
            try:
                await self.engine.abort(request_id)
            except:
                pass 
            raise Exception(f"Error during vLLM Async inference: {e}") from e

    async def get_model_response(self, messages: list[dict]=None, request_prompt_ids: list[int]=None, **kwargs) -> ModelOutput:
        """
        Main entry point using chat parsing + async completion.
        """
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        
        if request_prompt_ids is None:
            prompt = self.chat_parser.parse(
                messages, 
                add_generation_prompt=True, 
                is_first_msg=True, 
                tools=tools, 
                accumulate_reasoning=accumulate_reasoning
            )
            return await self.completion(prompt, **kwargs)
        else:
            return await self.completion(prompt_ids=request_prompt_ids, **kwargs)