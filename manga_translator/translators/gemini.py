import os
import re
import asyncio
import time
import json
from typing import List, Tuple, Generator, Optional, Dict, Any, Union

import aiohttp
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .config_gemini import ConfigGemini
from .keys import GEMINI_API_KEY, GEMINI_API_BASE


class GeminiTranslator(ConfigGemini, CommonTranslator):
    """
    Translation service using Google's Gemini API.
    
    This class handles translation of text using the Gemini generative AI model,
    including rate limiting, retries, and error handling.
    """
    
    _LANGUAGE_CODE_MAP: Dict[str, str] = VALID_LANGUAGES
    
    # Rate limiting and retry parameters
    _MAX_REQUESTS_PER_MINUTE: int = 60
    _TIMEOUT: int = 30
    _RETRY_ATTEMPTS: int = 2
    _TIMEOUT_RETRY_ATTEMPTS: int = 3
    _RATELIMIT_RETRY_ATTEMPTS: int = 3
    _MAX_TOKENS: int = 30720

    _ERROR_KEYWORDS: List[str] = [
        r"I must decline",
        r'(i(\'m| am)?\s+)?sorry(.|\n)*?(can(\'t|not)|unable to|cannot)\s+(assist|help)',
        r"抱歉，?我(无法[将把]?|不[能会]?)",
        r"申し訳ありませんが",
    ]

    def __init__(self, check_gemini_key: bool = True) -> None:
        """
        Initialize the Gemini translator.
        
        Args:
            check_gemini_key: If True, verify that the Gemini API key is available
        
        Raises:
            MissingAPIKeyException: If the Gemini API key is not available
        """
        _CONFIG_KEY: str = 'gemini'
        ConfigGemini.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)

        if not GEMINI_API_KEY and check_gemini_key:
            raise MissingAPIKeyException('GEMINI_API_KEY environment variable required')

        self.api_base: str = GEMINI_API_BASE or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key: str = GEMINI_API_KEY
        self.model_name: str = "gemini-2.0-flash"
        self.token_count: int = 0
        self.token_count_last: int = 0
        self._last_request_ts: float = 0

    def _cannot_assist(self, response: str) -> bool:
        """
        Check if the response indicates that the model refused to assist.
        
        Args:
            response: The response text from the model
            
        Returns:
            True if the response contains an error or refusal pattern, False otherwise
        """
        resp_lower: str = response.strip().lower()
        for kw in self._ERROR_KEYWORDS:
            if re.search(kw, resp_lower, re.IGNORECASE):
                return True
        return False

    async def _ratelimit_sleep(self) -> None:
        """
        Sleep if necessary to respect rate limits.
        
        This method ensures that requests to the API don't exceed
        the maximum requests per minute by introducing delays between calls.
        """
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now: float = time.time()
            delay: float = 60.0 / self._MAX_REQUESTS_PER_MINUTE
            elapsed: float = now - self._last_request_ts
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._last_request_ts = time.time()

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]) -> Generator[Tuple[str, int], None, None]:
        """
        Assemble prompts for the Gemini API by batching queries.
        
        Args:
            from_lang: Source language code
            to_lang: Target language code
            queries: List of text strings to translate
            
        Yields:
            Tuple containing the assembled prompt and the number of queries in the batch
        """
        MAX_CHAR_PER_PROMPT: int = self._MAX_TOKENS * 3
        chunk_queries: List[List[str]] = []
        current_length: int = 0
        batch: List[str] = []

        for q in queries:
            if current_length + len(q) + 10 > MAX_CHAR_PER_PROMPT and batch:
                chunk_queries.append(batch)
                batch = []
                current_length = 0
            batch.append(q)
            current_length += len(q) + 10
        if batch:
            chunk_queries.append(batch)

        system_prompt: str = self.chat_system_template.format(to_lang=to_lang)
        for this_batch in chunk_queries:
            prompt: str = system_prompt
            for i, query in enumerate(this_batch):
                prompt += f"\n<|{i+1}|>{query}"
            yield prompt.lstrip(), len(this_batch)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        """
        Translate a list of queries from one language to another.
        
        Args:
            from_lang: Source language code
            to_lang: Target language code
            queries: List of text strings to translate
            
        Returns:
            List of translated text strings in the same order as the input queries
        """
        translations: List[str] = [''] * len(queries)
        idx_offset: int = 0

        for prompt, batch_size in self._assemble_prompts(from_lang, to_lang, queries):
            batch_queries: List[str] = queries[idx_offset:idx_offset + batch_size]
            indices: List[int] = list(range(idx_offset, idx_offset + batch_size))

            success, partial_results = await self._translate_batch(
                from_lang, to_lang, batch_queries, indices, prompt
            )
            for i, r in zip(indices, partial_results):
                translations[i] = r

            idx_offset += batch_size

        return translations

    async def _translate_batch(
        self, 
        from_lang: str, 
        to_lang: str, 
        batch_queries: List[str], 
        batch_indices: List[int], 
        prompt: str, 
        split_level: int = 0
    ) -> Tuple[bool, List[str]]:
        """
        Translate a batch of queries with retries and fallback strategies.
        
        Args:
            from_lang: Source language code
            to_lang: Target language code
            batch_queries: List of queries to translate in this batch
            batch_indices: List of original indices for this batch
            prompt: The assembled prompt for this batch
            split_level: Current recursion level for batch splitting (for fallback)
            
        Returns:
            Tuple containing success flag and list of translations
        """
        partial_results: List[str] = [''] * len(batch_queries)
        if not batch_queries:
            return True, partial_results

        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                response_text: str = await self._request_with_retry(to_lang, prompt)
                new_translations: List[str] = re.split(r'<\|\d+\|>', response_text)

                if not new_translations[0].strip():
                    new_translations = new_translations[1:]

                if self._cannot_assist(response_text):
                    self.logger.warning(f"Refusal detected, retrying (attempt {attempt+1})")
                    continue

                if len(new_translations) < len(batch_queries):
                    self.logger.warning(f"Incomplete response, retrying (attempt {attempt+1})")
                    continue

                new_translations = [t.strip() for t in new_translations]
                if any(not t for t in new_translations):
                    self.logger.warning(f"Empty translation detected, retrying (attempt {attempt+1})")
                    continue

                for i in range(len(batch_queries)):
                    partial_results[i] = new_translations[i]

                self.logger.info(f"Batch of {len(batch_queries)} translated (attempt {attempt+1})")
                return True, partial_results

            except Exception as e:
                self.logger.warning(f"Batch failed: {str(e)} (attempt {attempt+1})")
                if attempt < self._RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(1)

        # Fallback: Split the batch and try again with smaller batches
        if split_level < 2 and len(batch_queries) > 1:
            self.logger.warning(f"Splitting batch of {len(batch_queries)} at level {split_level}")
            mid: int = len(batch_queries) // 2
            left_queries: List[str] = batch_queries[:mid]
            right_queries: List[str] = batch_queries[mid:]
            left_indices: List[int] = batch_indices[:mid]
            right_indices: List[int] = batch_indices[mid:]

            left_prompt, _ = next(self._assemble_prompts(from_lang, to_lang, left_queries))
            left_success, left_results = await self._translate_batch(
                from_lang, to_lang, left_queries, left_indices, left_prompt, split_level + 1
            )

            right_prompt, _ = next(self._assemble_prompts(from_lang, to_lang, right_queries))
            right_success, right_results = await self._translate_batch(
                from_lang, to_lang, right_queries, right_indices, right_prompt, split_level + 1
            )

            return (left_success and right_success), (left_results + right_results)

        self.logger.error(f"Translation failed after retries, returning originals")
        return False, batch_queries

    async def _request_with_retry(self, to_lang: str, prompt: str) -> str:
        """
        Make an API request with timeout and rate-limit handling.
        
        Args:
            to_lang: Target language code 
            prompt: The prompt to send to the Gemini API
            
        Returns:
            Response text from the Gemini API
            
        Raises:
            TimeoutError: If the request times out repeatedly
            Exception: For other API errors after retries
        """
        timeout_attempt: int = 0
        ratelimit_attempt: int = 0

        while True:
            await self._ratelimit_sleep()
            try:
                response_text: str = await asyncio.wait_for(
                    self._request_translation(to_lang, prompt),
                    timeout=self._TIMEOUT
                )
                return response_text

            except asyncio.TimeoutError:
                timeout_attempt += 1
                if timeout_attempt > self._TIMEOUT_RETRY_ATTEMPTS:
                    raise TimeoutError(f"Gemini request timed out after {self._TIMEOUT_RETRY_ATTEMPTS} attempts")
                self.logger.warning(f"Timeout, retrying (attempt {timeout_attempt})")
                await asyncio.sleep(1)

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    ratelimit_attempt += 1
                    if ratelimit_attempt > self._RATELIMIT_RETRY_ATTEMPTS:
                        raise
                    self.logger.warning(f"Rate limit hit, retrying (attempt {ratelimit_attempt})")
                    await asyncio.sleep(2)
                else:
                    raise

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        """
        Make a direct request to the Gemini API.
        
        Args:
            to_lang: Target language code
            prompt: The prompt to send to the API
            
        Returns:
            Text response from the Gemini API
            
        Raises:
            Exception: For API errors
        """
        url: str = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        generation_config: Dict[str, Any] = self.generation_config

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": generation_config.get("temperature", 0.7),
                "topP": generation_config.get("top_p", 0.95),
                "topK": generation_config.get("top_k", 40),
                "maxOutputTokens": generation_config.get("max_output_tokens", 1024),
                "stopSequences": generation_config.get("stop_sequences", ["</translation>"])
            },
            "safetySettings": self.safety_config or [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text: str = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error_text}")

                result: Dict[str, Any] = await response.json()
                try:
                    text: str = result["candidates"][0]["content"]["parts"][0]["text"]
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                    self.logger.debug(f"Gemini response: {text}")
                    return text.strip()
                except (KeyError, IndexError) as e:
                    raise Exception(f"Failed to parse Gemini response: {str(e)} - {json.dumps(result)}")