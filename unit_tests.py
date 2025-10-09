import json
import unittest
from pathlib import Path
from typing import List
from unittest.mock import Mock

from dotenv import load_dotenv
from manga_translator.config import Config, Translator
from manga_translator.translators import get_translator
from manga_translator.translators.common import OfflineTranslator
from manga_translator.manga_translator import MangaTranslator

load_dotenv("../../config/env.gpu", override=False)

ARABIC_RANGES = [(0x0600, 0x06FF), (0xFE70, 0xFEFF), (0xFB50, 0xFDFF)]
REFUSAL_KEYWORDS = ["sorry", "cannot", "unable", "decline", "refuse"]

def is_arabic_text(text: str) -> bool:
    """Check if text contains Arabic characters"""
    return any(start <= ord(char) <= end for char in text 
               for start, end in ARABIC_RANGES)

def has_refusal_keywords(text: str) -> bool:
    """Check if text contains refusal keywords"""
    return any(word in text.lower() for word in REFUSAL_KEYWORDS)

def create_mock_regions(texts: List[str]) -> List[Mock]:
    """Create mock text regions for testing"""
    regions = []
    for text in texts:
        region = Mock()
        region.translation = text
        regions.append(region)
    return regions


class TestMangaTranslatorFallbacksAndArabicSkip(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup MangaTranslator
        self.translator_instance = MangaTranslator(params={
            "verbose": False,
            "use_gpu": True,
            "kernel_size": 3,
            "detection_size": 2048,
            "inpainting_size": 512,
        })

        # Setup translator
        self.translator = get_translator(Translator.chatgpt)
        if isinstance(self.translator, OfflineTranslator):
            await self.translator.load("auto", "ARA", "cuda")
        
        config = Config(translator={"translator": "chatgpt", "target_lang": "ARA"})
        if config.translator:
            self.translator.parse_args(config.translator)

        # Load bad words
        bad_words_path = Path(__file__).parent / "bad_words.json"
        with open(bad_words_path) as f:
            self.bad_words = json.load(f)

    async def test_translation_fallback_on_refusal(self):
        """Test that translation fallback handles ChatGPT refusals correctly"""
        results = await self.translator.translate("auto", "ARA", self.bad_words, False)
        
        arabic_found = any(is_arabic_text(r) for r in results if r)
        self.assertTrue(arabic_found, "Fallback should produce Arabic translations")
        
        refusals_found = any(has_refusal_keywords(r) for r in results if r)
        self.assertFalse(refusals_found, "Fallback should avoid refusal responses")

    async def test_arabic_language_skip_all_arabic(self):
        """Test Arabic skip when all texts are already in Arabic"""
        arabic_texts = [
            "مرحبا بك", "كيف حالك", "أهلا وسهلا", "شكرا لك", "مع السلامة",
            "صباح الخير", "مساء الخير", "تصبح على خير", "أراك لاحقا", "إلى اللقاء",
            "نعم", "لا", "من فضلك", "عفوا", "آسف", "أهلا مرة أخرى",
        ]
        regions = create_mock_regions(arabic_texts)
        result = await self.translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ARA", min_ratio=0.5
        )
        self.assertTrue(result, "Should skip when all texts are Arabic")

    async def test_arabic_language_skip_no_arabic(self):
        """Test Arabic skip when no texts are in Arabic"""
        non_arabic_texts = [
            "Hello world", "Good morning", "Thank you", "How are you?", "Have a nice day",
            "Test phrase", "Another test", "English text", "No Arabic", "Simple sentence",
            *[f"Extra text{i}" for i in range(1, 7)]
        ]
        regions = create_mock_regions(non_arabic_texts)
        result = await self.translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ARA", min_ratio=0.5
        )
        self.assertTrue(result, "Should skip for target_lang='ARA' even with no Arabic")

    async def test_arabic_language_skip_mixed(self):
        """Test Arabic skip with mixed Arabic and non-Arabic texts"""
        mixed_texts = [
            "Hello world", "مرحبا بك", "Good morning", "كيف حالك", "Thank you",
            "أهلا وسهلا", "Another test", "شكرا لك", "English text", "مع السلامة",
            "No Arabic", "صباح الخير", "Simple sentence", "مساء الخير", "Extra text",
            "تصبح على خير"
        ]
        regions = create_mock_regions(mixed_texts)
        result = await self.translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ARA", min_ratio=0.5
        )
        self.assertTrue(result, "Should skip for target_lang='ARA' with mixed texts")

    async def test_language_check_non_arabic_target_matching(self):
        """Test language check for non-Arabic target with matching content"""
        english_texts = [
            "Hello world", "Good morning", "Thank you", "How are you?", "Have a nice day",
            "Test phrase", "Another test", "English text", "No Arabic", "Simple sentence",
            *[f"Extra text{i}" for i in range(1, 7)]
        ]
        regions = create_mock_regions(english_texts)
        result = await self.translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ENG", min_ratio=0.5
        )
        self.assertTrue(result, "Should skip when content matches target lang")

    async def test_language_check_non_arabic_target_non_matching(self):
        """Test language check for non-Arabic target with non-matching content"""
        arabic_texts = [
            "مرحبا بك", "كيف حالك", "أهلا وسهلا", "شكرا لك", "مع السلامة",
            "صباح الخير", "مساء الخير", "تصبح على خير", "أراك لاحقا", "إلى اللقاء",
            "نعم", "لا", "من فضلك", "عفوا", "آسف", "أهلا مرة أخرى",
        ]
        regions = create_mock_regions(arabic_texts)
        result = await self.translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ENG", min_ratio=0.5
        )
        self.assertFalse(result, "Should not skip when content doesn't match target")

    async def test_normal_translation(self):
        """Test normal translation with safe content"""
        normal_words = [
            "Hello world", "Good morning", "Thank you very much",
            "How are you today?", "Have a nice day",
        ]
        results = await self.translator.translate("auto", "ARA", normal_words, False)
        
        success_count = sum(1 for r in results if r and r.strip() and is_arabic_text(r))
        self.assertGreater(success_count, 0, "Should translate safe content to Arabic")


if __name__ == "__main__":
    unittest.main()
    