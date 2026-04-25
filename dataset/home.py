# dataset/home.py

from deep_translator import GoogleTranslator


def translator(text, src_code, tgt_code):
    """
    External fallback translator.
    Used when MarianMT fails or is slow.
    """

    result = GoogleTranslator(
        source=src_code,
        target=tgt_code
    ).translate(text)

    return result