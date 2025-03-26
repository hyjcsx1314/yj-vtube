from typing import AsyncIterator, Tuple, Callable, List
from functools import wraps
from .output_types import Actions, SentenceOutput, DisplayText
from ..utils.tts_preprocessor import tts_filter as filter_text
from ..live2d_model import Live2dModel
from ..config_manager import TTSPreprocessorConfig
from ..utils.sentence_divider import SentenceDivider
from ..utils.sentence_divider import SentenceWithTags, TagState
from loguru import logger


def sentence_divider(
    faster_first_response: bool = True,
    segment_method: str = "pysbd",
    valid_tags: List[str] = None,
):
    """
    Decorator that transforms token stream into sentences with tags

    Args:
        faster_first_response: bool - Whether to enable faster first response
        segment_method: str - Method for sentence segmentation
        valid_tags: List[str] - List of valid tags to process
    """

    def decorator(
        func: Callable[..., AsyncIterator[str]],
    ) -> Callable[..., AsyncIterator[SentenceWithTags]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncIterator[SentenceWithTags]:
            logger.info("Starting sentence_divider wrapper")
            divider = SentenceDivider(
                faster_first_response=faster_first_response,
                segment_method=segment_method,
                valid_tags=valid_tags or [],
            )
            token_stream = func(*args, **kwargs)
            logger.info("Got token stream, starting to process")
            async for sentence in divider.process_stream(token_stream):
                logger.debug(f"sentence_divider yielding: {sentence}")
                yield sentence

        return wrapper

    return decorator


def actions_extractor(live2d_model: Live2dModel):
    """
    Decorator that extracts actions from sentences
    """

    def decorator(
        func: Callable[..., AsyncIterator[SentenceWithTags]],
    ) -> Callable[..., AsyncIterator[Tuple[SentenceWithTags, Actions]]]:
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Tuple[SentenceWithTags, Actions]]:
            logger.info("Starting actions_extractor wrapper")
            sentence_stream = func(*args, **kwargs)
            async for sentence in sentence_stream:
                logger.debug(f"Processing sentence: {sentence.text}")
                actions = Actions()
                # Only extract emotions for non-tag text
                if not any(
                    tag.state in [TagState.START, TagState.END] for tag in sentence.tags
                ):
                    logger.debug("Extracting emotions")
                    expressions = live2d_model.extract_emotion(sentence.text)
                    if expressions:
                        logger.debug(f"Found expressions: {expressions}")
                        actions.expressions = expressions
                    else:
                        logger.debug("No expressions found")
                else:
                    logger.debug("Skipping emotion extraction due to tags")
                logger.debug(f"Yielding sentence with actions: {sentence.text}, {actions}")
                yield sentence, actions
            logger.info("Finished actions_extractor wrapper")

        return wrapper

    return decorator


def display_processor():
    """
    Decorator that processes text for display.
    """

    def decorator(
        func: Callable[..., AsyncIterator[Tuple[SentenceWithTags, Actions]]],
    ) -> Callable[..., AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]]:
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]:
            logger.info("Starting display_processor wrapper")
            stream = func(*args, **kwargs)

            async for sentence, actions in stream:
                logger.debug(f"Processing sentence: {sentence.text} with actions: {actions}")
                text = sentence.text
                # Handle think tag states
                for tag in sentence.tags:
                    logger.debug(f"Processing tag: {tag}")
                    if tag.name == "think":
                        if tag.state == TagState.START:
                            text = "("
                            logger.debug("Added opening parenthesis for think tag")
                        elif tag.state == TagState.END:
                            text = ")"
                            logger.debug("Added closing parenthesis for think tag")

                display = DisplayText(text=text)  # Simplified DisplayText creation
                logger.debug(f"Created display text: {display.text}")
                yield sentence, display, actions
            logger.info("Finished display_processor wrapper")

        return wrapper

    return decorator


def tts_filter(
    tts_preprocessor_config: TTSPreprocessorConfig = None,
):
    """
    Decorator that filters text for TTS.
    Skips TTS for think tag content.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]
        ],
    ) -> Callable[..., AsyncIterator[SentenceOutput]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncIterator[SentenceOutput]:
            logger.info("Starting tts_filter wrapper")
            sentence_stream = func(*args, **kwargs)
            config = tts_preprocessor_config or TTSPreprocessorConfig()
            logger.debug(f"Using TTS config: {config}")

            async for sentence, display, actions in sentence_stream:
                logger.debug(f"Processing sentence: {sentence.text} with display: {display.text}")
                if any(tag.name == "think" for tag in sentence.tags):
                    logger.debug("Found think tag, skipping TTS")
                    tts = ""
                else:
                    logger.debug("Processing text for TTS")
                    tts = filter_text(
                        text=display.text,
                        remove_special_char=config.remove_special_char,
                        ignore_brackets=config.ignore_brackets,
                        ignore_parentheses=config.ignore_parentheses,
                        ignore_asterisks=config.ignore_asterisks,
                        ignore_angle_brackets=config.ignore_angle_brackets,
                    )
                    logger.debug(f"Filtered TTS text: {tts}")

                logger.debug(f"[{display.name}] display: {display.text}")
                logger.debug(f"[{display.name}] tts: {tts}")

                output = SentenceOutput(
                    display_text=display,
                    tts_text=tts,
                    actions=actions,
                )
                logger.debug(f"Created sentence output: {output}")
                yield output
            logger.info("Finished tts_filter wrapper")

        return wrapper

    return decorator
