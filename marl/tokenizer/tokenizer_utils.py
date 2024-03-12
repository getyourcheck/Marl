from typing import Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
)
from marl.logger import init_logger

logger = init_logger(__name__)

PADDING_SIDE = "left"  # TODO


def get_tokenizer(
    tokenizer_name: str,
    *args,
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    padding_side: Optional[str] = PADDING_SIDE,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            padding_side=padding_side,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = "Failed to load the tokenizer. Try `trust_remote_code=True`."
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except OSError as e:
        if "Incorrect path_or_model_id" in str(e):  # e.g., v13.model
            tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                padding_side=padding_side,
                **kwargs,
            )
            logger.warning("Using LlamaTokenizer.")
        else:
            raise e
    except AttributeError as e:
        raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer
