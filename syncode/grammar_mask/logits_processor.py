import torch
from transformers import LogitsProcessor, PreTrainedTokenizer
from syncode.grammar_mask.grammar_constrainer import GrammarConstrainer
from syncode.mask_store.byte_tokenizer import ByteTokenizer
from syncode.parsers.grammars import Grammar
import logging
logger = logging.getLogger(__name__)


class SyncodeLogitsProcessor(LogitsProcessor):
    """
    This class is used to filter the logits of the model to only allow syntactically valid tokens for Python. 

    Args:
        grammar (str): The grammar to use for parsing e.g. "python".
        tokenizer (PreTrainedTokenizer): The tokenizer to use for decoding.
        use_cache (bool, optional): Whether to use the cache. Defaults to True.
        parse_output_only (bool, optional): Whether to parse the prompt. Defaults to False.
        num_samples (int, optional): The number of sequences to generate. Defaults to 1.
        dev_mode (bool, optional): Whether to run in development mode. Defaults to False.
        parser (str, optional): The parser to use. Defaults to 'lalr'.
        mode (str, optional): The mode to use. Defaults to 'grammar_mask'.
    """
    def __init__(self, 
        grammar: Grammar, 
        tokenizer: PreTrainedTokenizer, 
        use_cache=True,
        parse_output_only=True, 
        num_samples=1,
        dev_mode=False,
        parser='lalr',
        mode='grammar_mask',
        indent=False
        ):

        self.tokenizer = tokenizer
        self.byte_tokenizer = ByteTokenizer(tokenizer)
        
        # Create the grammar constrainer that handles most of the logic
        self.grammar_engine = GrammarConstrainer(
            grammar=grammar,
            tokenizer=tokenizer,
            byte_tokenizer=self.byte_tokenizer,
            use_cache=use_cache,
            parse_output_only=parse_output_only,
            batch_size=num_samples,
            dev_mode=dev_mode,
            parser=parser,
            mode=mode,
            indent=indent
        )

    def reset(self):
        """
        Resets the decoder state on every new prompt.
        """
        self.grammar_engine.reset()

    def is_valid(self, input_ids: torch.LongTensor, next_token: torch.LongTensor) -> bool:
        """
        Check if the next token is valid given the input_ids.
        """
        return self.grammar_engine.is_valid(input_ids, next_token)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Filter the scores to only allow syntactically valid tokens.
        """
        return self.grammar_engine.mask_scores(input_ids, scores)
    