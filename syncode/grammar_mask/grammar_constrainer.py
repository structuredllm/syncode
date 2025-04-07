from typing import Optional, Tuple
import torch
import re
from transformers import PreTrainedTokenizer
from syncode.mask_store.byte_tokenizer import ByteTokenizer
from syncode.parse_result import AcceptSequence, RemainderState
from syncode.parsers.incremental_parser import IncrementalParser, ParseResult
from syncode.parsers import create_parser, create_base_parser
from syncode.mask_store.mask_store import MaskStore
from syncode.parsers.grammars import Grammar
import logging
logger = logging.getLogger(__name__)

class GrammarConstrainer:
    """
    Core class for constraining LLM token generation based on formal grammar rules.
    
    This class handles the parsing of generated code, validates its grammatical correctness,
    and creates token masks to ensure syntactically valid generations.
    
    The class supports two primary operating modes:
    
    1. `grammar_mask` (Conservative/Overapproximation): 
       This mode is more permissive and overapproximates the set of acceptable tokens.
       It allows a wider range of tokens that might be syntactically valid given the
       limited lookahead of the parser. This mode preserves more of the LLM's original 
       token distribution while still enforcing basic syntactic correctness.
       
    2. `grammar_strict` (Strict/Underapproximation):
       This mode is stricter and underapproximates the set of acceptable tokens.
       It enforces tighter grammatical constraints and may be more invasive in the
       LLM's generation process. It sometimes breaks LLM tokens that would have been
       syntactically correct when considered as a whole, potentially affecting the
       fluency or accuracy of generation.
    
    Example illustrating the difference:
    Consider generating Python code with the partial input: `def calculate`
    
    In `grammar_mask` mode, it might allow tokens like:
    - "(num" (combining opening parenthesis and parameter name as one token)
    
    In `grammar_strict` mode, it would force separate tokens:
    - "(" followed by "num" (requiring two separate token generations)
    
    For more details on the approximation methods, refer to the SynCode paper:
    https://arxiv.org/abs/2403.01632


    start_delim (str, optional): Start delimiter marking the beginning of structured 
        (grammar-constrained) content.
        end_delim (str, optional): End delimiter marking the end of structured content. 
    
        NOTE: These delimiters are used to extract structured regions for parsing and grammar enforcement. 
        See *CRANE: Reasoning with Constrained LLM Generation* 
        ([arXiv:2502.09061](https://arxiv.org/abs/2502.09061)) for more details.
        Example: `start_delim="```python\n"` and `end_delim="```"` would parse only 
        the content between these markers.
    """
    def __init__(
        self, 
        grammar: Grammar,
        tokenizer: PreTrainedTokenizer,
        byte_tokenizer: ByteTokenizer,
        use_cache=True,
        parse_output_only=True,
        batch_size=1,
        dev_mode=False,
        parser='lalr',
        mode='grammar_mask',
        start_delim=None,
        end_delim=None,
        ):
        
        self.tokenizer = tokenizer
        self.byte_tokenizer = byte_tokenizer
        self.grammar = grammar
        self.dev_mode = dev_mode
        self.batch_size = batch_size
        self.parse_failed = False

        # For backtracking to syntactically valid completions
        self.last_valid_state = [0 for _ in range(self.batch_size)]
        self.function_ends = [None for _ in range(self.batch_size)]

        # We use this when only the LLM output is parsed and not (input+output)
        self.parse_output_only = parse_output_only
        self.start_from = None

        # Ignore whitespace tokens
        self._ignore_whitespace = self._get_ignore_whitespace(self.grammar)

        # Create parser
        self.inc_parser: IncrementalParser = create_parser(self.grammar, parser=parser, ignore_whitespace=self._ignore_whitespace)

        # Load dfa mask store with specified mode (grammar_mask or grammar_strict)
        self.dfa_mask_store = MaskStore.init_mask_store(
                                    grammar=self.grammar, 
                                    tokenizer=self.tokenizer, 
                                    use_cache=use_cache, 
                                    mode=mode,  # Controls approximation strategy for token masking
                                    )

        # Used for separating the structured content from the rest of the generated text
        # defaults to None, meaning no delimiters are used
        self.start_delim = start_delim
        self.end_delim = end_delim

    def reset(self):
        """
        Resets the decoder state on every new prompt.
        """
        self.last_valid_state = [0 for _ in range(self.batch_size)]
        self.function_ends = [None for _ in range(self.batch_size)]
        self.parse_failed = False
        self.start_from = None
        self.inc_parser.reset()

    def _set_start_from(self, input_ids):
        """
        Sets the starting point for parsing based on whether we're parsing only the output or the full input+output.
        """
        if self.start_from is None:
            if self.parse_output_only:
                self.start_from = input_ids.size(1)
            else:
                self.start_from = 0

    def is_valid(self, input_ids: torch.LongTensor, next_token: torch.LongTensor) -> bool:
        """
        Check if the next token is valid given the input_ids.

        Args:
            input_ids (torch.LongTensor): The input ids.
            next_token (torch.LongTensor): The next token.

        Returns:
            bool: True if the next token is valid, False otherwise.
        """
        assert len(input_ids) == 1, "Only one input is supported for now."
        self._set_start_from(input_ids)

        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
        partial_output, remainder_bytes = self._get_partial_outputs(input_ids)[0]

        res, skip = self._parse_partial_output(
            idx=0, 
            partial_output=partial_output, 
            remainder_bytes=remainder_bytes, 
            accepted_generation=False
            )
        
        if skip: return False
        
        if input_ids[0, -1] == self.tokenizer.eos_token_id:
            # Do not allow the model to generate EOS token until $END in the grammar is reached
            return AcceptSequence(['$END']) in res.accept_sequences
        
        if res.remainder_state == RemainderState.COMPLETE or res.remainder_state == RemainderState.MAYBE_COMPLETE:
            is_valid = True

        # Check if the remainder is a valid prefix for the last terminal
        is_valid = self.dfa_mask_store.is_valid_prefix(res)

        if is_valid:
            self._update_valid_state(partial_output, 0, res)

        return is_valid

    def mask_scores(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:    
        """
        Mask scores by zeroing out invalid next tokens based on grammar constraints.
        
        The exact behavior depends on whether we're using grammar_mask mode (conservative/
        overapproximation) or grammar_strict mode (strict/underapproximation). In both cases,
        tokens that would lead to definitely invalid syntax are masked out by setting their 
        scores to negative infinity.
        
        Args:
            input_ids (torch.LongTensor): The input ids.
            scores (torch.FloatTensor): The scores to be masked.
            
        Returns:
            torch.FloatTensor: The masked scores.
        """
        self._set_start_from(input_ids) # start_from is used for choosing where the parsing should start
        partial_outputs = self._get_partial_outputs(input_ids)

        for idx, (partial_output, remainder_bytes) in enumerate(partial_outputs):
            # 1. Parsing
            res, skip = self._parse_partial_output(idx, partial_output, remainder_bytes, accepted_generation=True)
            if skip: continue

            # 2. Computing the accept mask
            accept_mask = self.dfa_mask_store.get_accept_mask(res)

            # 3. Masking the scores
            if torch.sum(accept_mask) != 0: # If there are acceptable tokens for the current partial code 
                if len(scores[idx]) > len(accept_mask):
                    # Pad accept_mask with 0 values. Since scores[i] may be longer than tokenizer vocab size, we need to pad accept_mask with 0 values
                    accept_mask = torch.cat((accept_mask, torch.zeros(len(scores[idx]) - len(accept_mask), dtype=torch.bool)))
                elif len(scores[idx]) < len(accept_mask):
                    accept_mask = accept_mask[: len(scores[idx])]
                scores[idx] = scores[idx].masked_fill(~accept_mask.to(scores.device), -float("inf"))
            else: # Otherwise, report the error and mask no tokens
                logger.debug('No acceptable tokens for the current partial code!')
                logger.debug(repr(res))

        return scores

    def _parse_partial_output(self, idx: int, partial_output: str, remainder_bytes: bytes, accepted_generation=True) -> tuple[ParseResult, bool]:
        """
        Parse the partial code and return the result.
        """
        skip = False
        res = None
        
        try: 
            res = self.inc_parser.get_acceptable_next_terminals(partial_output)

            if len(remainder_bytes) > 0:
                res.remainder_state = RemainderState.INCOMPLETE
                res.remainder = res.remainder.encode('utf-8') + remainder_bytes
            else:
                res.remainder = res.remainder.encode('utf-8')

            self._update_valid_state(partial_output, idx, res)
        except Exception as e:
            if self.dev_mode == True and accepted_generation:
                raise e
            elif self.parse_failed == False and accepted_generation:
                self.parse_failed = True
                logger.info("-"*50)
                logger.info(f"Parsing failed! Falling back to unconstrained decoding.\nException: {e}\nPartial code: {partial_output}\nParsed lexical tokens: {self.inc_parser.parsed_lexer_tokens}")
                logger.info("-"*50)
            skip = True
        return res, skip

    def _get_partial_outputs(self, input_ids: torch.LongTensor) -> list[(str, bytes)]:
        """
        Get the partial codes for the input_ids and return the remainder bytes if the partial code is not a valid UTF-8 string.
        """     
        output = []
        for idx in range(len(input_ids)):
            if self.parse_output_only:
                partial_output, remainder_bytes = self._bytes_to_string(
                    self.byte_tokenizer.decode(
                        input_ids[idx, self.start_from:].tolist(), skip_special_tokens=True)
                    )
            else:
                partial_output, remainder_bytes = self._bytes_to_string(
                    self.byte_tokenizer.decode(
                        input_ids[idx].tolist(), skip_special_tokens=True)
                    )
            
            # Use self.start_delim and self.end_delim to extract the structured content
            # It is possible that there are multiple start_delim and end_delim in the current input
            

            output.append((partial_output, remainder_bytes))
        return output

    @staticmethod
    def extract_last_structured_block(
        text: str,
        start_delim: Optional[str],
        end_delim: Optional[str]
    ) -> Tuple[str, bool]:
        """
        Extracts the last structured block from `text` between `start_delim` and `end_delim`.

        Returns:
            (extracted_text: str, should_constrain: bool)
                - extracted_text: The content of the last delimited block.
                - should_constrain: True if a start delimiter is present without a matching end,
                                    meaning structured generation should continue.
        """
        if start_delim is None or end_delim is None:
            return "", False

        # Find all fully enclosed blocks
        pattern = re.escape(start_delim) + r"(.*?)" + re.escape(end_delim)
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))

        if matches:
            last_match = matches[-1]
            return last_match.group(1).strip(), False  # closed, no need to constrain further

        # If there's a start but no end, check for unclosed start
        last_start_idx = text.rfind(start_delim)
        last_end_idx = text.rfind(end_delim)
        
        # If the start delimiter appears after the last end delimiter, it's an open block
        if last_start_idx > last_end_idx:
            # Return the content after the last start delimiter, even if the end delimiter is missing
            return text[last_start_idx + len(start_delim):].strip(), True

        return "", False  # no open block => no constraint needed
    
    
    def _update_valid_state(self, partial_output: str, idx: int, r: ParseResult):
        """
        This a simple heuristic to cut off the generated output at the end of the function. 
        TODO: Put this under a flag to enable/disable this heuristic.
        """
        if idx < len(self.function_ends):
            if r.function_end: # If the function end is not None, then the last valid state is the function end
                if self.function_ends[idx] is None: self.function_ends[idx] = []
                self.function_ends[idx].append(len(partial_output) - len(r.remainder))

        if idx < len(self.last_valid_state):
            for accept_seq in r.accept_sequences:
                # 'EOF' is special terminal since $END does not work with python
                if accept_seq[0] == '$END' or accept_seq[0] == 'EOF':
                    self.last_valid_state[idx] = len(partial_output) - len(r.remainder)

    @staticmethod
    def _bytes_to_string(byte_sequence: bytes) -> tuple[str, bytes]:
        """
        Convert a byte sequence into a UTF-8 string plus a remainder that is not valid UTF-8.
        
        This function finds the longest valid UTF-8 prefix of the input byte sequence,
        converts it to a Python string, and returns any remaining bytes that couldn't be decoded.
        
        Args:
            byte_sequence: The input byte sequence
            
        Returns:
            A tuple (string, remainder) where:
            - string is the longest valid UTF-8 prefix of the input as a Python string
            - remainder is the rest of the bytes that could not be decoded as UTF-8
        """
        if not isinstance(byte_sequence, bytes):
            raise TypeError("Input must be a bytes object")
        
        if not byte_sequence:
            return "", b""
        
        # Try to decode the entire sequence first - common case optimization
        try:
            return byte_sequence.decode('utf-8'), b""
        except UnicodeDecodeError:
            pass
        
        # Find the longest valid prefix by incrementally checking each additional byte
        valid_end = 0
        
        while valid_end < len(byte_sequence):
            try:
                # Try to decode up to the current position
                byte_sequence[:valid_end+1].decode('utf-8')
                valid_end += 1
            except UnicodeDecodeError:
                break
        
        # Return the valid prefix and remainder
        if valid_end > 0:
            return byte_sequence[:valid_end].decode('utf-8'), byte_sequence[valid_end:]
        else:
            return "", byte_sequence

    def _get_ignore_whitespace(self, grammar):
        """
        Check if the grammar allows whitespace tokens to be ignored.
        """
        base_parser = create_base_parser(grammar)
        terminals = base_parser.terminals
        ignore_terminals = base_parser.ignore_tokens
        
        import regex
        ignore_whitespace = False
        for ig_name in ignore_terminals:
            for terminal in terminals:
                if terminal.name == ig_name:
                    if regex.match(terminal.pattern.to_regexp(), ' ') is not None:
                        ignore_whitespace = True # convert to boolean tensor mask. This is useful for fast union operations
        return ignore_whitespace
    