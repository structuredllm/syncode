from collections import defaultdict
import os, pickle
import time
import torch
import regex
import syncode.common as common
from tqdm import tqdm
from syncode.mask_store.byte_tokenizer import ByteTokenizer
from syncode.mask_store.fsm_set import JointFSMState, FSMSet
from syncode.mask_store.lookup_table import LookupTable
from syncode.parsers import create_base_parser
from syncode.larkm.lexer import TerminalDef
from syncode.parse_result import RemainderState, ParseResult
from syncode.parsers.grammars.grammar import Grammar
from typing import Iterable, Union
from transformers import PreTrainedTokenizer
import logging
logger = logging.getLogger(__name__)


class MaskStore:
    """
    We build an DFA that consists of DFAs for each terminal. We simulate the DFA by consuming the input string for each terminal DFA.

    There are 3 possible cases for the remainder string:

    1. COMPLETE: In this case, the last token is complete (and cannot be further extended) and we know the type of next terminal. Thus, we need to compute all tokens such that consuming the token leads to a live state for the terminal DFA or it reaches a final state for the terminal DFA.

    2. INCOMPLETE: In this case, the remainder is incomplete and does not match any terminal regex. Thus, we need to compute all tokens such that consuming the token leads to a live state for the current terminal DFA or again it reaches a final state for the current terminal DFA at some point.

    3. MAYBE_COMPLETE: In this case the remainder matches a type of terminal. It may happen that we add to the same matched part of the remainder. In that case, there are two possibilities. i) the matched terminal type does not change and thus we can use the next terminal set computed by assuming that. ii) the matched terminal type changes and then we do not know the next terminal set. Thus, we need to compute all tokens such that consuming the token leads to a live state for the current terminal DFA or again it reaches a final state for the current terminal DFA at some point.
    """
    def __init__(self, 
                 terminals: Iterable[TerminalDef], 
                 tokenizer: PreTrainedTokenizer, 
                 simplifications: dict={}, 
                 indent: bool=False,
                 mode='grammar_strict', # 'grammar_strict' or 'grammar_mask'
                 ignore_terminals: Iterable[str]=[],
                 parse_table=None
                 ):
        self._vocab = MaskStore._get_vocab_from_tokenizer(tokenizer)
        self._mode = mode

        # Tokenizer for byte tokens
        self.byte_tokenizer = ByteTokenizer(tokenizer)
        self.special_token_ids = tokenizer.all_special_ids 
        self.eos_token_id = tokenizer.eos_token_id

        # Create the FSMs for each terminal
        self._fsms = FSMSet(terminals, simplifications)

        # Check if whitespace is in ignore terminals
        self._ignore_whitespace = self.set_ignore_whitespace(terminals, ignore_terminals)
        logger.info(f"Ignore whitespace is {self._ignore_whitespace}")
        
        # Iterate through each pair of DFA state and next terminals and store the overapproximate tokens
        self._lookup_table = LookupTable(
            self._vocab, 
            eos_token_id=self.eos_token_id,
            special_token_ids=self.special_token_ids, 
            indent=indent, mode=mode
            )
        terminal_names = [terminal.name for terminal in terminals]

        followings_terminas_map = None
        if parse_table is not None:
            followings_terminas_map = self._compute_following_terminals_map(terminal_names, parse_table, ignore_terminals)

        # Create consume prefix cache
        self._consume_prefix_cache = {}
        self._store_token_masks(terminal_names, len(self._vocab), followings_terminas_map)

        self.indentation = indent       

        # NOTE: This should be called at the end of the constructor
        self._lookup_table.convert_lookups_from_list_to_mask() 

    def set_ignore_whitespace(self, terminals: Iterable[TerminalDef], ignore_terminals: Iterable[str]) -> bool:
        ignore_whitespace = False
        for ig_name in ignore_terminals:
            for terminal in terminals:
                if terminal.name == ig_name:
                    if regex.match(terminal.pattern.to_regexp(), ' ') is not None:
                        ignore_whitespace = True # convert to boolean tensor mask. This is useful for fast union operations
        return ignore_whitespace

    @staticmethod
    def init_mask_store(
        grammar: Grammar, 
        tokenizer, 
        use_cache=True, 
        mode='grammar_strict',
        indent=False
    ):
        '''
        Loads the fsm for the given language and tokenizer. If the fsm is not cached, it is created and cached. 
        '''
        tokenizer_name = type(tokenizer).__name__
        fsm_dir = common.SYNCODE_CACHE + 'mask_stores/' + tokenizer_name + '/'
        grammar_hash = grammar.hash()

        # TODO: Hashing using the tokenizer vocab size, this may be problmatic if we have two fine-tuned models with same tokenizer, same vocab size but different vocab
        fsm_path = f'{fsm_dir}{mode}_{grammar_hash}_{tokenizer.vocab_size}.pkl'
        
        if use_cache and os.path.exists(fsm_path):
            try:
                with open(fsm_path, 'rb') as f:
                    mask_store = pickle.load(f)
                    return mask_store
            except Exception as e:
                logger.warning(f"Error loading mask store: {e}")
                
        logger.info(f"Using cache: {use_cache} and fsm path {fsm_path} exist: {os.path.exists(fsm_path)}")
        logger.info(f"Creating mask store for {tokenizer_name} and {grammar}, may take more than 10 minutes. Caching at {os.path.abspath(fsm_path)}.")
        base_parser = create_base_parser(grammar)
        simplifications = grammar.simplifications()
        os.makedirs(fsm_dir, exist_ok=True)

        start_time = time.time()
        mask_store = MaskStore(
            base_parser.terminals, 
            tokenizer, 
            simplifications=simplifications, 
            mode=mode, 
            ignore_terminals=base_parser.ignore_tokens, 
            parse_table=base_parser.parser.parser._parse_table,
            indent=indent
            )
        logger.info(f"Time taken to create mask store: {time.time() - start_time:.2f} seconds")

        pickle.dump(mask_store, open(fsm_path, 'wb'))
        return mask_store

    def _compute_following_terminals_map(
            self, 
            terminals: Iterable[str], 
            parse_table,
            ignore_terminals: Iterable[str]
        ) -> defaultdict:
        """
        From terminals, filter out terminals that cannot follow the current terminal
        according to the grammar.

        If in the parsing table Action[cur_terminal, parser_state] = 'shift, new_parser_state' then next terminals
        are the terminals that are legal in new_parser_state.
        """
        following_terminals_map = defaultdict(set)
        terminals_set = set(terminals)
        cnt_seq_two_terms = 0

        # We iterate through each cur_terminal:    
        for cur_terminal in terminals:
            # Add all ignore terminals to the following terminals
            for next_terminal in ignore_terminals:
                following_terminals_map[cur_terminal].add(next_terminal)

            # We iterate through each parser_state:
            for _, row in parse_table.states.items():
                if cur_terminal in row:
                    action = row[cur_terminal]
                    # -> If we see a shift action to new_parser_state
                    if str(action[0]) == 'Shift':
                        new_parser_state = action[1]
                        for next_terminal in parse_table.states[new_parser_state]:
                            # Lark parse_table stores non-terminals and terminals together
                            if next_terminal in terminals_set:
                                # -> -> we add the terminals that are legal in new_parser_state
                                following_terminals_map[cur_terminal].add(next_terminal)
            cnt_seq_two_terms += len(following_terminals_map[cur_terminal])

        logger.info(f"Number of 2 length terminal sequences reduced from {len(terminals)*len(terminals)} to {cnt_seq_two_terms}")
        return following_terminals_map


    def _store_token_masks(self, terminals: Iterable[str], vocab_size: int, followings_terminas_map: dict=None):
        """
        Stores the token masks for each fsm state and next terminals
        """
        all_fsm_states = self._fsms.states()
        pbar = tqdm(total=len(all_fsm_states))

        for fsm_state in all_fsm_states:
            # Get the next terminals for the current fsm state
            if followings_terminas_map is not None and fsm_state.terminal in followings_terminas_map:
                following_terminals = followings_terminas_map[fsm_state.terminal]
            else:
                following_terminals = terminals

            # For each token, we check if it is a valid token for the current fsm state
            for token_idx in range(vocab_size): 
                # If the token is EOS token, we add it to the final state with the terminal '$END'
                if token_idx == self.eos_token_id: 
                    # Add 2 length terminal sequences where the second terminal is '$END'
                    if self._fsms.is_final(fsm_state):
                        self._lookup_table.fsm_state_and_next_terminal_to_tokens_add(
                            fsm_state, '$END', token_idx)
                else:
                    self._process_regular_tokens(
                        following_terminals, fsm_state, token_idx
                        )
            pbar.update(1)
        pbar.close()


    def _process_regular_tokens(self, terminals, fsm_state: JointFSMState, token_idx: int):
        token_bytes = self.byte_tokenizer.decode([token_idx])

        # For COMPLETE case:
        self._process_complete_case(fsm_state, token_idx, token_bytes)

        # For INCOMPLETE case:
        # Replace \t with 4 spaces
        remainder = token_bytes.replace(b'\t', b'    ')

        is_valid, remainder = self._fsms.consume_prefix(fsm_state, remainder)
        if is_valid:
            if len(remainder) == 0:
                # We reached a live state for the current terminal, thus we add the token in all sets of next terminals
                for next_terminal in terminals:
                    self._lookup_table.fsm_state_and_next_terminal_to_tokens_add(
                        fsm_state, next_terminal, token_idx)
            else:
                remainder = self._remove_left_whitespace(fsm_state, remainder)

                # We reached the final state while consuming the token, thus we conusme the remainder with all next terminals
                for next_terminal in terminals:
                    initial_state = self._fsms.initial(next_terminal)

                    if (initial_state, remainder) not in self._consume_prefix_cache:
                        # We use the cache to speed up the process only for the initial state
                        is_valid, remainder_new = self._fsms.consume_prefix(initial_state, remainder)
                        self._consume_prefix_cache[(initial_state, remainder)] = (is_valid, remainder_new)
                    else:
                        is_valid, remainder_new = self._consume_prefix_cache[(initial_state, remainder)]
                    
                    if self._mode == 'grammar_mask':
                        if is_valid: # In the non-strict mode we overapproximate
                            # We reached a live state for the next terminal, thus we add the 
                            # token in the  overapproximate sets of next terminals
                            self._lookup_table.fsm_state_and_next_terminal_to_tokens_add(
                                fsm_state, next_terminal, token_idx)
                    elif self._mode == 'grammar_strict':
                        if is_valid and len(remainder_new) == 0: 
                            # We reached a live state for the next terminal and the remainder 
                            # is empty, thus we add the token in the exact set of next terminals
                            self._lookup_table.fsm_state_and_next_terminal_to_tokens_add(fsm_state, next_terminal, token_idx)
                    else:
                        raise ValueError(f"Invalid mode: {self._mode}")


    def _process_complete_case(self, fsm_state, token_idx, token_bytes):
        remainder = token_bytes.replace(b'\t', b'    ')
        remainder = self._remove_left_whitespace(fsm_state, remainder)

        is_valid, remainder = self._fsms.consume_prefix(fsm_state, remainder)
        if is_valid and len(remainder) == 0:
            self._lookup_table.add_exact_lookup(fsm_state, token_idx)

    def _remove_left_whitespace(
        self, 
        fsm_state, 
        remainder: Union[str, bytes]
    ) -> Union[str, bytes]:
        """
        Ignore all left whitespace at the start of the terminal. This improves efficiency
        e.g. without this say if the model wants to generate '   def' then syncode will force it to generate '   ' and 'def' separately
        """
        if (self._fsms.initial(fsm_state.terminal) == fsm_state or self._fsms.is_final(fsm_state)) and self._ignore_whitespace:
            if isinstance(remainder, bytes):
                # For bytes, use lstrip() to remove all leading whitespace
                remainder = remainder.lstrip(b' ')
            elif isinstance(remainder, str):
                # For strings, use lstrip() to remove all leading whitespace
                remainder = remainder.lstrip(' ')
        return remainder

    def _lookup_next_tokens_for_fsm_state(self, fsm_state: JointFSMState, next_terminal) -> torch.Tensor:
        tokens = self._lookup_table.fsm_state_and_next_terminal_to_tokens(fsm_state, next_terminal)
        if tokens == []:
            # TODO: This is a hack. Do something better
            return self._lookup_table._get_default_mask()
        return tokens

    @staticmethod
    def _get_vocab_from_tokenizer(tokenizer, byte_string=False) -> Iterable[str]:
        """
        self.vocab is a list of readable token strings (e.g., ' hello' and '\n')
        sorted by their token IDs (so self.vocab[0] is the first token, etc). 

        if `byte_string` is True, then the vocab is returned as byte strings.
        """
        vocab = [v for k, v in
                        sorted([(t_id, tokenizer.decode([t_id]))
                                for _, t_id in tokenizer.get_vocab().items()])]

        # HACK: Is there a better way to know if a token has a prefix space?
        if 'Llama' in tokenizer.__class__.__name__:
            for i in range(len(vocab)):
                t = vocab[i]
                if 2*len(t) != len(tokenizer.decode([i, i], add_special_tokens=False)):
                    vocab[i] = ' ' + t
                if t == '':
                    vocab[i] = ' '
        
        if byte_string:
            vocab = [t.encode('utf-8') for t in vocab]

        return vocab


    def _lookup_next_tokens(
            self, 
            fsm_states: Iterable[JointFSMState], 
            remainder_state: RemainderState, 
            accept_sequences: Iterable
        ) -> torch.Tensor:
        """
        Lookup the next tokens for the current fsm states and remainder state and accept sequences. 
        """
        accept_token_mask = self._lookup_table._get_default_mask()

        # Case when the final string may be incomplete
        for fsm_state in fsm_states:
                for accept_sequence in accept_sequences:
                    if accept_sequence[0] == '$END':
                        accept_token_mask[self.eos_token_id] = 1

                    if fsm_state.terminal != accept_sequence[0]:
                        continue

                    if remainder_state == RemainderState.COMPLETE:
                            assert len(accept_sequence) == 1 # Since we only store length 1 accept sequences in this case
                            accept_token_mask |= self._lookup_table.complete_case_lookup(fsm_state)

                    if remainder_state == RemainderState.INCOMPLETE:
                            accept_token_mask |= self._lookup_table.incomplete_case_lookup(fsm_state)
                    
                    if remainder_state == RemainderState.MAYBE_COMPLETE:
                            if len(accept_sequence) == 1:
                                # mode='grammar_strict': incomplete_case_lookup is the same as complete_case_lookup
                                # mode='grammar_mask': incomplete_case_lookup is the overapproximate lookup
                                accept_token_mask |= self._lookup_table.incomplete_case_lookup(fsm_state)
                            elif len(accept_sequence) == 2:
                                accept_token_mask |= self._lookup_next_tokens_for_fsm_state(fsm_state, accept_sequence[1])
                            elif len(accept_sequence) == 3:
                                # If the DFA state is a final state we can jump to the start of next terminal
                                if self._fsms.is_final(fsm_state): 
                                    ignore_init_state = self._fsms.initial(accept_sequence[1])
                                    accept_token_mask |= self._lookup_next_tokens_for_fsm_state(ignore_init_state, accept_sequence[2])
                            else:
                                raise ValueError(f"Invalid accept sequence: {accept_sequence}")
        return accept_token_mask

    def get_fsm_states(self, r: ParseResult) -> Iterable[JointFSMState]:
        """
        Returns the DFA state for the current partial code
        """
        cur_incomplete_string = r.remainder
        if cur_incomplete_string is None:
            return []

        cur_fsm_states = self._fsms.compute_fsm_states(cur_incomplete_string)
        return cur_fsm_states

    def get_accept_mask(
            self, 
            r: ParseResult, 
            get_list=False
            ) -> torch.Tensor:
        """
        Returns the mask for the acceptable tokens for the current partial code

        Args:
            r (ParseResult): The parse result
            get_list (bool, optional): If True, returns the list of tokens instead of the mask. Defaults to False.
        """
        cur_incomplete_string = r.remainder
        assert type(cur_incomplete_string) == bytes

        if cur_incomplete_string is None:
            return torch.ones(len(self._vocab), dtype=torch.bool)

        cur_fsm_states = self._fsms.compute_fsm_states(cur_incomplete_string)
        accept_token_mask = self._lookup_next_tokens(
            cur_fsm_states, 
            r.remainder_state, 
            r.accept_sequences
            )

        if self.indentation and r.next_ac_indents is not None:
            indent_ac_token = self._lookup_table.get_indentation_tokens(r.next_ac_indents)
            accept_token_mask &= indent_ac_token
            
        if get_list: # This is useful for testing
            return self._get_tokens_list(accept_token_mask)
        return accept_token_mask
    
    def is_valid_prefix(self, r: ParseResult) -> bool:
        """
        Check if r.remainder is a valid prefix for accept sequences in r
        """
        cur_incomplete_string = r.remainder

        cur_fsm_states = self._fsms.compute_fsm_states(cur_incomplete_string)
        for accept_sequence in r.accept_sequences:
            for fsm_state in cur_fsm_states:
                if fsm_state.terminal == accept_sequence[0]:
                    return True
        return False

    def _list_to_mask(self, tokens_idx_list) -> torch.Tensor:
        indices = torch.tensor(tokens_idx_list)
        tokens_mask = self._lookup_table._get_default_mask()
        tokens_mask[indices] = 1
        return tokens_mask

    def _get_tokens_list(self, token_mask) -> Iterable[str]:
        return [self._vocab[idx.item()] for idx in torch.where(token_mask == True)[0]]
