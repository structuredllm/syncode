from enum import Enum
from typing import Tuple, Optional

class AcceptSequence(list):
    """
    Stores the sequence of terminals that are accepted by the parser.
    """
    def __init__(self, accept_terminals: Tuple[str]):
        self.accept_terminals = accept_terminals
    
    def __getitem__(self, id):
        return self.accept_terminals[id]

    def __repr__(self):
        return 'accept_terminals: {}'.format(self.accept_terminals)
    
    def __eq__(self, other):
        return self.accept_terminals == other.accept_terminals

    def __hash__(self):
        return hash(str(self.accept_terminals))

    def __len__(self):
        return len(self.accept_terminals)
    
    def __add__(self, other):
        return AcceptSequence(self.accept_terminals + other.accept_terminals)

class RemainderState(Enum):
    """
    The state of the reminder after parsing partial code.
    """
    COMPLETE = 0
    MAYBE_COMPLETE = 1
    INCOMPLETE = 2

class ParseResult:
    """ 
    Stores the result of parsing. 
    """
    def __init__(self, accept_sequences, remainder, remainder_state: RemainderState, next_ac_indents=None, function_end=False):
        self.remainder = remainder
        self.remainder_state = remainder_state
        self.accept_sequences = accept_sequences
        self.next_ac_indents: Optional[IndentationConstraint] = next_ac_indents
        self.function_end = function_end
        
    @staticmethod
    def from_accept_terminals(cur_accept_terminals, next_accept_terminals, remainder, remainder_state: RemainderState, next_ac_indents=None, final_terminal=None, ignore_terminals=None) -> 'ParseResult':
        """
        Create a ParseResult from current and next accept terminals.
        """
        if remainder_state == RemainderState.COMPLETE: 
            accept_sequences = {AcceptSequence([t]) for t in next_accept_terminals}
        elif remainder_state == RemainderState.INCOMPLETE:
            accept_sequences = {AcceptSequence([t]) for t in cur_accept_terminals}
        else:
            accept_sequences = set()
            assert final_terminal is not None
            for t in cur_accept_terminals:
                if t == final_terminal:
                    for t2 in next_accept_terminals:
                        accept_sequences.add(AcceptSequence([final_terminal, t2]))
                    if ignore_terminals is not None:
                        for t2 in ignore_terminals:
                            accept_sequences.add(AcceptSequence([final_terminal, t2]))
                else:
                    accept_sequences.add(AcceptSequence([t]))
        
        if ignore_terminals is not None: # Does this cause imprecision?
            # Add the sequences that only contain ignore_terminals    
            accept_sequences = accept_sequences.union({AcceptSequence([t]) for t in ignore_terminals})

        next_ac_indents: IndentationConstraint = next_ac_indents

        if remainder_state == RemainderState.INCOMPLETE: # If the terminal is not complete, then next_accept_terminals should be None
            assert len(next_accept_terminals) == 0
        function_end = True if '$END' in next_accept_terminals else False
        return ParseResult(accept_sequences, remainder, remainder_state, next_ac_indents, function_end=function_end)

    def __repr__(self):
        return 'remainder : {}, remainder_state: {}, accept_sequences: {}, next_ac_indents: {}'.format(repr(self.remainder), self.remainder_state, self.accept_sequences, self.next_ac_indents)
    
    def __eq__(self, other):
        return self.remainder == other.remainder and self.remainder_state == other.remainder_state and self.cur_accept_terminals == other.cur_accept_terminals and self.next_accept_terminals == other.next_accept_terminals and self.next_ac_indents == other.next_ac_indents

class IndentationConstraint:
    """
    Stores the indentation constraints for a terminal.
    """
    def __init__(self, accept_indents=None, greater_than_indent_val=None):
        self.accept_indents = accept_indents
        self.greater_than_indent_val = greater_than_indent_val
        assert accept_indents is None or greater_than_indent_val is None # Exactly one of them should be None

    def __repr__(self):
        return 'accept_indents: {}, greater_than_indent_val: {}'.format(self.accept_indents, self.greater_than_indent_val)

    def __eq__(self, other):
        return self.accept_indents == other.accept_indents and self.greater_than_indent_val == other.greater_than_indent_val
    