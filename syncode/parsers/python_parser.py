import copy, regex
from typing import Iterator
import syncode.larkm as lark
import syncode.common as common
from syncode.larkm import Token
from syncode.larkm.indenter import Indenter
from syncode.parsers.incremental_parser import IncrementalParser
from syncode.parse_result import IndentationConstraint, ParseResult, RemainderState
from typing import Optional, Iterable
import logging
logger = logging.getLogger(__name__)

class PythonIncrementalParser(IncrementalParser):
    """
    This class implements an incremental parser for Python code.
    """
    def __init__(self, base_parser, indenter, partial_code=None,**kwargs):
        super().__init__(base_parser, **kwargs)

        if partial_code is not None: # extract indentation type from partial code
            indenter.tab_len = self._get_indentation(partial_code)  # NOTE: tab_len is useful when \t and spaces are used for indentation in same code
        self.tab_len = indenter.tab_len
        self.indent_level = [0] # Current indentation level
    
    def reset(self):
        super().reset()
        self.indent_level = [0]

    def _get_indentation(self, partial_code) -> int:
        m = regex.match(r"(.*?):(.*?)\n(.*?)(?![ \t])", partial_code, flags=regex.DOTALL)
        indent_type = m.group(3)
        tab_len = 4 # Default tab length
        if '\t' not in indent_type: # that means we are using spaces for indentation
            tab_len = indent_type.count(' ')
        return tab_len

    def get_acceptable_next_terminals(self, partial_code) -> ParseResult:
        lexer_tokens, lexing_incomplete = self._lex_code(partial_code)
        self.next_ac_terminals = self._accepts(self.interactive)

        # Restore the previous state of the parser
        self._restore_recent_parser_state(lexer_tokens)
        interactive = self.interactive
        
        next_ac_indents = None

        # Parse the tokens
        self.time_accepts = 0
        parse_incomplete = False
        
        try:
            while self.cur_pos < len(lexer_tokens):
                token = lexer_tokens[self.cur_pos]
                self.cur_pos += 1

                if token.type == '_INDENT':
                    indent = token.count(' ') + token.count('\t') * self.tab_len
                    self._update_indent_levels(self.indent_level, indent)            
                elif token.type == '_DEDENT': # Do not shoot dedent tokens unless there is some code on the next line
                    self.dedent_queue.append(token)
                    continue
                else:
                    self.parsed_lexer_tokens.append(token) # parsed_token_seq holds all tokens except _INDENT and _DEDENT

                    while not len(self.dedent_queue)==0: # Shoot all the dedent tokens that are in the queue
                        self.indent_level.pop()
                        dedent_token = self.dedent_queue.pop()
                        interactive.feed_token(dedent_token)
                        self.cur_ac_terminals, self.next_ac_terminals = self.next_ac_terminals, self._accepts(interactive)
                
                interactive.feed_token(token)

                # Store the current state of the parser
                self._store_parser_state(
                    self.cur_pos-1,
                    lexer_tokens, 
                    interactive.parser_state.copy(), 
                    self._accepts(interactive),
                    indent_levels=copy.copy(self.indent_level)
                )
        except lark.exceptions.UnexpectedToken as e:
            parse_incomplete = True
            self._handle_parsing_error(lexer_tokens, token, e)

        remainder_state, final_terminal = None, None
        
        # Compute current terminal string
        remainder_state, current_term_str, final_terminal = self._get_remainder(
            partial_code, 
            lexing_incomplete=lexing_incomplete, 
            parse_incomplete=parse_incomplete
            )  
        
        cur_ac_terminals = self.cur_ac_terminals
        next_ac_terminals = self.next_ac_terminals
        next_ac_indents = None

        if remainder_state == RemainderState.MAYBE_COMPLETE or remainder_state == RemainderState.COMPLETE:
            if len(self.parsed_lexer_tokens) > 0 and self.parsed_lexer_tokens[-1].type == '_NL':
                # Calculate the last indetation level
                last_indent_str = self.parsed_lexer_tokens[-1].value.split('\n')[-1]
                
                last_indent = last_indent_str.count(' ') + last_indent_str.count('\t') * self.tab_len
                next_ac_indents = [indent-last_indent for indent in self.indent_level if indent >= last_indent]

                if '_INDENT' in self.next_ac_terminals:
                    next_ac_indents = IndentationConstraint(greater_than_indent_val=next_ac_indents[-1]) # next indentation level in this case
                elif '_INDENT' in self.cur_ac_terminals:
                    next_ac_indents = IndentationConstraint(greater_than_indent_val=next_ac_indents[-1]-1)
                else:  
                    next_ac_indents = IndentationConstraint(accept_indents=next_ac_indents)  

                # '_NL' is always accepted in this case
                cur_ac_terminals.add('_NL')
                next_ac_terminals.add('_NL') 

                # feed _DEDENT tokens in the interactive parser
                # See test_grammar_python.test_parser25
                while not len(self.dedent_queue)==0 and '_DEDENT' in self.next_ac_terminals:
                    dedent_token = self.dedent_queue.pop()
                    interactive.feed_token(dedent_token)
                    self.cur_ac_terminals = self.next_ac_terminals
                    self.next_ac_terminals = self._accepts(interactive)
                    next_ac_terminals |= self.next_ac_terminals

        return ParseResult.from_accept_terminals(cur_ac_terminals, next_ac_terminals, current_term_str, remainder_state, next_ac_indents=next_ac_indents, final_terminal=final_terminal, ignore_terminals=self.base_parser.lexer_conf.ignore)

    def _update_indent_levels(self, indent_level, indent):
        # if self.cur_pos != len(lexer_tokens): # Store previous indentation levels except the last one
        if indent > indent_level[-1]:
            indent_level.append(indent)
        else:
            while indent < indent_level[-1]:
                indent_level.pop()

    def _lex_code(self, code: str) -> Iterable[Token]:
        # Collect Lexer tokens
        lexer_tokens: Iterable[Token] = []
        interactive = self.base_parser.parse_interactive(code)
        lexer_state = interactive.lexer_thread.state
        indenter: PythonIndenter = self.base_parser.lexer_conf.postlex
        lexing_incomplete = False

        # Reset the indentation level
        indenter.indent_level, indenter.paren_level = [0], 0

        try:
            while lexer_state.line_ctr.char_pos < len(lexer_state.text):
                # PostLexConnector -> BasicLexer
                blexer = interactive.lexer_thread.lexer.lexer
                
                token = blexer.next_token(lexer_state)
                self.lexer_pos = lexer_state.line_ctr.char_pos
                
                # Perform postlexing indentation
                if token.type == indenter.NL_type:
                    lexer_tokens += indenter._handle_NL(token)
                else:
                    lexer_tokens.append(token)
                if token.type in indenter.OPEN_PAREN_types:
                        indenter.paren_level += 1
                elif token.type in indenter.CLOSE_PAREN_types:
                        indenter.paren_level -= 1
                        assert indenter.paren_level >= 0
        except lark.exceptions.UnexpectedCharacters as e: 
            lexing_incomplete = True
            pass # This may happen when the partial code has an ignore terminal
        except EOFError as e:
            pass

        return lexer_tokens, lexing_incomplete


class PythonIndenter(Indenter):
        """
        This class implements the indenter for Python code.
        """
        NL_type = "_NL"
        OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
        CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
        INDENT_type = "_INDENT"
        DEDENT_type = "_DEDENT"
        tab_len = 4

        def _handle_NL(self, token: Token) -> Iterator[Token]:
            '''
            This is taken from Lark library and modified to handle the case when there is a LONG_STRING comment in the _NL token.
            '''
            if self.paren_level > 0:
                return

            m = regex.match(r'(.*)(\'\'\'.*?\'\'\'|""".*?""")(.*)', token.value, flags=regex.DOTALL)
            if m is not None: # There is a LONG_STRING comment in the _NL token
                try:
                    indent_str = m.group(1).rsplit('\n', 1)[1] # Tabs and spaces
                except IndexError:
                    logger.error(f'Could not find the indentation for LONG_STRING comment in the token: {token}')
                    indent_str = ''

                indent = indent_str.count(' ') + indent_str.count('\t') * self.tab_len
                self.indent_level.append(indent)
                yield Token.new_borrow_pos(self.NL_type, m.group(0), token)
                yield Token.new_borrow_pos(self.INDENT_type, indent_str, token) 
                yield Token.new_borrow_pos('LONG_STRING', m.group(1), token)

            yield token

            indent_str = token.rsplit('\n', 1)[1] # Tabs and spaces
            indent = indent_str.count(' ') + indent_str.count('\t') * self.tab_len

            if indent > self.indent_level[-1]:
                self.indent_level.append(indent)
                yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
            else:
                while indent < self.indent_level[-1]:
                    self.indent_level.pop()
                    yield Token.new_borrow_pos(self.DEDENT_type, indent_str, token)
