import logging
from typing import Any, Dict, List, Tuple

import sys, os

from syncode.larkm.parsers.lalr_analysis import ParseTable, Reduce, Shift
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from syncode.larkm.common import LexerConf, ParserConf
from syncode.larkm.lexer import Token
from rust_parser import RustLexer, RustParser
from syncode.parsers.incremental_parser import IncrementalParser

logger = logging.getLogger(__name__)

class ParserRS:
    """
    A Rust-based implementation of Lark's LR(1) parser.
    This provides a significant performance boost while maintaining compatibility.
    """   
    def __init__(self, base_parser) -> None:
        """
        Initialize the parser with the given base parser.
        
        Args:
            base_parser: Lark's base parser containing configurations.
        """
        lexer_conf: LexerConf = base_parser.lexer_conf
        parser_conf: ParserConf = base_parser.parser.parser.parser_conf
        parse_table: ParseTable = base_parser.parser.parser._parse_table

        # Convert terminal definitions for Rust
        terminal_defs = convert_terminal_defs(lexer_conf.terminals)
        
        # Extract and convert rules for Rust parser
        rules = []
        rule_to_id = {}
        for rule_id, rule in enumerate(parser_conf.rules):
            rules.append((rule_id, rule.origin.name, [s.name for s in rule.expansion]))
            rule_to_id[rule] = rule_id
        
        # Extract the parsing tables
        states_dict = {}
        for state_idx, state_actions in parse_table.states.items():
            state_transitions = {}
            for symbol, (action, arg) in state_actions.items():
                # Convert action to format expected by Rust
                if action == Shift:
                    state_transitions[symbol] = ('shift', str(arg))
                elif action == Reduce:
                    rule_id = rule_to_id[arg]
                    state_transitions[symbol] = ('reduce', str(rule_id))
                # else:
                #     state_transitions[symbol] = ('error', '')
            
            states_dict[str(state_idx)] = state_transitions
        
        # Extract ignore types
        ignore_types = set(lexer_conf.ignore) if not isinstance(lexer_conf.ignore, set) else lexer_conf.ignore
        
        # Create the Rust parser
        start_symbol = parser_conf.start[0]
        self._rust_parser = RustParser()
        self._rust_parser.initialize(
            terminal_defs,
            ignore_types,
            lexer_conf.use_bytes if hasattr(lexer_conf, 'use_bytes') else False,
            rules,
            states_dict,
            # We assume there is a single start symbol
            start_symbol,
            parse_table.start_states[start_symbol],
            parse_table.end_states[start_symbol],
        )
    
    def parse_text(self, text, start=None) -> Tuple[bool, str]:
        """
        Parse the given text and return success status and any unparsed text.
        All lexing and parsing happens in Rust for maximum performance.
        
        Args:
            text: The text to parse.
            start: Optional start symbol (not used in this implementation as it's set in initialization)
            
        Returns:
            Tuple of (success, remaining_text) where success is a boolean indicating if parsing was successful.
        """
        # Call the Rust parser directly with the input text
        result = self._rust_parser.parse_text(text)
        
        # Check for parse errors
        if "error" in result:
            # Handle error (you might want to customize this)
            error_info = result["error"]
            error_type = error_info.get("type", "unknown-error")
            
            if error_type == "unexpected-token":
                token = error_info.get("token", {})
                pos = token.get("pos", 0)
                line = token.get("line", 1)
                column = token.get("column", 0)
                expected = error_info.get("expected", [])
                value = token.get("value", "")
                type_name = token.get("type", "")
                
                raise Exception(f"Unexpected token {type_name}('{value}') at line {line}, column {column}. Expected: {', '.join(expected)}")
            elif error_type == "unexpected-eof":
                raise Exception("Unexpected end of file")
            elif error_type == "lexer-error":
                pos = error_info.get("pos", 0)
                line = error_info.get("line", 1)
                column = error_info.get("column", 0)
                char = error_info.get("char", "")
                
                raise Exception(f"Lexer error at line {line}, column {column}. Unexpected character: '{char}'")
            else:
                raise Exception(f"Parser error: {error_info.get('message', 'unknown error')}")
        
        # Extract success and consumed position
        success = result.get("success", False)
        consumed_length = result.get("consumed", 0)
        remaining_text = text[consumed_length:]
        
        # Return success and the remaining text
        return success, remaining_text

    def get_acceptable_next_terminals(self, partial_code):
        raise NotImplementedError("WIP")


class LexerRS:
    """
    A Rust-based implementation of Lark's lexer.
    This provides a significant performance boost while maintaining compatibility.
    """ 
    def __init__(self, conf: 'LexerConf') -> None:
        """
        Initialize the lexer with the given configuration.
        
        Args:
            conf: Lexer configuration from Lark.
        """
        # Create the Rust lexer
        self._rust_lexer = RustLexer()
        
        # Convert and pass terminal definitions to Rust
        terminal_defs = convert_terminal_defs(conf.terminals)
        
        # Ensure ignore_types is a set
        ignore_types = set(conf.ignore) if not isinstance(conf.ignore, set) else conf.ignore
        
        # Initialize the Rust lexer
        self._rust_lexer.initialize(
            terminal_defs,
            ignore_types,  # Now guaranteed to be a set
            conf.callbacks,
            conf.use_bytes
        )
        
    def lex_text(self, text: str) -> List[Token]:
        """
        Lex the entire text and return all tokens.
        This is more efficient as it uses the Rust implementation directly.
        """
        # Call the Rust lexer directly for better performance
        results = self._rust_lexer.lex_text(text)

        # Check for errors
        if results and "error" in results[0] and results[0]["error"] != "eof":
            if results[0]["error"] == "unexpected-char":
                handle_lexer_error(results[0])
                return []
        
        # Convert results to Lark tokens
        tokens = []
        for result in results:
            if "error" in result:
                continue  # Skip error results
                
            token = Token(
                result["type"],
                result["value"],
                result["start_pos"],
                result["line"],
                result["column"],
                result["end_line"],
                result["end_column"],
                result["end_pos"]
            )
            tokens.append(token)
            
        return tokens


def handle_error(error: Dict, text: str) -> None:
    """Handle and log parsing or lexical errors."""
    if error["type"] == "unexpected-token":
        logger.error(f"Unexpected token '{error['token']}' at position {error['pos']}")
        logger.error(f"Expected one of: {', '.join(error['expected'])}")
    elif error["type"] == "unexpected-eof":
        logger.error(f"Unexpected end of input")
    elif error["type"] == "unexpected-char":
        logger.error(f"Unexpected character at position {error['pos']}, line {error['line']}, column {error['column']}")
        logger.error(f"Expected one of: {', '.join(error['allowed'])}")


def handle_lexer_error(error: Dict) -> None:
    """Handle and log lexical errors."""
    logger.error(f"Unexpected character at position {error['pos']}, line {error['line']}, column {error['column']}")
    if 'allowed' in error:
        logger.error(f"Expected one of: {', '.join(error['allowed'])}")

def convert_terminal_defs(terminals):
    """Convert terminal definitions from Lark format to Rust-compatible format."""
    terminal_defs = []
    for t in terminals:
        # Extract pattern as dictionary for Rust
        if isinstance(t.pattern.flags, frozenset):
            flags = set(t.pattern.flags)
        else:
            flags = t.pattern.flags

        pattern_dict = {
            "type": t.pattern.type,
            "value": t.pattern.value,
            "flags": flags
        }
        terminal_defs.append((t.name, pattern_dict, t.priority))
    return terminal_defs
