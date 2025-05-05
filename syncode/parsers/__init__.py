import os
from syncode.parsers import incremental_parser
from syncode.parsers.itergen_parser import IGParser
from syncode.parsers.python_parser import PythonIncrementalParser, PythonIndenter
from syncode.parsers.go_parser import GoIncrementalParser
import syncode.common as common
from syncode.larkm.lark import Lark
from syncode.parsers.grammars.grammar import Grammar

def create_parser(
          grammar: Grammar, 
          parser='lalr', 
          use_symbol_pos_map=False,
          **kwargs
        ) -> incremental_parser.IncrementalParser:   
        """ 
        Creates an incremental parser for the given grammar. The parser is cached for future use.
        parser (str, optional): The type of parser to use. Can be 'lalr' or 'lr'. Defaults to 'lalr'.        
        """
        indenter = None
        parser_cache_dir = common.SYNCODE_CACHE + 'parsers/'
        cache_filename = parser_cache_dir + f'{grammar}_{parser}_{grammar.hash()}_parser.pkl'
        os.makedirs(os.path.dirname(parser_cache_dir), exist_ok=True)

        if grammar.name == 'python' and not use_symbol_pos_map:
            indenter = PythonIndenter()

        base_parser = create_base_parser(grammar, parser, indenter, cache_filename)

        # First check if we should use the IGParser with symbol position map
        if use_symbol_pos_map:
            return IGParser(base_parser, **kwargs)
        
        if grammar.name == 'python':
            return PythonIncrementalParser(base_parser, indenter, **kwargs)
        elif grammar.name == 'go':
            return GoIncrementalParser(base_parser, **kwargs)
        return incremental_parser.IncrementalParser(base_parser, **kwargs)


def create_base_parser(grammar, parser='lalr', indenter=None, cache_filename=None):
    base_parser = Lark( # This is the standard Lark parser
                        grammar.ebnf,
                        parser=parser,
                        lexer="basic",
                        start="start",
                        postlex=indenter,
                        propagate_positions=True,
                        cache = cache_filename
                    )
                
    return base_parser
