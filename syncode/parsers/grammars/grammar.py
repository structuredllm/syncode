import hashlib
import os

class Grammar:
    """
    The grammar can be provided in 3 ways:
    1) Name of the grammar available in syncode/parsers/grammars. Can be 'python', 'go', 'tiny', 'calc' etc.
    2) Full path to the grammar file. In this case, the grammar file should be in Lark format.
    3) The grammar itself in Lark/EBNF form.
    """
    def __init__(self, name):
        self.name = name
        self.ebnf = None
        grammar_filename = None
        assert name is not None, 'Grammar name not provided in grammar mode!'
        if name in ['python', 'go', 'sql', 'tiny', 'calc', 'json', 'c', 'java', 'javascript', 'prover9']:
            grammar_filename = f'{os.path.dirname(__file__)}/{name}_grammar.lark'
        elif name.endswith('.lark'): 
            if os.path.exists(name):
                # In this case we assume that the user provides the full path to the grammar file
                grammar_filename = name
            else:
                raise ValueError(f'grammar input file {name} does not exist!')            
    
        if grammar_filename is not None:
            if os.path.exists(grammar_filename):
                    with open(grammar_filename, 'r') as file:
                        self.ebnf = file.read()
            else:
                raise ValueError(f'grammar file {grammar_filename} does not exist!')
        else: # Grammar can also be specified as a string in EBNF form
            self.ebnf = name
            self.name = 'custom'

    def __str__(self):
        return self.name 
    
    def hash(self) -> int:
        return str(int(hashlib.sha256(self.ebnf.encode('utf-8')).hexdigest(), 16))[:10]
    
    def simplifications(self):
        """
        The Simplification class presents a mapping from the original regex to the simplified regex for certain terminals. 
        There are two reasons for doing this. In some cases, the original regex is complex and requires large DFAs. Computing the overapproximate tokens for such DFAs is expensive. The other reason is that the interegular library does not support some regex features such as lookaheads. Thus, we use the simplified regex to compute the overapproximate tokens.

        # NOTE: We are not changing the actual regex of the terminals while parsing. We are only using the simplified regex for computing the overapproximate tokens maintaining the soundness.
        """
        if self.name == 'python':
            # Simplifications for python
            python_simplifications = {
                            'COMMENT': '(?i:(?s:(#.*|\'\'\'.*?\'\'\'|""".*?""")))', 
                            '_NL': '(?s:(?i:\n(.*)))', 
                            'LONG_STRING': '(?i:(?s:(\'\'\'.*?\'\'\'|""".*?""")))', 
                            'STRING': '(?s:(?i:[ubf]?r?(".*?"|\'.*?\')))'
                            }
            return python_simplifications
        return {}
