import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from syncode.parsers import create_parser
from syncode.parsers.grammars.grammar import Grammar
from syncode.parse_result import AcceptSequence, RemainderState

javascript_grammar = Grammar('javascript')
inc_parser = create_parser(javascript_grammar)

# Note: If there is no trailing whitespace in the partial code,
#   then the current terminal appears in the accept sequence,
#   and the remainder state is MAYBE_COMPLETE.

class TestJavaScriptParser(unittest.TestCase):
    def test_java_parser1(self):
        inc_parser.reset()
        code = """// Declare an array of names
const names = ["John", "Mary", "Jane"];

// Iterate over the array using a for loop
for (let i = 0; i < names.length; i++) {
  // Print the current name to the console
  console.log(names[i]);
}
"""
        out = inc_parser.base_parser.parse(code)
        print(out.pretty())
