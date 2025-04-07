import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')
from syncode.grammar_mask.grammar_constrainer import GrammarConstrainer

class TestGrammarDelimeter(unittest.TestCase):
    def test_all_cases(self):
        test_cases = [
            {
                "desc": "Single math block",
                "text": "Here's a formula: <<x^2 + y^2 = z^2>> in the middle of text.",
                "start": "<<",
                "end": ">>",
                "expected_text": "x^2 + y^2 = z^2",
                "expected_constrain": False,
            },
            {
                "desc": "Multiple math blocks, returns last",
                "text": "Intro <<a+b>> some more <<c+d>> ending",
                "start": "<<",
                "end": ">>",
                "expected_text": "c+d",
                "expected_constrain": False,
            },
            {
                "desc": "Missing closing delimiter",
                "text": "This math is broken <<1+1=",
                "start": "<<",
                "end": ">>",
                "expected_text": "1+1=",
                "expected_constrain": True,
            },
            {
                "desc": "Nested-looking delimiters",
                "text": "Messy: <<1 + <<2>> + 3>>",
                "start": "<<",
                "end": ">>",
                "expected_text": "1 + <<2",  # Still not closed due to the second <<
                "expected_constrain": False,
            },
            {
                "desc": "No delimiters present",
                "text": "No math here, just plain text.",
                "start": "<<",
                "end": ">>",
                "expected_text": "",
                "expected_constrain": False,
            },
            {
                "desc": "Math with newlines",
                "text": "Start <<\na = b + c\nf(x) = x^2\n>> end",
                "start": "<<",
                "end": ">>",
                "expected_text": "a = b + c\nf(x) = x^2",
                "expected_constrain": False,
            },
            {
                "desc": "Only start delim exists after closed one",
                "text": "Closed first: <<1+2>> then opened: <<3+4",
                "start": "<<",
                "end": ">>",
                "expected_text": "3+4",
                "expected_constrain": True,
            },
            {
                "desc": "Edge case where text ends with prefix of the delimiter",
                "text": "some text << xyz >",
                "start": "<<",
                "end": ">>",
                "expected_text": "",
                "expected_constrain": False,
            },
            # New test cases for Python code block
            {
                "desc": "Python code block with code",
                "text": "Some introductory text before code block ```python\nx = 5\nprint(x)``` more text after.",
                "start": "```python\n",
                "end": "```",
                "expected_text": "x = 5\nprint(x)",
                "expected_constrain": False,
            },
            {
                "desc": "Python code block with no closing delimiter",
                "text": "Here is some code: ```python\nx = 5\nprint(x)",
                "start": "```python\n",
                "end": "```",
                "expected_text": "x = 5\nprint(x)",
                "expected_constrain": True,
            },
            {
                "desc": "Multiple code blocks with a start delimiter present after a closed block",
                "text": "First block ```python\nx = 10``` and then a second block ```python\ny = 20",
                "start": "```python\n",
                "end": "```",
                "expected_text": "y = 20",
                "expected_constrain": True,
            },
        ]

        for case in test_cases:
            with self.subTest(msg=case["desc"]):
                result_text, should_constrain = GrammarConstrainer.extract_last_structured_block(
                    case["text"], case["start"], case["end"]
                )
                self.assertEqual(result_text, case["expected_text"])
                self.assertEqual(should_constrain, case["expected_constrain"])

if __name__ == "__main__":
    unittest.main()
