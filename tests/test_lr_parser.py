import unittest
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from syncode.parsers import create_parser
from syncode.parsers.grammars.grammar import Grammar

class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common setup that runs once before all tests
        cls.tiny_grammar = Grammar('tiny')
        cls.calc_grammar = Grammar('calc')
        cls.python_grammar = Grammar('python')

    def test_tiny(self):
        inc_parser = create_parser(self.tiny_grammar, parser='lr')
        partial_code = "ccdd"
        out = inc_parser.base_parser.parse(partial_code)
        self.assertIsNotNone(out, "Parsing of tiny grammar failed.")

    def test_calc(self):
        inc_parser = create_parser(self.calc_grammar, parser='lr')
        partial_code = "113 + 235 + 1111"
        out = inc_parser.base_parser.parse(partial_code)
        self.assertEqual(out.children[0].children[0].children[1].children[0], '235', "Calc grammar parsing did not produce expected output.")

    def test_time(self):
        """
        Results:
        Time taken for building LR(1): 102.65705108642578
        Time taken for building LALR(1): 0.3684959411621094
        LR(1) states: 752
        LALR(1) states: 4926
        Time taken for parsing with LR(1): 0.14716744422912598
        Time taken for parsing with LALR(1): 1.5615639686584473
        """
        time1 = time.time()
        inc_lr_parser = create_parser(Grammar('python'), parser='lr')
        time2 = time.time()
        print("Time taken for building LR(1):", time2 - time1)
        inc_lalr_parser = create_parser(Grammar('python'), parser='lalr')
        time3 = time.time()
        print("Time taken for building LALR(1):", time3 - time2)
        
        print(len(inc_lr_parser.base_parser.parser.parser._parse_table.states))
        print(len(inc_lalr_parser.base_parser.parser.parser._parse_table.states))

        prompt = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n'

        generated_code = '\t"""\n\tfor i in range(len(numbers) -1, -1, -1):\n\t\tfor j in range(i+1, len(numbers) -1, -1):\n\t\t\tif abs(numbers[i] - numbers[j] ) < threshold:\n\t\t\t\treturn True\n\treturn False\n\n\ndef has_close_elements_with_threshold(numbers: List[float] , threshold: float) -> bool:\n\ta="shu'

        i = 0
        while i<len(generated_code):
            i += 2
            r = inc_lalr_parser.get_acceptable_next_terminals(prompt + generated_code[:i])
        assert r.remainder == '"shu'
        assert r.next_ac_indents == None
        time4 = time.time()
        print("Time taken for parsing with LALR(1):", time4 - time3)

        i = 0
        while i<len(generated_code):
            i += 2
            r = inc_lr_parser.get_acceptable_next_terminals(prompt + generated_code[:i])
        assert r.remainder == '"shu'
        assert r.next_ac_indents == None
        time5 = time.time()
        print("Time taken for parsing with LR(1):", time5 - time4)

    @unittest.skip("Skipping the correctness comparison test.")
    def test_correct(self):
        inc_lr_parser = create_parser(self.python_grammar, parser='lr')
        inc_lalr_parser = create_parser(self.python_grammar, parser='lalr')

        prompt = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n'
        generated_code = '\t"""\n\tfor i in range(len(numbers) -1, -1, -1):\n\t\tfor j in range(i+1, len(numbers) -1, -1):\n\t\t\tif abs(numbers[i] - numbers[j] ) < threshold:\n\t\t\t\treturn True\n\treturn False\n\n\ndef has_close_elements_with_threshold(numbers: List[float] , threshold: float) -> bool:\n\ta="shu'

        i = 0
        while i < len(generated_code):
            i += 2
            r1 = inc_lalr_parser.get_acceptable_next_terminals(prompt + generated_code[:i])
            r2 = inc_lr_parser.get_acceptable_next_terminals(prompt + generated_code[:i])
            self.assertEqual(r1, r2, "LR and LALR parsers produced different results.")

    def test_caching(self):
        time1 = time.time()
        inc_parser1 = create_parser(Grammar('python'), parser='lr')
        time2 = time.time()
        print("Time taken for creating LR(1) parser:", time2 - time1)
        inc_parser2 = create_parser(Grammar('python'), parser='lr')
        time3 = time.time()
        print("Time taken for creating LR(1) parser second time:", time3 - time2)
