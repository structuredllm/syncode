import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')
from syncode.parsers import create_parser
from syncode.parsers.grammars.grammar import Grammar
from syncode.parse_result import AcceptSequence, RemainderState

go_grammar = Grammar('go')
inc_parser = create_parser(go_grammar, ignore_whitespace=True)

class TestGoParser(unittest.TestCase):

    @unittest.skip("Skipping the correctness comparison test.")
    def test_tree_printer(self):
        '''NOTE: The EOS ignoring is handled separately in Go incremental parser.'''
        inc_parser.reset()
        partial_code = '''package main
import "fmt"
func main() {{
  x := y.(z)
}}
'''
        t = inc_parser.base_parser.parse(partial_code)
        self.assertTrue(True)  # Assuming success if no exception is raised

    def test_lexer(self):
        inc_parser.reset()
        partial_code = '''package main
import "fmt"
func main() {{
  var x int = 10
  var y int ='''
        out, _ = inc_parser._lex_code(partial_code)
        self.assertEqual(out[-1].type, 'EQUAL')

    def test_interactive_parser(self):
        inc_parser.reset()
        partial_code = '''package main
import "fmt"
func main() {{
  var x int = 10
  var y int ='''
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['EQUAL', 'DECIMAL_LIT']), res.accept_sequences)

    def test_parser2(self):
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Check if in given list of numbers, are any two numbers closer to each other than\n// given threshold.\n// >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n// False\n// >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n// True\n// \nfunc has_close_elements (numbers []float64, threshold float64) bool {\n'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIsNotNone(res) # Adjust according to what you're testing for

    def test_go_parser(self):
        inc_parser.reset()
        code = f'''package main
    import "fmt"
    func main() {{
        a, b = 3, 5
    }}
    '''
        out = inc_parser.base_parser.parse(code)
        print(out.pretty())

    def test_go_parser2(self):
        inc_parser.reset()
        code = f'''package main

    import "fmt"

    func main() {{
    var x int = 10
    var y int = 20
    var z int = x + y
    fmt.Println(z)
    }}
    '''
        out = inc_parser.base_parser.parse(code)
        # self.assertIsNotNone(out) # Adjust according to what you're testing for

    def test_go_parser3(self):
        inc_parser.reset()
        failed = False
        try:
            code = '''package main
import "fmt"

func main() {{
  var x int = 10
  var y int =
  var z int = x + y
  fmt.Println(z)
}}
''' 
            out = inc_parser.base_parser.parse(code)
        except Exception as e:
            failed = True
        self.assertTrue(failed)

    def test_go_parser4(self):
        inc_parser.reset()
        partial_code =  'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Check if in given list of numbers, are any two numbers closer to each other than\n// given threshold.\n// >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n// False\n// >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n// True\n// \nfunc has_close_elements (numbers []float64, threshold float64) bool {\n\n  // 1. \n'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertEqual('// 1. \n', res.remainder)

    def test_go_parser5(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n// >>> string_sequence(0)\n// \'0\'\n// >>> string_sequence(5)\n// \'0 1 2 3 4 5\'\n// \nfunc string_sequence (n int) string {\n\tvar result string\n\tfor i := 0; i < n; i++ {\n\t\tresult += strconv.Itoa(i'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertEqual('i', res.remainder)

    def test_go_parser6(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n// >>> intersperse([], 4)\n// []\n// >>> intersperse([1, 2, 3], 4)\n// [1, 4, 2, 4, 3]\n// \nfunc intersperse (numbers []int, delimeter int) []int {\n\tvar (  i, j int\n\t\tk int\n\t\tl []int\n\t)\n\tfor i = 0; i < len(numbers); i++ {\n\t\tfor j = i + 1; j < len(numbers); j++ {\n\t\t\tk = numbers[i] + delimeter + numbers[j]\n\t\t\tl = append(l, k)\n\t\t}\n\t}\n\treturn l\n}\n\nfunc main() {\n\t// Get the JSON string from the URL\n\tvar (  url string\n\t\tjsonData string\n\t\terr error\n\t)\n\turl = "http://localhost:8080/api/v1/json/get_data"\n\tjsonData, err '
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        print(res)
        self.assertIn(AcceptSequence(['EQUAL']), res.accept_sequences)

    def test_go_parser7(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n// >>> intersperse([], 4)\n// []\n// >>> intersperse([1, 2, 3], 4)\n// [1, 4, 2, 4, 3]\n// \nfunc intersperse (numbers []int, delimeter int) []int {\n\tvar (\n'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        # self.assertIn('__IGNORE_1', res.next_accept_terminals) # Assuming you're checking for a specific token

    def test_go_parser8(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\nfunc has_close_elements (numbers []float64, threshold float64) bool {\n\tif threshold > numbers[len(numbers)/'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['SLASH', 'DECIMAL_LIT']), res.accept_sequences)
        self.assertEqual('/', res.remainder)

    def test_go_parser9(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\nfunc has_close_elements (numbers []float64, threshold float64) bool {\n\tvar (\n\t\tmin, max float64\n\t\tmin_index, max_index int\n\t)\n\tfor i, n := range numbers'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['NAME', 'LBRACE']), res.accept_sequences)

    @unittest.skip("EOF is no longer used?")
    def test_go_parser10(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\nfunc has_close_elements (numbers []float64, threshold float64) bool { x:= 1; '
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        # self.assertIn(AcceptSequence(['EOF']), res.accept_sequences)

    def test_go_parser11(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Check if in given list of numbers, are any two numbers closer to each other than\n// given threshold.\n// >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n// False\n// >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n// True\n// \nfunc has_close_elements (numbers []float64, threshold float64) bool {\n\tif len(numbers) < 2 {\n\t\treturn false\n\t}\n\n\tfor i := 0; i < len(numbers)-1; i++ {\n\t\tfor j := i + 1; j < len(numbers); j++ {\n\t\t\tif reflect.DeepEqual(numbers[i],'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['COMMA', 'DECIMAL_LIT']), res.accept_sequences)

    def test_go_parser12(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n// >>> intersperse([], 4)\n// []\n// >>> intersperse([1, 2, 3], 4)\n// [1, 4, 2, 4, 3]\n// \nfunc intersperse (numbers []int, delimeter int) []int {\n\tresult := make([]int64'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['NAME', 'COMMA']), res.accept_sequences)
        self.assertEqual('int64', res.remainder)
        self.assertEqual(RemainderState.MAYBE_COMPLETE, res.remainder_state)

    def test_go_parser13(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"math"\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Return list of prime factors of given integer in the order from smallest to largest.\n// Each of the factors should be listed number of times corresponding to how many times it appeares in factorization.\n// Input number should be equal to the product of all factors\n// >>> factorize(8)\n// [2, 2, 2]\n// >>> factorize(25)\n// [5, 5]\n// >>> factorize(70)\n// [2, 5, 7]\n// \nfunc factorize (n int) []int {\n\tfactors := make([]int, 0)\n\tfor i := 2; i <= int(math.Sqrtln'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['NAME', 'LPAR']), res.accept_sequences)
        self.assertEqual('Sqrtln', res.remainder)
        self.assertEqual(RemainderState.MAYBE_COMPLETE, res.remainder_state)

    def test_go_parser14(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\n// You\'re an expert Golang programmer\n// Out of list of strings, return the longest one. Return the first one in case of multiple\n// strings of the same length. Return None in case the input list is empty.\n// >>> longest([])\n// \n// >>> longest([\'a\', \'b\', \'c\'])\n// \'a\'\n// >>> longest([\'a\', \'bb\', \'ccc\'])\n// \'ccc\'\n// \nfunc longest (strings []string) interface{} {\n\t'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['NL', 'IF']), res.accept_sequences)

    def test_go_parser15(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\nfunc numerical_letter_grade (grades []interface{}) []string {\n\tletter_grades := make([]string, len(grades))\n\tfor i, grade := range grades {\n\t\tswitch grade.('
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['LPAR', 'TYPE']), res.accept_sequences)
        self.assertIn(AcceptSequence(['LPAR', 'IGNORED', 'NAME']), res.accept_sequences)

    def test_go_parser16(self):
        inc_parser.reset()
        partial_code = 'package main\n\nimport (\n\t"encoding/json"\n\t"reflect"\n)\nfunc hex_key (num interface{}) int {\n\tvar hex_digits = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"}\n\tvar hex_count int\n\tvar hex_primes []int\n\tvar hex_prime_count int\n\tvar hex_prime_count_map = make(map[int]int)\n\n\t// Convert hexadecimal number to int\n\thex_num, ok := num.(string'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        self.assertIn(AcceptSequence(['NAME', 'RPAR']), res.accept_sequences)

    def test_go_parser17(self):
        inc_parser.reset()
        partial_code = 'pack'
        res = inc_parser.get_acceptable_next_terminals(partial_code)
        assert res.remainder == 'pack'
        self.assertIn(AcceptSequence(['PACKAGE']), res.accept_sequences)
        assert res.remainder_state == RemainderState.INCOMPLETE
