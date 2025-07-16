import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')
from syncode.parsers import create_parser
from syncode.parsers.grammars.grammar import Grammar
from syncode.parse_result import AcceptSequence, RemainderState

java_grammar = Grammar('java')
inc_parser = create_parser(java_grammar)

# Note: If there is no trailing whitespace in the partial code,
#   then the current terminal appears in the accept sequence,
#   and the remainder state is MAYBE_COMPLETE.

class TestJavaParser(unittest.TestCase):
    def test_java_parser1(self):
        inc_parser.reset()
        partial_code = "public class "
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == ''
        assert AcceptSequence(['CNAME']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser2(self):
        inc_parser.reset()
        partial_code = """
public class HelloWor"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == 'HelloWor'

        # Lack of whitespace at the end
        #   => maybe complete; include the current terminal in the accept sequence
        assert AcceptSequence(['CNAME', 'LBRACE']) in r.accept_sequences
        assert AcceptSequence(['CNAME', 'IMPLEMENTS']) in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser3(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main("""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == '('

        # Looking for a type (e.g. "short") as part of a parameter
        assert AcceptSequence(['LPAR', 'SHORT']) in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser4(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[]"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == ']'

        # Need a variable name before closing parentheses
        assert AcceptSequence(['RSQB', 'CNAME']) in r.accept_sequences
        assert AcceptSequence(['RSQB', 'RPAR']) not in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser5(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == 'args'
        assert AcceptSequence(['CNAME', 'RPAR']) in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser6(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == ''
        assert AcceptSequence(['LBRACE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser7(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) {
        System.
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == ''
        assert AcceptSequence(['CNAME']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser8(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) {
        System.out.println("H
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        # __ANON_1 is a string literal
        assert AcceptSequence(['__ANON_1']) in r.accept_sequences

        # Incomplete because quotes need to be closed
        assert r.remainder_state == RemainderState.INCOMPLETE

    def test_java_parser9(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) {
        System.out.println("Hello World!")
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['SEMICOLON']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser10(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) {
        System.out.println("Hello World!");
    }
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['RBRACE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser11(self):
        inc_parser.reset()
        partial_code = """
public class HelloWorld
{
	public static void main(String[] args) {
        System.out.println("Hello World!");
    }
}
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['$END']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser12(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['CNAME']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser13(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['EQUAL']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser14(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['LPAR']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser15(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for ( """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['INT']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser16(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; """
        r = inc_parser.get_acceptable_next_terminals(partial_code)

        assert AcceptSequence(['CNAME']) in r.accept_sequences
        assert AcceptSequence(['FALSE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser17(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)

        # __ANON_16 is +=; __ANON_9 is ==
        assert AcceptSequence(['CNAME', '__ANON_16']) in r.accept_sequences
        assert AcceptSequence(['CNAME', '__ANON_9']) not in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser18(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['LBRACE']) in r.accept_sequences
        assert AcceptSequence(['CONTINUE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser19(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci("""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['LPAR', 'CNAME']) in r.accept_sequences
        assert AcceptSequence(['LPAR', 'RPAR']) in r.accept_sequences
        assert r.remainder_state == RemainderState.MAYBE_COMPLETE

    def test_java_parser20(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if """
        r = inc_parser.get_acceptable_next_terminals(partial_code)

        # Only ( or whitespace can follow if
        assert AcceptSequence(['LPAR']) in r.accept_sequences
        assert AcceptSequence(['CNAME']) not in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser21(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if (n == """
        
        r = inc_parser.get_acceptable_next_terminals(partial_code)

        # Need something before close parenthesis
        assert AcceptSequence(['CNAME']) in r.accept_sequences
        assert AcceptSequence(['DIGIT']) in r.accept_sequences
        assert AcceptSequence(['RPAR']) not in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser22(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if (n == 0)
            return 0;
"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['ELSE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser23(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if (n == 0)
            return 0;
        else """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['LPAR']) in r.accept_sequences
        assert AcceptSequence(['IF']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser24(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if (n == 0)
            return 0;
        else if (n == 1)
            return 1;
        return slowFibonacci(n-1) """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['PLUS']) in r.accept_sequences
        assert AcceptSequence(['SEMICOLON']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    def test_java_parser25(self):
        inc_parser.reset()
        partial_code = """
public class Fibonacci
{
	public static void main(String[] args) {
        int num = 10;
        for (int i = 0; i < num; i++)
            System.out.println("F_" + i + " = " + slow_fibonacci(i));
    }
    
    public static int slowFibonacci(int n) {
        if (n == 0)
            return 0;
        else if (n == 1)
            return 1;
        return slowFibonacci(n-1) + slowFibonacci(n-2); """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['RBRACE']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

    @unittest.skip("Comments not yet supported.")
    def test_java_parser26(self):
        inc_parser.reset()
        partial_code = """
/**
 * Comment test
 */
public """
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert AcceptSequence(['CLASS']) in r.accept_sequences
        assert r.remainder_state == RemainderState.COMPLETE

if __name__ == '__main__':
    unittest.main()
    