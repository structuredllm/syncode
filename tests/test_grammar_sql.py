import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from syncode.parsers import create_parser
from syncode.parsers.grammars.grammar import Grammar
from syncode.parse_result import AcceptSequence, RemainderState

sql_grammar = Grammar('sql')
inc_parser = create_parser(sql_grammar)

class TestSQLParser(unittest.TestCase):
    def test_sql_parser1(self):
        # Tests when the last incomplete word is unparsed
        inc_parser.reset()
        partial_code = "SELECT     singer.name,     singer.country,     singer.age FROM     singer     JOIN singer_in_concert ON singer.singer_id = singer_in_concert.singer_id     JO"
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == 'JO'
        assert AcceptSequence(['JOIN_EXPR']) in r.accept_sequences
        assert r.remainder_state == RemainderState.INCOMPLETE

    def test_sql_parser2(self):
        inc_parser.reset()
        # TODO: SELECT is accepted here since it can be prefix for a CNAME
        partial_code = "\nSELECT s.name, s.song_release_year\nFROM singer s\nJOIN concert c ON s.concert_id = c.concert_id\nJOIN singer_in_concert si ON c.singer_id = si.singer_id\nWHERE si.singer_id = (\n"
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        assert r.remainder == ''
        assert r.remainder_state == RemainderState.COMPLETE
        