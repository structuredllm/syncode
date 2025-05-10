// src/lexer.rs
use regex::Regex;
use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::{Anchored, util::start};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Token struct to represent lexer tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub value: String,      // The content of the token.
    pub type_name: String,  // The type of the token in the grammar, "" if unlexable.
    pub start_pos: usize,
    pub end_pos: usize,
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({}, {})", self.type_name, self.value)
    }
}

/// Pattern types to match Lark's patterns.
#[derive(Clone, Debug)]
pub enum Pattern {
    Str(String),
    Regex(String, HashSet<String>), // regex pattern and flags
}

/// Terminal definition matching Lark's TerminalDef.
#[derive(Clone, Debug)]
pub struct TerminalDef {
    pub name: String,
    pub pattern: Pattern,
    pub priority: i32,
}

// Error types
/// A type to describe errors with the lexer object itself (e.g. invalid
/// initialization).
#[derive(Debug, Clone)]
pub enum LexerError {
    UnexpectedChar {
        pos: usize,
        line: usize,
        column: usize,
        allowed: Vec<String>,
        char: char,
    },
    Eof {
        pos: usize,
        line: usize,
        column: usize,
    },
    InitError(String),
    RegexError(String),
}

/// Result enum to represent either a token or an error that resulted from the
/// input (e.g. no lexical token was found).
// #[derive(Debug, Clone)]
// pub enum LexResult {
//     Token(Token), // Some lexical token of the grammar.
//     Error(Token),
// Eof {
//     // End-of-file.
//     pos: usize,
//     line: usize,
//     column: usize,
// },
// }

/// Wrap
pub struct Scanner {
    // The DFA for matching patterns
    dfa: dense::DFA<Vec<u32>>,

    // Maps DFA match pattern to token type name
    index_to_type: HashMap<usize, String>,

    // Maps token type name to whether it can contain newlines
    newline_types: HashSet<String>,

    // Terminal definitions for reference
    terminals: Vec<TerminalDef>,

    // All allowed types
    pub allowed_types: HashSet<String>,
}

impl Scanner {
    pub fn new(terminals: Vec<TerminalDef>) -> Result<Self, LexerError> {
        let mut newline_types = HashSet::new();
        let mut allowed_types = HashSet::with_capacity(terminals.len());
        let mut index_to_type = HashMap::with_capacity(terminals.len());

        // Determine which patterns might contain newlines
        for terminal in &terminals {
            let pattern_str = match &terminal.pattern {
                Pattern::Str(s) => s,
                Pattern::Regex(r, _) => r,
            };

            if pattern_str.contains("\\n")
                || pattern_str.contains("\n")
                || pattern_str.contains("\\s")
                || pattern_str.contains("[^")
                || (pattern_str.contains(".") && pattern_str.contains("(?s"))
            {
                newline_types.insert(terminal.name.clone());
            }

            allowed_types.insert(terminal.name.clone());
        }

        // Sort terminals by priority (highest first)
        let mut sorted_terminals = terminals.clone();
        sorted_terminals.sort_by(|a, b| {
            let prio_cmp = b.priority.cmp(&a.priority);
            if prio_cmp != std::cmp::Ordering::Equal {
                return prio_cmp;
            }

            // If priorities are equal, sort by pattern length (descending)
            let len_a = match &a.pattern {
                Pattern::Str(s) => s.len(),
                Pattern::Regex(_, _) => 0,
            };

            let len_b = match &b.pattern {
                Pattern::Str(s) => s.len(),
                Pattern::Regex(_, _) => 0,
            };

            len_b.cmp(&len_a)
        });

        // Create patterns for the DFA
        let mut patterns = Vec::with_capacity(sorted_terminals.len());

        // Process each terminal
        for (i, terminal) in sorted_terminals.iter().enumerate() {
            index_to_type.insert(i, terminal.name.clone());

            let pattern = match &terminal.pattern {
                Pattern::Str(s) => {
                    // For string literals, escape special regex chars
                    format!("{}", regex::escape(s))
                }
                Pattern::Regex(pattern, flags) => {
                    // For regex patterns, apply flags
                    let mut regex_pattern = String::new();
                    if flags.contains("i") {
                        regex_pattern.push_str("(?i)");
                    }
                    if flags.contains("s") {
                        regex_pattern.push_str("(?s)");
                    }
                    if flags.contains("m") {
                        regex_pattern.push_str("(?m)");
                    }

                    regex_pattern.push_str(pattern);
                    regex_pattern
                }
            };

            patterns.push(pattern);
        }

        // Convert patterns to string references
        let pattern_refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();

        // Build the DFA
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .minimize(true) // Minimize the DFA for better performance
                    .start_kind(StartKind::Anchored),
            ) // Only match from the start of the input
            .build_many(&pattern_refs)
            .map_err(|e| LexerError::RegexError(format!("Failed to build DFA: {}", e)))?;

        Ok(Scanner {
            dfa,
            index_to_type,
            newline_types,
            terminals: sorted_terminals,
            allowed_types,
        })
    }

    /// Match the next token in the input, beginning at position pos, and
    /// return it along with the type of terminal that it is. Look for the
    /// longest possible match.
    pub fn match_token<'a>(&self, text: &'a str, pos: usize) -> Option<(&'a str, &str)> {
        if pos >= text.len() {
            return None;
        }

        let rest = &text[pos..];
        let bytes = rest.as_bytes();

        let config = start::Config::new().anchored(Anchored::Yes);
        let mut state = self.dfa.start_state(&config).expect("no look-around");

        if self.dfa.is_dead_state(state) {
            return None;
        }

        // Keep track of the best match so far
        let mut best_match: Option<(usize, usize)> = None; // (pattern_idx, length)
        let mut current_len = 0;

        // Walk through the DFA state by state
        for &byte in bytes {
            state = self.dfa.next_state(state, byte);

            if self.dfa.is_dead_state(state) {
                break;
            }

            let eoi_state = self.dfa.next_eoi_state(state);
            current_len += 1;

            if self.dfa.is_match_state(eoi_state) {
                let pattern_idx = self.dfa.match_pattern(eoi_state, 0).as_usize();

                match best_match {
                    None => {
                        best_match = Some((pattern_idx, current_len));
                    }
                    Some((_, len)) if current_len > len => {
                        // Prefer longer matches
                        best_match = Some((pattern_idx, current_len));
                    }
                    _ => {} // Keep existing best match
                }
            }
        }

        // Return the best match found as string slices
        if let Some((pattern_idx, match_len)) = best_match {
            if let Some(type_name) = self.index_to_type.get(&pattern_idx) {
                return Some((&rest[..match_len], type_name.as_str()));
            }
        }

        None
    }
}

/// Main Lexer struct.
pub struct Lexer {
    scanner: Option<Scanner>,
    terminals: Vec<TerminalDef>,
    ignore_types: HashSet<String>,
    newline_types: HashSet<String>,
}

impl Lexer {
    pub fn new() -> Self {
        Lexer {
            scanner: None,
            terminals: Vec::new(),
            ignore_types: HashSet::new(),
            newline_types: HashSet::new(),
        }
    }

    pub fn initialize(
        &mut self,
        terminals: Vec<TerminalDef>,
        ignore_types: HashSet<String>,
    ) -> Result<(), LexerError> {
        self.ignore_types = ignore_types;

        // Determine which patterns might contain newlines
        for terminal in &terminals {
            let pattern_str = match &terminal.pattern {
                Pattern::Str(s) => s,
                Pattern::Regex(r, _) => r,
            };

            if pattern_str.contains("\\n")
                || pattern_str.contains("\n")
                || pattern_str.contains("\\s")
                || pattern_str.contains("[^")
                || (pattern_str.contains(".") && pattern_str.contains("(?s"))
            {
                self.newline_types.insert(terminal.name.clone());
            }
        }

        // Create scanner
        match Scanner::new(terminals.clone()) {
            Ok(scanner) => {
                self.scanner = Some(scanner);
                self.terminals = terminals;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Get the next token from text, updating pos, line, and column to the end
    /// (?) of the new token.
    fn next_token(
        &self,
        text: &str,
        mut pos: usize,
        mut line: usize,
        mut column: usize,
        //        last_token: Option<&Token>,
    ) -> Result<(Token, bool), LexerError> {
        // Ensure scanner is initialized
        let scanner = match &self.scanner {
            Some(s) => s,
            None => {
                return Err(LexerError::InitError(
                    "Scanner not initialized. Call initialize() first.".to_string(),
                ));
            }
        };

        // Use a loop instead of recursion to handle ignored tokens
        loop {
            // Check if we're at the end of the text
            // if pos >= text.len() {
            //     return Ok(LexResult::Eof {
            //         pos: text.len(),
            //         line,
            //         column,
            //     });
            // }

            // Try to match next token
            if let Some((value, type_name)) = scanner.match_token(text, pos) {
                // Note: type_name is now &str, not String, so we don't use & prefix
                let ignored = self.ignore_types.contains(type_name);

                // If this token is ignored, update position and continue the loop
                if ignored {
                    let contains_newline = self.newline_types.contains(type_name);

                    // Update line and column information
                    if contains_newline {
                        // Calculate new line and column for tokens with newlines
                        for c in value.chars() {
                            if c == '\n' {
                                line += 1;
                                column = 1;
                            } else {
                                column += 1;
                            }
                        }
                    } else {
                        column += value.chars().count();
                    }

                    // Move position forward and continue the loop
                    pos += value.len();
                    continue;
                }

                // For non-ignored tokens, create and return the token
                let start_pos = pos;
                let end_pos = start_pos + value.len();
                let start_line = line;
                let start_column = column;

                // Calculate end line and column
                let contains_newline = self.newline_types.contains(type_name);
                let (end_line, end_column) = if contains_newline {
                    // Calculate for tokens with newlines
                    let mut current_line = line;
                    let mut current_column = column;

                    for c in value.chars() {
                        if c == '\n' {
                            current_line += 1;
                            current_column = 1;
                        } else {
                            current_column += 1;
                        }
                    }

                    (current_line, current_column)
                } else {
                    // Simple calculation for tokens without newlines
                    (line, column + value.chars().count())
                };

                return Ok((Token {
                    value: value.to_string(),         // Convert &str to String
                    type_name: type_name.to_string(), // Convert &str to String
                    start_pos,
                    end_pos,
                    line: start_line,
                    column: start_column,
                    end_line,
                    end_column,
                }, false));
            } else {
                // No match found. Suppose that everything left is the
                // remainder, which requires us to assume that the string does
                // not diverge from the grammar at any previous
                // point. Otherwise, you could end up with a broken token in
                // the middle of a sequence of otherwise lexable forms (e.g. `1
                // + 0x + 3` in Python). If SynCode is doing its job correctly,
                // such a string should never be generated.
                return Ok((Token {
                    type_name: "".to_string(),
                    value: text[pos..].to_string(),
                    start_pos: pos,
                    end_pos: text.len(),
                    line,
                    column,
                    // TODO: How to compute these values?
                    end_line: usize::MAX,
                    end_column: usize::MAX,
                }, true));
            }
        }
    }

    /// Lex the entire text and return all the tokens along with the remainder.
    ///
    /// The remainder (see sec. 4.2 of the paper) is either the last lexical
    /// token, in the case where the entire input could be lexed, or the
    /// unlexable suffix, in the case where the end of the input could not be
    /// lexed.
    pub fn lex_text(&self, text: &str) -> Result<(Vec<Token>, Token), LexerError> {
        if self.scanner.is_none() {
            return Err(LexerError::InitError(
                "Scanner not initialized. Call initialize() first.".to_string(),
            ));
        }

        // Pre-allocate a reasonably-sized vector based on estimated token density
        let estimated_token_count = text.len() / 8;
        let mut tokens = Vec::with_capacity(estimated_token_count);

        let mut remainder: Token;

        let mut pos = 0;
        let mut line = 1;
        let mut column = 1;

        // Start timing for performance measurement
        let start_time = std::time::Instant::now();

        loop {
            let (new_token, is_remainder) = self.next_token(text, pos, line, column)?;

            if is_remainder {
		// We should quit early, because we've seen all there is to see.
		let elapsed = start_time.elapsed();
		eprintln!(
		    "Rust lexing completed in {:?} - produced {} tokens",
		    elapsed,
		    tokens.len()
		);
                return Ok((tokens, new_token));
            }

	    // Otherwise, continue counting forward to get new tokens.
	    pos = new_token.end_pos;
            line = new_token.end_line;
            column = new_token.end_column;

	    tokens.push(new_token.clone());

	    // The remainder will be the last token we've seen, unless
            // the last thing we see is unlexable.
            remainder = new_token;

	    if pos >= text.len() {
		let elapsed = start_time.elapsed();
		eprintln!(
		    "Rust lexing completed in {:?} - produced {} tokens",
		    elapsed,
		    tokens.len()
		);
		return Ok((tokens, remainder))
	    }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn lexer_initialization() {
        let mut lexer = Lexer::new();

        // Create terminal definitions
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());

        let terminal_defs = vec![
            TerminalDef {
                name: "WORD".to_string(),
                pattern: word_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: space_pattern,
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Check if it was initialized correctly
        assert!(lexer.scanner.is_some());
        assert_eq!(lexer.terminals.len(), 2);
        assert_eq!(lexer.ignore_types.len(), 1);
    }

    #[test]
    fn simple_lexing() {
        let mut lexer = Lexer::new();

        // Create terminal definitions
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());

        let terminal_defs = vec![
            TerminalDef {
                name: "WORD".to_string(),
                pattern: word_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: space_pattern,
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Lex a simple text
        let tokens = lexer.lex_text("hello world").unwrap();

        // Should have 2 tokens: "hello" and "world"
        // (plus one EOF marker)
        assert_eq!(tokens.0.len(), 2);

        assert_eq!(tokens.0[0].value, "hello");
        assert_eq!(tokens.0[0].type_name, "WORD");

        assert_eq!(tokens.0[1].value, "world");
        assert_eq!(tokens.0[1].type_name, "WORD");

        // The remainder should be the last token in the input.
        assert_eq!(tokens.0[1], tokens.1);
    }

    #[test]
    fn complex_string_literals() {
        let mut lexer = Lexer::new();

        // Create pattern for simpler string literals without lookbehind
        let string_pattern =
            Pattern::Regex(r#"("""[^"]*"""|'''[^']*''')"#.to_string(), HashSet::new());

        // Simple word pattern for other text
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        let equals_pattern = Pattern::Str("=".to_string());
        let dot_pattern = Pattern::Str(".".to_string());

        let terminal_defs = vec![
            TerminalDef {
                name: "STRING".to_string(),
                pattern: string_pattern,
                priority: 2,
            },
            TerminalDef {
                name: "WORD".to_string(),
                pattern: word_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "EQUALS".to_string(),
                pattern: equals_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "DOT".to_string(),
                pattern: dot_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: space_pattern,
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Test a simple triple-quoted string
        let text = r#"x = """This is a simple string"""."#;
        let tokens = lexer.lex_text(text).unwrap();

        // Extract token types
        let token_types: Vec<String> = tokens
            .0
            .iter()
            .map(|token| token.type_name.clone())
            .collect();

        // Expected: WORD, EQUALS, STRING, DOT
        assert_eq!(
            token_types,
            vec![
                "WORD".to_string(),
                "EQUALS".to_string(),
                "STRING".to_string(),
                "DOT".to_string()
            ]
        );
    }

    #[test]
    fn numeric_literals() {
        let mut lexer = Lexer::new();

        // Create patterns for different numeric literals
        let dec_number_pattern =
            Pattern::Regex(r"0|[1-9]\d*".to_string(), HashSet::from(["i".to_string()]));
        let hex_number_pattern =
            Pattern::Regex(r"0x[\da-f]*".to_string(), HashSet::from(["i".to_string()]));
        let oct_number_pattern =
            Pattern::Regex(r"0o[0-7]*".to_string(), HashSet::from(["i".to_string()]));
        let bin_number_pattern =
            Pattern::Regex(r"0b[0-1]*".to_string(), HashSet::from(["i".to_string()]));
        let float_number_pattern = Pattern::Regex(
            r"((\d+\.\d*|\.\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))".to_string(),
            HashSet::from(["i".to_string()]),
        );

        // Other patterns
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        let equal_pattern = Pattern::Str("=".to_string());
        let semicolon_pattern = Pattern::Str(";".to_string());

        let terminal_defs = vec![
            TerminalDef {
                name: "FLOAT_NUMBER".to_string(),
                pattern: float_number_pattern,
                priority: 3,
            },
            TerminalDef {
                name: "HEX_NUMBER".to_string(),
                pattern: hex_number_pattern,
                priority: 2,
            },
            TerminalDef {
                name: "OCT_NUMBER".to_string(),
                pattern: oct_number_pattern,
                priority: 2,
            },
            TerminalDef {
                name: "BIN_NUMBER".to_string(),
                pattern: bin_number_pattern,
                priority: 2,
            },
            TerminalDef {
                name: "DEC_NUMBER".to_string(),
                pattern: dec_number_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "WORD".to_string(),
                pattern: word_pattern,
                priority: 0,
            },
            TerminalDef {
                name: "EQUAL".to_string(),
                pattern: equal_pattern,
                priority: 0,
            },
            TerminalDef {
                name: "SEMICOLON".to_string(),
                pattern: semicolon_pattern,
                priority: 0,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: space_pattern,
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Test cases for numeric literals
        let test_cases = vec![
            (
                "x = 42;",
                vec![
                    ("WORD".to_string(), "x".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("DEC_NUMBER".to_string(), "42".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "hex = 0xFF;",
                vec![
                    ("WORD".to_string(), "hex".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("HEX_NUMBER".to_string(), "0xFF".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "oct = 0o77;",
                vec![
                    ("WORD".to_string(), "oct".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("OCT_NUMBER".to_string(), "0o77".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "bin = 0b1010;",
                vec![
                    ("WORD".to_string(), "bin".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("BIN_NUMBER".to_string(), "0b1010".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "pi = 3.14159;",
                vec![
                    ("WORD".to_string(), "pi".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("FLOAT_NUMBER".to_string(), "3.14159".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "e = 2.71e-3;",
                vec![
                    ("WORD".to_string(), "e".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("FLOAT_NUMBER".to_string(), "2.71e-3".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "val = .5;",
                vec![
                    ("WORD".to_string(), "val".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("FLOAT_NUMBER".to_string(), ".5".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
            (
                "sci = 6.022e23;",
                vec![
                    ("WORD".to_string(), "sci".to_string()),
                    ("EQUAL".to_string(), "=".to_string()),
                    ("FLOAT_NUMBER".to_string(), "6.022e23".to_string()),
                    ("SEMICOLON".to_string(), ";".to_string()),
                ],
            ),
        ];

        for (text, expected_tokens) in test_cases {
            let tokens = lexer.lex_text(text).unwrap();

            // Check token types and values (excluding EOF)
            let token_info: Vec<(String, String)> = tokens
                .0
                .iter()
                .map(|token| (token.type_name.clone(), token.value.clone()))
                .collect();

            assert_eq!(token_info, expected_tokens, "Failed for text: {}", text);
        }
    }

    #[test]
    fn remainder_is_lexical_token() {
        // Example from page 10 of the paper. In the case where the string
        // could be lexed all the way to the end, the remainder is the last
        // lexical terminal (because that could change its type with future
        // additions).
        let terminals = vec![
            TerminalDef {
                name: "WORD".to_string(),
                pattern: Pattern::Regex("[a-zA-Z_]\\w*".to_string(), HashSet::new()),
                priority: 2,
            },
            TerminalDef {
                name: "DEC_NUMBER".to_string(),
                pattern: Pattern::Regex("\\d+".to_string(), HashSet::new()),
                priority: 2,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: Pattern::Regex("\\s+".to_string(), HashSet::new()),
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        let mut lexer = Lexer::new();
        lexer.initialize(terminals, ignore_types).unwrap();

        let text = "123 ret";
        let (tokens, remainder) = lexer.lex_text(text).unwrap();

        // We expect:
        // tokens: [123, ret]
        // remainder: ret
        assert_eq!(
            tokens[0],
            Token {
                value: "123".to_string(),
                type_name: "DEC_NUMBER".to_string(),
                start_pos: 0,
                end_pos: 3,
                line: 1,
                column: 1,
                end_line: 1,
                end_column: 4
            }
        );

	assert_eq!(
            tokens[1],
            Token {
                value: "ret".to_string(),
                type_name: "WORD".to_string(),
                start_pos: 4,
                end_pos: 7,
                line: 1,
                column: 5,
                end_line: 1,
                end_column: 8
            }
        );

	assert_eq!(tokens[1], remainder);
    }

    #[test]
    fn remainder_is_not_lexical_token() {
        // In the case where the string could not be lexed all the way to the end, the remainder is unlexed suffix.
        let terminals = vec![
            TerminalDef {
                name: "WORD".to_string(),
                pattern: Pattern::Regex("[a-zA-Z_]\\w*".to_string(), HashSet::new()),
                priority: 2,
            },
            TerminalDef {
                name: "HEX_NUMBER".to_string(),
                pattern: Pattern::Regex(
                    r"0x[\da-f]+".to_string(),
                    HashSet::from(["i".to_string()]),
                ),
                priority: 2,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: Pattern::Regex("\\s+".to_string(), HashSet::new()),
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        let mut lexer = Lexer::new();
        lexer.initialize(terminals, ignore_types).unwrap();

        let text = "return 0x";
        let (tokens, remainder) = lexer.lex_text(text).unwrap();

        // We expect:
        // tokens: [return]
        // remainder: 0x
        assert_eq!(
            tokens[0],
            Token {
                value: "return".to_string(),
                type_name: "WORD".to_string(),
                start_pos: 0,
                end_pos: 6,
                line: 1,
                column: 1,
                end_line: 1,
                end_column: 7
            }
        );

        assert_eq!(
            remainder,
            Token {
                value: "0x".to_string(),
                type_name: "".to_string(),
                start_pos: 7,
                end_pos: 9,
                line: 1,
                column: 8,
                end_line: usize::MAX,
                end_column: usize::MAX
            }
        );
    }

    #[test]
    fn multiline_tracking() {
        let mut lexer = Lexer::new();

        // Create pattern for newlines and other tokens
        let newline_pattern = Pattern::Regex(r"\n".to_string(), HashSet::new());
        let word_pattern = Pattern::Regex("[a-zA-Z_]\\w*".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("[ \t]+".to_string(), HashSet::new());

        let terminal_defs = vec![
            TerminalDef {
                name: "WORD".to_string(),
                pattern: word_pattern,
                priority: 2,
            },
            TerminalDef {
                name: "NEWLINE".to_string(),
                pattern: newline_pattern,
                priority: 1,
            },
            TerminalDef {
                name: "SPACE".to_string(),
                pattern: space_pattern,
                priority: 0,
            },
        ];

        let ignore_types = HashSet::from(["SPACE".to_string()]);

        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();

        // Test multiline text
        let text = "first\nsecond\nthird";

        let tokens = lexer.lex_text(text).unwrap();

        // Check line numbers
        assert_eq!(tokens.0.len(), 5); // 3 words + 2 newlines

        // First word should be on line 1
        assert_eq!(tokens.0[0].line, 1);
        assert_eq!(tokens.0[0].value, "first");

        // After first newline, we should be on line 2
        assert_eq!(tokens.0[2].line, 2);
        assert_eq!(tokens.0[2].value, "second");

        // After second newline, we should be on line 3
        assert_eq!(tokens.0[4].line, 3);
        assert_eq!(tokens.0[4].value, "third");

        // The remainder should be the last token seen.
        assert_eq!(tokens.1, tokens.0[4]);
    }
}
