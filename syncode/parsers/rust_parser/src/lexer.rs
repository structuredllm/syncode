// src/lexer.rs
use std::collections::{HashMap, HashSet};
use std::fmt;
use regex::Regex;
use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::{Anchored, Input, util::start};

// Token struct to represent lexer tokens
#[derive(Clone, Debug)]
pub struct Token {
    pub value: String,
    pub type_name: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

impl Token {
    pub fn new(
        value: String, 
        type_name: String, 
        start_pos: usize, 
        line: usize, 
        column: usize, 
        end_line: usize, 
        end_column: usize, 
        end_pos: usize
    ) -> Self {
        Token {
            value,
            type_name,
            start_pos,
            end_pos,
            line,
            column,
            end_line,
            end_column,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({}, {})", self.type_name, self.value)
    }
}

// Pattern types to match Lark's patterns
#[derive(Clone, Debug)]
pub enum Pattern {
    Str(String),
    Regex(String, HashSet<String>), // regex pattern and flags
}

// Terminal definition matching Lark's TerminalDef
#[derive(Clone, Debug)]
pub struct TerminalDef {
    pub name: String,
    pub pattern: Pattern,
    pub priority: i32,
}

// LineCounter to keep track of line and column numbers
#[derive(Clone, Debug)]
pub struct LineCounter {
    pub char_pos: usize,
    pub line: usize,
    pub column: usize,
    pub line_start_pos: usize,
    pub newline_char: String,
}

impl LineCounter {
    pub fn new(newline_char: &str) -> Self {
        LineCounter {
            char_pos: 0,
            line: 1,
            column: 1,
            line_start_pos: 0,
            newline_char: newline_char.to_string(),
        }
    }
    
    pub fn feed(&mut self, text: &str, test_newline: bool) {
        if test_newline {
            // Fast newline counting
            let bytes = text.as_bytes();
            let newline_byte = self.newline_char.as_bytes()[0]; // Assuming single-byte newline
            
            let mut newlines = 0;
            let mut last_newline_pos = None;
            
            for (i, &b) in bytes.iter().enumerate() {
                if b == newline_byte {
                    newlines += 1;
                    last_newline_pos = Some(i);
                }
            }
            
            if newlines > 0 {
                self.line += newlines;
                if let Some(pos) = last_newline_pos {
                    self.line_start_pos = self.char_pos + pos + 1;
                }
            }
        }
        
        self.char_pos += text.len();
        self.column = self.char_pos - self.line_start_pos + 1;
    }
}

// LexerState to track current lexer state
#[derive(Clone, Debug)]
pub struct LexerState {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub line_ctr: LineCounter,
    pub last_token: Option<Token>,
}

// Error types
#[derive(Debug, Clone)]
pub enum LexerError {
    UnexpectedChar { pos: usize, line: usize, column: usize, allowed: Vec<String>, char: char },
    Eof { pos: usize, line: usize, column: usize },
    InitError(String),
    RegexError(String),
}

// Result enum to represent either a token or an error
#[derive(Debug, Clone)]
pub enum LexResult {
    Token(Token),
    Error { error_type: String, pos: usize, line: usize, column: usize, allowed: Option<Vec<String>>, char: char },
    Eof { pos: usize, line: usize, column: usize },
}

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
            
            if pattern_str.contains("\\n") || pattern_str.contains("\n") || 
               pattern_str.contains("\\s") || pattern_str.contains("[^") ||
               (pattern_str.contains(".") && pattern_str.contains("(?s")) {
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
                },
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
            .configure(dense::Config::new()
                .minimize(true)    // Minimize the DFA for better performance
                .start_kind(StartKind::Anchored))   // Only match from the start of the input
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

    // Updated match_token implementation
    
// Updated match_token that returns string slices
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
                },
                Some((_, len)) if current_len > len => {
                    // Prefer longer matches
                    best_match = Some((pattern_idx, current_len));
                },
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

// Main Lexer implementation
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
    
    pub fn initialize(&mut self, terminals: Vec<TerminalDef>, ignore_types: HashSet<String>) -> Result<(), LexerError> {
        self.ignore_types = ignore_types;
        
        // Determine which patterns might contain newlines
        for terminal in &terminals {
            let pattern_str = match &terminal.pattern {
                Pattern::Str(s) => s,
                Pattern::Regex(r, _) => r,
            };
            
            if pattern_str.contains("\\n") || pattern_str.contains("\n") || 
               pattern_str.contains("\\s") || pattern_str.contains("[^") ||
               (pattern_str.contains(".") && pattern_str.contains("(?s")) {
                self.newline_types.insert(terminal.name.clone());
            }
        }
        
        // Create scanner
        match Scanner::new(terminals.clone()) {
            Ok(scanner) => {
                self.scanner = Some(scanner);
                self.terminals = terminals;
                Ok(())
            },
            Err(e) => Err(e)
        }
    }
    
    pub fn next_token(&self, text: &str, mut pos: usize, mut line: usize, mut column: usize, 
        last_token: Option<&Token>) -> Result<LexResult, LexerError> {
        // Ensure scanner is initialized
        let scanner = match &self.scanner {
        Some(s) => s,
        None => return Err(LexerError::InitError("Scanner not initialized. Call initialize() first.".to_string())),
        };

        // Use a loop instead of recursion to handle ignored tokens
        loop {
            // Check if we're at the end of the text
            if pos >= text.len() {
                return Ok(LexResult::Eof {
                    pos: text.len(),
                    line,
                    column,
                });
            }

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
            
            let token = Token {
                value: value.to_string(),           // Convert &str to String
                type_name: type_name.to_string(),   // Convert &str to String
                start_pos,
                end_pos,
                line: start_line,
                column: start_column,
                end_line,
                end_column,
            };
            
            return Ok(LexResult::Token(token));
            } else {
            // No match found - unexpected character
            let allowed_types: Vec<String> = scanner.allowed_types
                .difference(&self.ignore_types)
                .cloned()
                .collect();
            
            return Ok(LexResult::Error {
                error_type: "unexpected-char".to_string(),
                char: text.chars().nth(pos).unwrap_or_default(),
                pos,
                line,
                column,
                allowed: Some(allowed_types),
            });
            }
        }
    }
    
    // Lex the entire text and return all tokens
    pub fn lex_text(&self, text: &str) -> Result<Vec<LexResult>, LexerError> {
        if self.scanner.is_none() {
            return Err(LexerError::InitError("Scanner not initialized. Call initialize() first.".to_string()));
        }
        
        // Pre-allocate a reasonably-sized vector based on estimated token density
        let estimated_token_count = text.len() / 8;
        let mut tokens = Vec::with_capacity(estimated_token_count);
        
        let mut pos = 0;
        let mut line = 1;
        let mut column = 1;
        let mut last_token = None;
        
        // Start timing for performance measurement
        let start_time = std::time::Instant::now();
    
        while pos < text.len() {
            // Pass a reference to last_token if it exists
            let last_token_ref = last_token.as_ref();
            
            match self.next_token(text, pos, line, column, last_token_ref)? {
                LexResult::Token(token) => {
                    // Update position for next token
                    pos = token.end_pos;
                    line = token.end_line;
                    column = token.end_column;
                    
                    // Store token for next iteration - take ownership
                    last_token = Some(token.clone());
                    
                    tokens.push(LexResult::Token(token));
                },
                error @ LexResult::Error { .. } => {
                    // Return the error
                    tokens.push(error);
                    break;
                },
                eof @ LexResult::Eof { .. } => {
                    // End of file reached
                    tokens.push(eof);
                    break;
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        eprintln!("Rust lexing completed in {:?} - produced {} tokens", elapsed, tokens.len());
    
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    #[test]
    fn test_token_creation() {
        let token = Token::new(
            "hello".to_string(),
            "WORD".to_string(),
            0,
            1,
            1,
            1,
            6,
            5
        );
        
        assert_eq!(token.value, "hello");
        assert_eq!(token.type_name, "WORD");
        assert_eq!(token.start_pos, 0);
        assert_eq!(token.line, 1);
        assert_eq!(token.column, 1);
        assert_eq!(token.end_line, 1);
        assert_eq!(token.end_column, 6);
        assert_eq!(token.end_pos, 5);
    }
    
    #[test]
    fn test_lexer_initialization() {
        let mut lexer = Lexer::new();
        
        // Create terminal definitions
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        
        let terminal_defs = vec![
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 1 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
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
    fn test_simple_lexing() {
        let mut lexer = Lexer::new();
        
        // Create terminal definitions
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        
        let terminal_defs = vec![
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 1 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
        ];
        
        let ignore_types = HashSet::from(["SPACE".to_string()]);
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Lex a simple text
        let tokens = lexer.lex_text("hello world").unwrap();
        
        // Should have 2 tokens: "hello" and "world"
        // (plus one EOF marker)
        assert_eq!(tokens.len(), 3);
        
        // Check the first token
        match &tokens[0] {
            LexResult::Token(token) => {
                assert_eq!(token.value, "hello");
                assert_eq!(token.type_name, "WORD");
            },
            _ => panic!("Expected token, got something else"),
        }
        
        // Check the second token
        match &tokens[1] {
            LexResult::Token(token) => {
                assert_eq!(token.value, "world");
                assert_eq!(token.type_name, "WORD");
            },
            _ => panic!("Expected token, got something else"),
        }
        
        // Check EOF
        match &tokens[2] {
            LexResult::Eof { .. } => {},
            _ => panic!("Expected EOF, got something else"),
        }
    }
    
    #[test]
    fn test_lexer_error_handling() {
        let mut lexer = Lexer::new();
        
        // Create terminal definitions
        let word_pattern = Pattern::Regex("[a-zA-Z]+".to_string(), HashSet::new());
        
        let terminal_defs = vec![
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 1 },
        ];
        
        let ignore_types = HashSet::new();
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Try to lex text with an invalid character
        let result = lexer.lex_text("abc123").unwrap();
        
        // Should have token "abc" and then an error
        assert_eq!(result.len(), 2);
        
        match &result[0] {
            LexResult::Token(token) => {
                assert_eq!(token.value, "abc");
            },
            _ => panic!("Expected token, got something else"),
        }
        
        match &result[1] {
            LexResult::Error { error_type, pos, .. } => {
                assert_eq!(error_type, "unexpected-char");
                assert_eq!(*pos, 3);
            },
            _ => panic!("Expected error, got something else"),
        }
    }

    #[test]
    fn test_complex_string_literals() {
        let mut lexer = Lexer::new();
        
        // Create pattern for simpler string literals without lookbehind
        let string_pattern = Pattern::Regex(r#"("""[^"]*"""|'''[^']*''')"#.to_string(), HashSet::new());
        
        // Simple word pattern for other text
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        let equals_pattern = Pattern::Str("=".to_string());
        let dot_pattern = Pattern::Str(".".to_string());
        
        let terminal_defs = vec![
            TerminalDef { name: "STRING".to_string(), pattern: string_pattern, priority: 2 },
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 1 },
            TerminalDef { name: "EQUALS".to_string(), pattern: equals_pattern, priority: 1 },
            TerminalDef { name: "DOT".to_string(), pattern: dot_pattern, priority: 1 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
        ];
        
        let ignore_types = HashSet::from(["SPACE".to_string()]);
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Test a simple triple-quoted string
        let text = r#"x = """This is a simple string"""."#;
        let tokens = lexer.lex_text(text).unwrap();
        
        // Extract token types
        let token_types: Vec<String> = tokens.iter()
            .filter_map(|t| match t {
                LexResult::Token(token) => Some(token.type_name.clone()),
                _ => None,
            })
            .collect();
        
        // Expected: WORD, EQUALS, STRING, DOT
        assert_eq!(token_types, vec![
            "WORD".to_string(), 
            "EQUALS".to_string(), 
            "STRING".to_string(), 
            "DOT".to_string()
        ]);
    }
    
    #[test]
    fn test_numeric_literals() {
        let mut lexer = Lexer::new();
        
        // Create patterns for different numeric literals
        let dec_number_pattern = Pattern::Regex(r"0|[1-9]\d*".to_string(), HashSet::from(["i".to_string()]));
        let hex_number_pattern = Pattern::Regex(r"0x[\da-f]*".to_string(), HashSet::from(["i".to_string()]));
        let oct_number_pattern = Pattern::Regex(r"0o[0-7]*".to_string(), HashSet::from(["i".to_string()]));
        let bin_number_pattern = Pattern::Regex(r"0b[0-1]*".to_string(), HashSet::from(["i".to_string()]));
        let float_number_pattern = Pattern::Regex(
            r"((\d+\.\d*|\.\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))".to_string(), 
            HashSet::from(["i".to_string()])
        );
        
        // Other patterns
        let word_pattern = Pattern::Regex("\\w+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        let equal_pattern = Pattern::Str("=".to_string());
        let semicolon_pattern = Pattern::Str(";".to_string());
        
        let terminal_defs = vec![
            TerminalDef { name: "FLOAT_NUMBER".to_string(), pattern: float_number_pattern, priority: 3 },
            TerminalDef { name: "HEX_NUMBER".to_string(), pattern: hex_number_pattern, priority: 2 },
            TerminalDef { name: "OCT_NUMBER".to_string(), pattern: oct_number_pattern, priority: 2 },
            TerminalDef { name: "BIN_NUMBER".to_string(), pattern: bin_number_pattern, priority: 2 },
            TerminalDef { name: "DEC_NUMBER".to_string(), pattern: dec_number_pattern, priority: 1 },
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 0 },
            TerminalDef { name: "EQUAL".to_string(), pattern: equal_pattern, priority: 0 },
            TerminalDef { name: "SEMICOLON".to_string(), pattern: semicolon_pattern, priority: 0 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
        ];
        
        let ignore_types = HashSet::from(["SPACE".to_string()]);
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Test cases for numeric literals
        let test_cases = vec![
            ("x = 42;", vec![("WORD".to_string(), "x".to_string()), ("EQUAL".to_string(), "=".to_string()), ("DEC_NUMBER".to_string(), "42".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("hex = 0xFF;", vec![("WORD".to_string(), "hex".to_string()), ("EQUAL".to_string(), "=".to_string()), ("HEX_NUMBER".to_string(), "0xFF".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("oct = 0o77;", vec![("WORD".to_string(), "oct".to_string()), ("EQUAL".to_string(), "=".to_string()), ("OCT_NUMBER".to_string(), "0o77".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("bin = 0b1010;", vec![("WORD".to_string(), "bin".to_string()), ("EQUAL".to_string(), "=".to_string()), ("BIN_NUMBER".to_string(), "0b1010".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("pi = 3.14159;", vec![("WORD".to_string(), "pi".to_string()), ("EQUAL".to_string(), "=".to_string()), ("FLOAT_NUMBER".to_string(), "3.14159".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("e = 2.71e-3;", vec![("WORD".to_string(), "e".to_string()), ("EQUAL".to_string(), "=".to_string()), ("FLOAT_NUMBER".to_string(), "2.71e-3".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("val = .5;", vec![("WORD".to_string(), "val".to_string()), ("EQUAL".to_string(), "=".to_string()), ("FLOAT_NUMBER".to_string(), ".5".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
            ("sci = 6.022e23;", vec![("WORD".to_string(), "sci".to_string()), ("EQUAL".to_string(), "=".to_string()), ("FLOAT_NUMBER".to_string(), "6.022e23".to_string()), ("SEMICOLON".to_string(), ";".to_string())]),
        ];
        
        for (text, expected_tokens) in test_cases {
            let tokens = lexer.lex_text(text).unwrap();
            
            // Check token types and values (excluding EOF)
            let token_info: Vec<(String, String)> = tokens.iter()
                .filter_map(|t| match t {
                    LexResult::Token(token) => Some((token.type_name.clone(), token.value.clone())),
                    _ => None,
                })
                .collect();
            
            assert_eq!(token_info, expected_tokens, "Failed for text: {}", text);
        }
    }
    
    #[test]
    fn test_error_recovery() {
        let mut lexer = Lexer::new();
        
        // Define patterns but exclude some characters to force errors
        let word_pattern = Pattern::Regex("[a-zA-Z_]\\w*".to_string(), HashSet::new());
        let number_pattern = Pattern::Regex("\\d+".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("\\s+".to_string(), HashSet::new());
        
        // Note: We don't define patterns for special characters to test error handling
        
        let terminal_defs = vec![
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 2 },
            TerminalDef { name: "NUMBER".to_string(), pattern: number_pattern, priority: 1 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
        ];
        
        let ignore_types = HashSet::from(["SPACE".to_string()]);
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Test text with characters that should cause errors
        let text = "variable = 123; // comment";
        
        let tokens = lexer.lex_text(text).unwrap();
        
        // We expect:
        // - "variable" as WORD
        // - Error at '='
        let first_token = &tokens[0];
        let second_result = &tokens[1];
        
        match first_token {
            LexResult::Token(token) => {
                assert_eq!(token.value, "variable");
                assert_eq!(token.type_name, "WORD");
            },
            _ => panic!("Expected token, got something else"),
        }
        
        match second_result {
            LexResult::Error { error_type, pos, .. } => {
                assert_eq!(error_type, "unexpected-char");
                assert_eq!(*pos, 9); // Position of '='
            },
            _ => panic!("Expected error, got something else"),
        }
    }
    
    #[test]
    fn test_multiline_tracking() {
        let mut lexer = Lexer::new();
        
        // Create pattern for newlines and other tokens
        let newline_pattern = Pattern::Regex(r"\n".to_string(), HashSet::new());
        let word_pattern = Pattern::Regex("[a-zA-Z_]\\w*".to_string(), HashSet::new());
        let space_pattern = Pattern::Regex("[ \t]+".to_string(), HashSet::new());
        
        let terminal_defs = vec![
            TerminalDef { name: "WORD".to_string(), pattern: word_pattern, priority: 2 },
            TerminalDef { name: "NEWLINE".to_string(), pattern: newline_pattern, priority: 1 },
            TerminalDef { name: "SPACE".to_string(), pattern: space_pattern, priority: 0 },
        ];
        
        let ignore_types = HashSet::from(["SPACE".to_string()]);
        
        // Initialize the lexer
        lexer.initialize(terminal_defs, ignore_types).unwrap();
        
        // Test multiline text
        let text = "first\nsecond\nthird";
        
        let tokens = lexer.lex_text(text).unwrap();
        
        // Filter out just the tokens
        let real_tokens: Vec<Token> = tokens.iter()
            .filter_map(|t| match t {
                LexResult::Token(token) => Some(token.clone()),
                _ => None,
            })
            .collect();
        
        // Check line numbers
        assert_eq!(real_tokens.len(), 5); // 3 words + 2 newlines
        
        // First word should be on line 1
        assert_eq!(real_tokens[0].line, 1);
        assert_eq!(real_tokens[0].value, "first");
        
        // After first newline, we should be on line 2
        assert_eq!(real_tokens[2].line, 2);
        assert_eq!(real_tokens[2].value, "second");
        
        // After second newline, we should be on line 3
        assert_eq!(real_tokens[4].line, 3);
        assert_eq!(real_tokens[4].value, "third");
    }
}