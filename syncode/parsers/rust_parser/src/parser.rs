use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

// Import token types from lexer module
use crate::lexer::{Token, LexResult};

// Rule represents a grammar production rule
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rule {
    pub id: usize,
    pub origin: String,
    pub expansion: Vec<String>,
}

impl Rule {
    pub fn new(id: usize, origin: String, expansion: Vec<String>) -> Self {
        Rule {
            id,
            origin,
            expansion,
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.origin,
            self.expansion.join(" ")
        )
    }
}

// Action enum for LR parsing
#[derive(Clone, Debug, PartialEq)]
pub enum Action<S: std::fmt::Debug> {
    Shift(S),
    Reduce(Rule),
    Error,
}

// ParseTable represents an LR(1) parsing table
#[derive(Clone, Debug)]
pub struct ParseTable<S: Clone + Eq + Hash + std::fmt::Debug> {
    pub states: HashMap<S, HashMap<String, Action<S>>>,
    pub start_states: HashMap<String, S>,
    pub end_states: HashMap<String, S>,
}

impl<S: Clone + Eq + Hash + std::fmt::Debug> ParseTable<S> {
    pub fn new() -> Self {
        ParseTable {
            states: HashMap::new(),
            start_states: HashMap::new(),
            end_states: HashMap::new(),
        }
    }
}

// Main data structure to represent the parser result
#[derive(Clone, Debug)]
pub struct ParseResult {
    pub success: bool,
    pub consumed: usize,
}

// TreeNode for the abstract syntax tree
#[derive(Clone, Debug)]
pub enum TreeNode {
    Leaf(Token),
    Node {
        rule: usize,
        rule_name: String,
        children: Vec<TreeNode>,
        meta: Option<HashMap<String, String>>,
    },
}

// ParseConf holds configuration for the parser
#[derive(Clone)]
pub struct ParseConf<S: Clone + Eq + Hash + std::fmt::Debug> {
    pub parse_table: ParseTable<S>,
    pub start: String,
    pub start_state: S,
    pub end_state: S,
}

impl<S: Clone + Eq + Hash + std::fmt::Debug> ParseConf<S> {
    pub fn new(parse_table: ParseTable<S>, start: String) -> Result<Self, ParserError> {
        let start_state = match parse_table.start_states.get(&start) {
            Some(state) => state.clone(),
            None => return Err(ParserError::ConfigError(format!("Start symbol '{}' not found in parse table", start))),
        };

        let end_state = match parse_table.end_states.get(&start) {
            Some(state) => state.clone(),
            None => return Err(ParserError::ConfigError(format!("End state for start symbol '{}' not found", start))),
        };

        Ok(ParseConf {
            parse_table,
            start,
            start_state,
            end_state,
        })
    }
}

// ParserState holds the current state of the parser
#[derive(Clone)]
pub struct ParserState<S: Clone + Eq + Hash + std::fmt::Debug> {
    pub parse_conf: ParseConf<S>,
    pub state_stack: Vec<S>,
    pub token_index: usize,
    pub last_pos: usize,
}

impl<S: Clone + Eq + Hash + std::fmt::Debug> ParserState<S> {
    pub fn new(parse_conf: ParseConf<S>) -> Self {
        ParserState {
            state_stack: vec![parse_conf.start_state.clone()],
            token_index: 0,
            last_pos: 0,
            parse_conf,
        }
    }

    pub fn position(&self) -> &S {
        &self.state_stack[self.state_stack.len() - 1]
    }

    // Feed a token to the parser and process it according to the LR(1) algorithm
    pub fn feed_token(&mut self, token: &Token) -> Result<(), ParserError> {
        let state_stack = &mut self.state_stack;
        let end_state = &self.parse_conf.end_state;
        let states = &self.parse_conf.parse_table.states;
        
        // Update last_pos to the end position of this token
        self.last_pos = token.end_pos;

        loop {
            let state = state_stack.last().unwrap().clone();
            
            // Look up the current state and token type in the parse table
            let action = match states.get(&state).and_then(|transitions| transitions.get(&token.type_name)) {
                Some(action) => action.clone(),
                None => {
                    // Collect the expected token types for error reporting
                    let expected = states.get(&state)
                        .map(|transitions| {
                            transitions.keys()
                                .filter(|key| key.chars().next().map_or(false, |c| c.is_uppercase()))
                                .cloned()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    
                    return Err(ParserError::UnexpectedToken {
                        token: token.clone(),
                        expected,
                        state_index: self.token_index,
                    });
                }
            };
            
            // eprintln!("Current state: {:?}, Token: {:?}", state, token);
            // eprintln!("Action: {:?}", action);
            // eprintln!("Transitions: {:?}", states.get(&state));

            match action {
                Action::Shift(next_state) => {
                    // Just push next state on shift
                    state_stack.push(next_state);
                    return Ok(()); // Not yet accepted
                },
                Action::Reduce(rule) => {
                    // On a reduce, pop states according to the rule expansion length
                    let size = rule.expansion.len();
                    
                    if size > 0 {
                        // Pop the appropriate number of states
                        for _ in 0..size {
                            if state_stack.pop().is_none() {
                                return Err(ParserError::StackUnderflow);
                            }
                        }
                    }
                    
                    // Look up the next state based on current state and rule origin
                    let current_state = state_stack.last().unwrap();
                    let transitions = states.get(current_state).ok_or_else(|| {
                        ParserError::InvalidState(format!("State not found in parse table: {:?}", current_state))
                    })?;
                    
                    let next_action = transitions.get(&rule.origin).ok_or_else(|| {
                        ParserError::InvalidState(format!("No transition for {} in state {:?}", rule.origin, current_state))
                    })?;
                    
                    if let Action::Shift(next_state) = next_action {
                        state_stack.push(next_state.clone());
                    } else {
                        return Err(ParserError::InvalidAction(format!(
                            "Expected Shift after reduce, got {:?}", next_action
                        )));
                    }
                },
                Action::Error => {
                    return Err(ParserError::SyntaxError(format!(
                        "Parser error at token: {} ({})",
                        token.value, token.type_name
                    )));
                }
            }
        }
    }
}

// LR(1) Parser implementation
#[derive(Clone)]
pub struct Parser<S: Clone + Eq + Hash + std::fmt::Debug> {
    pub conf: ParseConf<S>,
}

impl<S: Clone + Eq + Hash + std::fmt::Debug> Parser<S> {
    pub fn new(conf: ParseConf<S>) -> Self {
        Parser { conf }
    }
    
    // Parse tokens without building a tree, just validating
    pub fn parse(&self, tokens: &[LexResult]) -> Result<ParseResult, ParserError> {
        let start_time = std::time::Instant::now();
        let mut state = ParserState::new(self.conf.clone());
        let token_count = tokens.len();
        
        // Process all tokens
        for (i, lex_result) in tokens.iter().enumerate() {
            state.token_index = i;
            
            match lex_result {
                LexResult::Token(token) => {
                    let is_last = i == token_count - 1;

                    match state.feed_token(token) {
                        // If last token then we should return the last token else we continue
                        Ok(()) => {
                            if is_last {
                                eprintln!("Parsing completed in {:?}", start_time.elapsed());
                                return Ok(ParseResult { success: true, consumed: state.last_pos });
                            } else {
                                continue;
                            }
                        },

                        // if there is an error during last token then we return the last token and success true
                        // else we return the error
                        Err(e) => {
                            if is_last {
                                return Ok(ParseResult { success: true, consumed: state.last_pos });
                            } else {
                                return Err(e);
                            }
                        }
                    }
                },
                LexResult::Error { error_type, pos, line, column, allowed: _, char } => {
                    return Err(ParserError::LexerError {
                        error_type: error_type.clone(),
                        pos: *pos,
                        line: *line,
                        column: *column,
                        char: *char,
                    });
                },
                LexResult::Eof { pos, line, column } => {
                    // Process EOF token specially
                    let eof_token = Token {
                        value: "".to_string(),
                        type_name: "$END".to_string(),
                        start_pos: *pos,
                        end_pos: *pos,
                        line: *line,
                        column: *column,
                        end_line: *line,
                        end_column: *column,
                    };
                }
            }
        }
        
        
        // If we get here, we've consumed all tokens but not accepted
        Ok(ParseResult { success: false, consumed: state.last_pos })
    }
}

// Error types for the parser
#[derive(Debug, Clone)]
pub enum ParserError {
    UnexpectedToken {
        token: Token,
        expected: Vec<String>,
        state_index: usize,
    },
    UnexpectedEof,
    LexerError {
        error_type: String,
        pos: usize,
        line: usize,
        column: usize,
        char: char,
    },
    StackUnderflow,
    EmptyStack,
    InvalidState(String),
    InvalidAction(String),
    SyntaxError(String),
    ConfigError(String),
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::UnexpectedToken { token, expected, state_index: _ } => {
                write!(f, "Unexpected token '{}' (type: {}) at line {}, column {}. Expected one of: {:?}",
                    token.value, token.type_name, token.line, token.column, expected)
            },
            ParserError::UnexpectedEof => {
                write!(f, "Unexpected end of input")
            },
            ParserError::LexerError { error_type, pos, line, column, char } => {
                write!(f, "Lexer error: {} at position {} (line {}, column {}): '{}'",
                    error_type, pos, line, column, char)
            },
            ParserError::StackUnderflow => {
                write!(f, "Parser stack underflow")
            },
            ParserError::EmptyStack => {
                write!(f, "Parser stack is empty")
            },
            ParserError::InvalidState(msg) => {
                write!(f, "Invalid parser state: {}", msg)
            },
            ParserError::InvalidAction(msg) => {
                write!(f, "Invalid parser action: {}", msg)
            },
            ParserError::SyntaxError(msg) => {
                write!(f, "Syntax error: {}", msg)
            },
            ParserError::ConfigError(msg) => {
                write!(f, "Parser configuration error: {}", msg)
            },
        }
    }
}
