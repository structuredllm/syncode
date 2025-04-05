// src/python_bindings.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use std::collections::{HashMap, HashSet};

use crate::lexer::{Lexer, LexerError, Token, Pattern, TerminalDef, LexResult};
use crate::parser::{Parser, Rule, ParseConf, ParseTable, ParserError, ParseResult, Action};
use crate::util;


// Python representation of a lexer token
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyLexerToken {
    #[pyo3(get, set)]
    pub value: String,
    #[pyo3(get, set)]
    pub type_name: String,
    #[pyo3(get, set)]
    pub start_pos: usize,
    #[pyo3(get, set)]
    pub end_pos: usize,
    #[pyo3(get, set)]
    pub line: usize,
    #[pyo3(get, set)]
    pub column: usize,
    #[pyo3(get, set)]
    pub end_line: usize,
    #[pyo3(get, set)]
    pub end_column: usize,
}

#[pymethods]
impl PyLexerToken {
    #[new]
    fn new(
        value: String, 
        type_name: String, 
        start_pos: usize, 
        line: usize, 
        column: usize, 
        end_line: usize, 
        end_column: usize, 
        end_pos: usize
    ) -> Self {
        PyLexerToken {
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

    // Create a Python dict from this token
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("type", &self.type_name)?;
        dict.set_item("value", &self.value)?;
        dict.set_item("start_pos", self.start_pos)?;
        dict.set_item("line", self.line)?;
        dict.set_item("column", self.column)?;
        dict.set_item("end_line", self.end_line)?;
        dict.set_item("end_column", self.end_column)?;
        dict.set_item("end_pos", self.end_pos)?;
        Ok(dict)
    }
}

// Convert from Rust Token to Python PyLexerToken
impl From<Token> for PyLexerToken {
    fn from(token: Token) -> Self {
        PyLexerToken {
            value: token.value,
            type_name: token.type_name,
            start_pos: token.start_pos,
            end_pos: token.end_pos,
            line: token.line,
            column: token.column,
            end_line: token.end_line,
            end_column: token.end_column,
        }
    }
}

// Convert from Python PyLexerToken to Rust Token
impl From<PyLexerToken> for Token {
    fn from(token: PyLexerToken) -> Self {
        Token {
            value: token.value,
            type_name: token.type_name,
            start_pos: token.start_pos,
            end_pos: token.end_pos,
            line: token.line,
            column: token.column,
            end_line: token.end_line,
            end_column: token.end_column,
        }
    }
}

// Create a token from a Python dictionary
fn token_from_dict(dict: &PyDict) -> PyResult<PyLexerToken> {
    Ok(PyLexerToken {
        type_name: dict.get_item("type").unwrap().extract()?,
        value: dict.get_item("value").unwrap().extract()?,
        start_pos: dict.get_item("start_pos").unwrap().extract()?,
        line: dict.get_item("line").unwrap().extract()?,
        column: dict.get_item("column").unwrap().extract()?,
        end_line: dict.get_item("end_line").unwrap().extract()?,
        end_column: dict.get_item("end_column").unwrap().extract()?,
        end_pos: dict.get_item("end_pos").unwrap().extract()?,
    })
}

// Create an error dict for Python
fn create_error_dict<'py>(
    py: Python<'py>,
    error_type: &str,
    pos: usize,
    line: usize,
    column: usize,
    allowed: Option<Vec<String>>,
    char: char,
) -> PyResult<&'py PyDict> {
    let result = PyDict::new(py);
    result.set_item("error", error_type)?;
    result.set_item("pos", pos)?;
    result.set_item("line", line)?;
    result.set_item("column", column)?;
    result.set_item("char", char.to_string())?;
    
    if let Some(allowed_types) = allowed {
        if allowed_types.is_empty() {
            result.set_item("allowed", vec!["<END-OF-FILE>"])?;
        } else {
            result.set_item("allowed", allowed_types)?;
        }
    }
    
    Ok(result)
}

// Main lexer wrapper for Python
#[pyclass]
pub struct RustLexer {
    lexer: Lexer,
    callbacks: HashMap<String, PyObject>,
}

#[pymethods]
impl RustLexer {
    #[new]
    fn new() -> Self {
        RustLexer {
            lexer: Lexer::new(),
            callbacks: HashMap::new(),
        }
    }
    
    fn initialize(
        &mut self, 
        py: Python<'_>,
        terminal_defs: Vec<(String, PyObject, i32)>, // (name, pattern, priority)
        ignore_types: HashSet<String>,
        callbacks: HashMap<String, PyObject>,
        _use_bytes: bool, // Kept for compatibility but not used
    ) -> PyResult<()> {
        self.callbacks = callbacks;

        // Process terminal definitions
        let mut terminals = Vec::new();        

        for (name, pattern_obj, priority) in terminal_defs {
            let pattern_dict = pattern_obj.extract::<&PyDict>(py)?;
            
            // Extract type and value
            let pattern_type: String = pattern_dict.get_item("type")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'type' in pattern dictionary"))?
                .extract()?;
                
            let pattern_value: String = pattern_dict.get_item("value")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'value' in pattern dictionary"))?
                .extract()?;
            
            // Extract flags if available
            let flags = if let Some(flags_obj) = pattern_dict.get_item("flags") {
                let flags_set = flags_obj.extract::<&PySet>()?;
                let mut flags_set_rust = HashSet::new();
                for flag in flags_set.iter() {
                    flags_set_rust.insert(flag.extract::<String>()?);
                }
                flags_set_rust
            } else {
                HashSet::new()
            };
            
            // Create the pattern
            let pattern = if pattern_type == "str" {
                Pattern::Str(pattern_value)
            } else if pattern_type == "re" {
                Pattern::Regex(pattern_value, flags)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown pattern type: {}", pattern_type)
                ));
            };
            
            // Add to terminals
            let terminal = TerminalDef {
                name,
                pattern,
                priority,
            };
            
            terminals.push(terminal);
        }
        
        // Initialize the lexer
        match self.lexer.initialize(terminals, ignore_types) {
            Ok(_) => Ok(()),
            Err(LexerError::RegexError(e)) => {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to compile regular expressions: {}", e)
                ))
            },
            Err(LexerError::InitError(e)) => {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Failed to initialize lexer"
            ))
        }
    }
    
    // Modify just the lex_text method in python_bindings.rs to add timing 
    fn lex_text<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Vec<&'py PyDict>> {        
        // Call the Rust lexer directly
        let rust_tokens = match self.lexer.lex_text(text) {
            Ok(tokens) => tokens,
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Lexer error: {:?}", e)
                ));
            }
        };
                
        let mut python_tokens = Vec::new();
        
        for token_result in rust_tokens {
            match token_result {
                LexResult::Token(mut token) => {
                    // Apply callback if exists
                    if let Some(callback) = self.callbacks.get(&token.type_name) {
                        // Convert to Python token
                        let py_token = PyLexerToken::from(token.clone()).to_dict(py)?;
                        let result = callback.call1(py, (py_token,))?;
                        
                        // Convert result back
                        let result_dict = result.downcast::<PyDict>(py)?;
                        let py_token = token_from_dict(result_dict)?;
                        token = py_token.into();
                    }
                    
                    // Convert to Python dict
                    let py_token = PyLexerToken::from(token).to_dict(py)?;
                    python_tokens.push(py_token);
                },
                LexResult::Error { error_type, pos, line, column, allowed, char } => {
                    let error_dict = create_error_dict(py, &error_type, pos, line, column, allowed, char)?;
                    return Ok(vec![error_dict]);
                },
                LexResult::Eof { .. } => {
                    // End of file reached, do not include in token list
                    break;
                }
            }
        }
        Ok(python_tokens)
    }
}

// Main parser wrapper for Python
#[pyclass]
pub struct RustParser {
    parser: Option<Parser<usize>>,
    rules: HashMap<usize, Rule>,
    lexer: Lexer, // Add a lexer field
}

#[pymethods]
impl RustParser {
    #[new]
    fn new() -> Self {
        RustParser {
            parser: None,
            rules: HashMap::new(),
            lexer: Lexer::new(), // Initialize an empty lexer
        }
    }
    
    fn initialize(
        &mut self,
        py: Python<'_>,
        terminal_defs: Vec<(String, PyObject, i32)>, // (name, pattern, priority)
        ignore_types: HashSet<String>,
        use_bytes: bool,
        rules: Vec<(usize, String, Vec<String>)>, // (id, origin, expansion)
        states_dict: &PyDict,
        start_symbol: String,
        start_state: usize,
        end_state: usize,
    ) -> PyResult<()> {
        // Process terminal definitions for the lexer
        let mut terminals = Vec::new();        

        for (name, pattern_obj, priority) in terminal_defs {
            let pattern_dict = pattern_obj.extract::<&PyDict>(py)?;
            
            // Extract type and value
            let pattern_type: String = pattern_dict.get_item("type")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'type' in pattern dictionary"))?
                .extract()?;
                
            let pattern_value: String = pattern_dict.get_item("value")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'value' in pattern dictionary"))?
                .extract()?;
            
            // Extract flags if available
            let flags = if let Some(flags_obj) = pattern_dict.get_item("flags") {
                let flags_set = flags_obj.extract::<&PySet>()?;
                let mut flags_set_rust = HashSet::new();
                for flag in flags_set.iter() {
                    flags_set_rust.insert(flag.extract::<String>()?);
                }
                flags_set_rust
            } else {
                HashSet::new()
            };
            
            // Create the pattern
            let pattern = if pattern_type == "str" {
                Pattern::Str(pattern_value)
            } else if pattern_type == "re" {
                Pattern::Regex(pattern_value, flags)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown pattern type: {}", pattern_type)
                ));
            };
            
            // Add to terminals
            let terminal = TerminalDef {
                name,
                pattern,
                priority,
            };
            
            terminals.push(terminal);
        }
        
        // Initialize the lexer directly in the parser
        match self.lexer.initialize(terminals, ignore_types.clone()) {
            Ok(_) => {},
            Err(LexerError::RegexError(e)) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to compile regular expressions: {}", e)
                ))
            },
            Err(LexerError::InitError(e)) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Failed to initialize lexer"
            ))
        }
        
        // Process rules
        let mut rule_map = HashMap::new();
        for (id, origin, expansion) in rules {
            let rule = Rule::new(id, origin, expansion);
            rule_map.insert(id, rule);
        }
        self.rules = rule_map;
        
        // Convert states dictionary to Rust format
        let mut rust_states_dict = HashMap::new();
        for (state_key, transitions_obj) in states_dict.iter() {
            let state_idx = state_key.extract::<String>()?;
            let transitions_dict = transitions_obj.downcast::<PyDict>()?;
            let mut state_transitions = HashMap::new();
            
            for (symbol, action_obj) in transitions_dict.iter() {
                // eprintln!("State: {}, Symbol: {}", state_idx, symbol);
                // eprintln!("Action: {:?}", action_obj);
                let symbol_str = symbol.extract::<String>()?;
                let action_tuple = action_obj.extract::<(String, String)>()?;
                state_transitions.insert(symbol_str, action_tuple);
            }
            
            rust_states_dict.insert(state_idx, state_transitions);
        }
        
        // Create the parse table
        let parse_table = util::load_parse_table(&self.rules, rust_states_dict, &start_symbol, start_state, end_state);
        
        // Create parser configuration
        match ParseConf::new(parse_table, start_symbol) {
            Ok(conf) => {
                self.parser = Some(Parser::new(conf));
                Ok(())
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to initialize parser: {}", e)
            ))
        }
    }
    
    // Parse text and return whether parsing was successful and how much was consumed
    fn parse_text<'py>(&self, py: Python<'py>, text: &str) -> PyResult<&'py PyDict> {
        let parser = match &self.parser {
            Some(p) => p,
            None => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Parser not initialized. Call initialize() first."
            )),
        };
        
        // Use the lexer that's already initialized in the parser
        let lexer_result = self.lexer.lex_text(text);
        
        let tokens = match lexer_result {
            Ok(tokens) => tokens,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Lexer error: {:?}", e)
            )),
        };
        
        // Parse the tokens
        match parser.parse(&tokens) {
            Ok(result) => {
                let parse_result = PyDict::new(py);
                parse_result.set_item("success", result.success)?;
                parse_result.set_item("consumed", result.consumed)?;
                Ok(parse_result)
            },
            Err(e) => {
                // Convert parser error to a dictionary
                match e {
                    ParserError::UnexpectedToken { token, expected, state_index } => {
                        let error_dict = PyDict::new(py);
                        let token_dict = PyDict::new(py);
                        
                        token_dict.set_item("type", token.type_name)?;
                        token_dict.set_item("value", token.value)?;
                        token_dict.set_item("pos", token.start_pos)?;
                        token_dict.set_item("line", token.line)?;
                        token_dict.set_item("column", token.column)?;
                        
                        error_dict.set_item("error", PyDict::new(py))?;
                        let err_dict = error_dict.get_item("error").unwrap().downcast::<PyDict>()?;
                        err_dict.set_item("type", "unexpected-token")?;
                        err_dict.set_item("token", token_dict)?;
                        err_dict.set_item("expected", expected)?;
                        err_dict.set_item("pos", token.start_pos)?;
                        err_dict.set_item("state_index", state_index)?;
                        
                        Ok(error_dict)
                    },
                    ParserError::UnexpectedEof => {
                        let error_dict = PyDict::new(py);
                        error_dict.set_item("error", PyDict::new(py))?;
                        let err_dict = error_dict.get_item("error").unwrap().downcast::<PyDict>()?;
                        err_dict.set_item("type", "unexpected-eof")?;
                        Ok(error_dict)
                    },
                    ParserError::LexerError { error_type, pos, line, column, char } => {
                        let error_dict = PyDict::new(py);
                        error_dict.set_item("error", PyDict::new(py))?;
                        let err_dict = error_dict.get_item("error").unwrap().downcast::<PyDict>()?;
                        err_dict.set_item("type", "lexer-error")?;
                        err_dict.set_item("error_type", error_type)?;
                        err_dict.set_item("pos", pos)?;
                        err_dict.set_item("line", line)?;
                        err_dict.set_item("column", column)?;
                        err_dict.set_item("char", char.to_string())?;
                        Ok(error_dict)
                    },
                    _ => {
                        let error_dict = PyDict::new(py);
                        error_dict.set_item("error", PyDict::new(py))?;
                        let err_dict = error_dict.get_item("error").unwrap().downcast::<PyDict>()?;
                        err_dict.set_item("type", "parser-error")?;
                        err_dict.set_item("message", format!("{}", e))?;
                        Ok(error_dict)
                    }
                }
            }
        }
    }
}
