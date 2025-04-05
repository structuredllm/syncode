// src/lib.rs
mod lexer;
mod parser;
mod python_bindings;
mod util;

use pyo3::prelude::*;
use python_bindings::{RustLexer, PyLexerToken, RustParser};

/// A Python module implemented in Rust.
#[pymodule]
fn rust_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustLexer>()?;
    m.add_class::<RustParser>()?;
    m.add_class::<PyLexerToken>()?;
    Ok(())
}