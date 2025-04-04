// src/lib.rs
mod lexer;
mod python_bindings;

use pyo3::prelude::*;
use python_bindings::{RustLexer, PyLexerToken};

/// A Python module implemented in Rust.
#[pymodule]
fn rust_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustLexer>()?;
    m.add_class::<PyLexerToken>()?;
    Ok(())
}