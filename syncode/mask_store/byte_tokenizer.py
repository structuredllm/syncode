"""bytetokenizerr.py: A flexible wrapper to handle different HuggingFace tokenizer types.

This module provides a ByteTokenizer class that can adapt to different types of tokenizers:
- RAW: Tokens in original form without processing (e.g., tiktoken)
- BYTE_FALLBACK: Tokens encoded with byte-fallback conversion (e.g., Llama-2)
- BYTE_LEVEL: Tokens encoded with byte-to-unicode conversion (e.g., GPT-2, Llama-3)

The ByteTokenizer allows working with these tokenizers in a consistent byte-level manner.
"""
from transformers import AutoTokenizer
from functools import reduce, cache
from enum import Enum
import re
import dataclasses
from typing import List, Optional, Dict, Tuple, Union, Any
import time


class VocabType(Enum):
    """The type of vocabulary used by the tokenizer."""
    RAW = 0
    BYTE_FALLBACK = 1
    BYTE_LEVEL = 2


def bytes_to_unicode():
    """
    Returns a mapping between utf-8 bytes and unicode strings.
    Used for byte-level BPE tokenization (GPT-2 style).
    
    This makes the tokens representable in a standard text editor by mapping
    bytes to printable unicode characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@cache
def enbyte_bytelevel(token: str) -> bytes:
    """Turn a byte-level BPE token into the corresponding bytes.

    Example:
    --------
    >>> enbyte_bytelevel('âĪ')
    b'\xe2\x88'
    """
    dict_bytes = {v: k for k, v in bytes_to_unicode().items()}
    # Replace non-ASCII special tokens with ASCII for compatibility
    token = token.replace('｜', '|').replace('▁', '_')
    try:
        return bytes([dict_bytes[c] for c in token])
    except KeyError as e:
        # For characters not in the mapping, attempt a direct encoding
        # This handles special tokens not covered by the mapping
        return token.encode('utf-8')


@cache
def enbyte_bytefallback(token: str) -> bytes:
    """Turn a byte-fallback token into the corresponding bytes.
    
    Example:
    --------
    >>> enbyte_bytefallback('<0x1B>')
    b'\x1b'
    >>> enbyte_bytefallback('▁apple')
    b' apple'
    """
    # Handle byte fallback format like <0x1B>
    if re.match(r'<0x[0-9A-F]{2}>', token):
        byte_value = int(token[3:5], 16)
        return bytes([byte_value])
    
    # Handle space prefix format - Gemma/Llama style with '▁'
    if token.startswith('▁'):
        # If it's just a single or multiple '▁', it's just spaces (common in indentation)
        if set(token) == {'▁'}:
            return b' ' * len(token)
        # Otherwise it's a space followed by content
        return b' ' + token[1:].encode('utf-8')
        
    return token.encode('utf-8')


@cache
def enbyte_raw(token: bytes) -> bytes:
    """Turn a raw token directly into bytes.
    
    Example:
    --------
    >>> enbyte_raw(b'hello')
    b'hello'
    """
    return token


def debyte_bytelevel(array: bytes) -> list[str]:
    """Turn bytes into a list of corresponding code points for byte-level BPE.

    Example:
    --------
    >>> debyte_bytelevel(b'\xe2\x88')
    ['â', 'Ī']
    """
    byte_dict = bytes_to_unicode()
    return [byte_dict[b] for b in array]


def detect_vocab_type(tokenizer):
    """
    Detect the vocabulary type of a tokenizer.
    
    Returns:
    --------
    VocabType: The detected vocabulary type
    """
    vocab = tokenizer.get_vocab()
    
    # Check for byte fallback pattern (e.g., <0x0A> tokens)
    if any(token.startswith('<0x') and token.endswith('>') for token in vocab):
        return VocabType.BYTE_FALLBACK
    
    # Check for tiktoken-based tokenizers
    if hasattr(tokenizer, 'tokenizer') and 'tiktoken' in str(type(tokenizer.tokenizer)):
        return VocabType.RAW
    
    # Check filename pattern for tiktoken tokenizers
    if (hasattr(tokenizer, 'vocab_files_names') and 
        'vocab_file' in tokenizer.vocab_files_names and 
        'tiktoken' in tokenizer.vocab_files_names['vocab_file']):
        return VocabType.RAW
    
    # Look for the "Ġ" character which is common in byte-level BPE tokenizers
    if any(('Ġ' in token or '▁' in token) for token in vocab):
        return VocabType.BYTE_LEVEL
        
    # Default to RAW type if no specific patterns are detected
    return VocabType.RAW


class ByteTokenizer:
    """A class to convert tokenizers of different types to work at the byte level."""

    def __init__(self, tokenizer, vocab_type=None):
        self.tokenizer = tokenizer
        
        # Detect vocab type if not provided
        if vocab_type is None:
            self.vocab_type = detect_vocab_type(tokenizer)
        else:
            self.vocab_type = vocab_type
            
        # Select appropriate encoding function based on vocab type
        if self.vocab_type == VocabType.BYTE_LEVEL:
            self.enbyte_fn = enbyte_bytelevel
        elif self.vocab_type == VocabType.BYTE_FALLBACK:
            self.enbyte_fn = enbyte_bytefallback
        else:  # RAW
            self.enbyte_fn = enbyte_raw
            
        # Build vocabulary mappings
        self.vocab = tokenizer.get_vocab()
        self.byte_vocab = {}
        self.vocab_byte = {}
        
        # Create mappings for all vocabulary items
        for token, token_id in self.vocab.items():
            try:
                byte_token = self.enbyte_fn(token)
                self.byte_vocab[byte_token] = token_id
                self.vocab_byte[token_id] = byte_token
            except Exception as e:
                # Skip problematic tokens but log them
                print(f"Warning: Could not convert token '{token}' to bytes: {e}")
                
        # Cache special token IDs as a set for faster lookups
        self.special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

        # Added tokens are typically special tokens
        # if added_tokens_decoder is not None self.tokenizer.added_tokens_decoder.keys()
        # to special_token_ids
        if hasattr(tokenizer, "added_tokens_decoder"):
            self.special_token_ids.update(tokenizer.added_tokens_decoder.keys())


    @classmethod
    def from_pretrained(cls, model_id, vocab_type=None):
        """
        Create a ByteTokenizer from a pre-trained model ID.
        
        Parameters:
        -----------
        model_id: str
            The HuggingFace model ID
        vocab_type: VocabType, optional
            The vocabulary type to use, if known
            
        Returns:
        --------
        ByteTokenizer: A new ByteTokenizer instance
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(tokenizer, vocab_type)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> bytes:
        """
        Decode token_ids to bytes.
        
        Parameters:
        -----------
        token_ids: list[int]
            List of token IDs to decode
        skip_special_tokens: bool, default=False
            Whether to skip special tokens in the decoded output
            
        Returns:
        --------
        bytes: The decoded bytes
        
        Examples:
        ---------
        >>> tok = ByteTokenizer.from_pretrained('gpt2')
        >>> tok.decode([19526, 254, 25001, 121, 28938, 245, 171, 120, 253]).decode('utf-8')
        '你好吗？'
        """
        if not token_ids:
            return b''
        
        # Use a mutable bytearray for faster concatenation
        result = bytearray()
        vocab_byte = self.vocab_byte
        
        # Fast path: no special token handling needed
        if not skip_special_tokens or not self.special_token_ids:
            for token_id in token_ids:
                try:
                    # Try/except is faster than 'in' check for dictionary access
                    result.extend(vocab_byte[token_id])
                except KeyError:
                    # Fall back to tokenizer for unknown tokens
                    text = self.tokenizer.decode([token_id])
                    result.extend(text.encode('utf-8'))
            return bytes(result)
        
        # Path with special token handling
        special_token_ids = self.special_token_ids
        for token_id in token_ids:
            if token_id in special_token_ids:
                continue
            
            try:
                result.extend(vocab_byte[token_id])
            except KeyError:
                text = self.tokenizer.decode([token_id])
                result.extend(text.encode('utf-8'))
                
        return bytes(result)

    def encode(self, byte_text: bytes) -> list[int]:
        """
        Encode bytes to token_ids.
        
        Parameters:
        -----------
        byte_text: bytes
            Bytes to encode
            
        Returns:
        --------
        list[int]: List of token IDs
        
        Examples:
        ---------
        >>> tok = ByteTokenizer.from_pretrained('gpt2')
        >>> tok.encode('你好吗？'.encode())
        [19526, 254, 25001, 121, 28938, 245, 171, 120, 253]
        """
        # For byte-level tokenizers, we use a greedy algorithm
        input_ids = []
        original_byte_text = byte_text
        
        # Handle RAW tokenizers differently
        if self.vocab_type == VocabType.RAW:
            # For RAW tokenizers, we can decode the bytes and use the original tokenizer
            try:
                text = byte_text.decode('utf-8')
                return self.tokenizer.encode(text, add_special_tokens=False)
            except UnicodeDecodeError:
                # If we can't decode as UTF-8, fall back to our byte-level logic
                pass
        
        # Greedy tokenization for byte-level and byte-fallback tokenizers
        while byte_text:
            matched = False
            # Try largest prefix first
            for i in range(len(byte_text), 0, -1):
                prefix = byte_text[:i]
                if prefix in self.byte_vocab:
                    input_ids.append(self.byte_vocab[prefix])
                    byte_text = byte_text[i:]
                    matched = True
                    break
            
            # If no match found for any prefix, add the first byte as an unknown token
            # and continue with the rest
            if not matched:
                # Try to get the unknown token ID from the tokenizer
                unk_token_id = self.tokenizer.unk_token_id
                if unk_token_id is None:
                    # If no explicit unknown token, use a default
                    unk_token_id = 0
                
                input_ids.append(unk_token_id)
                byte_text = byte_text[1:]
                
        return input_ids

    def encode_batch(self, batch_byte_text: list[bytes]) -> list[list[int]]:
        """
        Encode a batch of bytes to token_ids.
        
        Parameters:
        -----------
        batch_byte_text: list[bytes]
            List of bytes sequences to encode
            
        Returns:
        --------
        list[list[int]]: List of lists of token IDs
        """
        return [self.encode(text) for text in batch_byte_text]

    def batched_decode(self, token_id_batches: List[List[int]], skip_special_tokens: bool = False) -> List[bytes]:
        """
        Decode multiple batches of token IDs.
        
        Parameters:
        -----------
        token_id_batches: List[List[int]]
            A list of batches of token IDs to decode
            
        skip_special_tokens: bool, default=False
            Whether to skip special tokens in the decoded output
            
        Returns:
        --------
        List[bytes]: List of decoded byte sequences
        """
        # Pre-allocate result list for better performance
        results = [None] * len(token_id_batches)
        
        # Process all batches
        for i, token_ids in enumerate(token_id_batches):
            results[i] = self.decode(token_ids, skip_special_tokens)
        
        return results
