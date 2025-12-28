"""
Bitfield-based mask operations for memory-efficient selection filtering.

Using packed bitfields reduces memory usage by 8x compared to boolean arrays,
and improves cache locality for better performance on large datasets.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import numba as nb
    _HAVE_NUMBA = True
except ImportError:
    nb = None
    _HAVE_NUMBA = False


class BitfieldMask:
    """
    Memory-efficient boolean mask using packed bits.
    
    Instead of 1 byte per boolean, uses 1 bit per boolean (8x compression).
    Optimized for large datasets with millions of points.
    
    Performance benefits:
    - 8x less memory usage
    - Better cache locality
    - Faster logical operations on modern CPUs with SIMD
    
    Examples
    --------
    >>> mask = BitfieldMask(1000000)  # 1M points
    >>> mask.set_bit(500000, True)
    >>> mask.get_bit(500000)
    True
    >>> bool_array = mask.to_boolean()  # Convert back to numpy bool array
    """
    
    __slots__ = ('_data', '_size', '_dtype')
    
    def __init__(self, size: int, dtype: np.dtype = np.uint64):
        """
        Create a bitfield mask for `size` points.
        
        Parameters
        ----------
        size : int
            Number of data points to represent
        dtype : np.dtype, optional
            Underlying integer type (uint8, uint32, or uint64).
            uint64 (default) gives best performance on 64-bit systems.
        """
        self._size = int(size)
        self._dtype = np.dtype(dtype)
        bits_per_word = self._dtype.itemsize * 8
        n_words = (size + bits_per_word - 1) // bits_per_word
        self._data = np.zeros(n_words, dtype=self._dtype)
    
    @property
    def size(self) -> int:
        """Number of bits/points this mask represents."""
        return self._size
    
    @property
    def nbytes(self) -> int:
        """Memory usage in bytes."""
        return self._data.nbytes
    
    def set_bit(self, index: int, value: bool = True) -> None:
        """Set bit at index to value."""
        bits_per_word = self._dtype.itemsize * 8
        word_idx = index // bits_per_word
        bit_idx = index % bits_per_word
        if value:
            self._data[word_idx] |= (1 << bit_idx)
        else:
            self._data[word_idx] &= ~(1 << bit_idx)
    
    def get_bit(self, index: int) -> bool:
        """Get bit at index."""
        bits_per_word = self._dtype.itemsize * 8
        word_idx = index // bits_per_word
        bit_idx = index % bits_per_word
        return bool((self._data[word_idx] >> bit_idx) & 1)
    
    def to_boolean(self) -> np.ndarray:
        """Convert to standard numpy boolean array."""
        if _HAVE_NUMBA:
            return _bitfield_to_boolean_numba(self._data, self._size, self._dtype.itemsize * 8)
        return _bitfield_to_boolean_numpy(self._data, self._size, self._dtype.itemsize * 8)
    
    @classmethod
    def from_boolean(cls, bool_array: np.ndarray, dtype: np.dtype = np.uint64) -> "BitfieldMask":
        """Create BitfieldMask from boolean numpy array."""
        mask = cls(len(bool_array), dtype=dtype)
        if _HAVE_NUMBA:
            _boolean_to_bitfield_numba(bool_array, mask._data, dtype.itemsize * 8)
        else:
            _boolean_to_bitfield_numpy(bool_array, mask._data, dtype.itemsize * 8)
        return mask
    
    def __or__(self, other: "BitfieldMask") -> "BitfieldMask":
        """Bitwise OR operation."""
        if self._size != other._size:
            raise ValueError("Mask sizes must match")
        result = BitfieldMask(self._size, dtype=self._dtype)
        result._data = self._data | other._data
        return result
    
    def __and__(self, other: "BitfieldMask") -> "BitfieldMask":
        """Bitwise AND operation."""
        if self._size != other._size:
            raise ValueError("Mask sizes must match")
        result = BitfieldMask(self._size, dtype=self._dtype)
        result._data = self._data & other._data
        return result
    
    def __invert__(self) -> "BitfieldMask":
        """Bitwise NOT operation."""
        result = BitfieldMask(self._size, dtype=self._dtype)
        result._data = ~self._data
        # Clear unused bits in last word
        bits_per_word = self._dtype.itemsize * 8
        remainder = self._size % bits_per_word
        if remainder > 0:
            mask = (1 << remainder) - 1
            result._data[-1] &= mask
        return result
    
    def count_set(self) -> int:
        """Count number of set bits (True values)."""
        if _HAVE_NUMBA:
            return _count_set_bits_numba(self._data)
        # Use numpy's binary representation counting
        return int(np.sum([bin(x).count('1') for x in self._data]))
    
    def copy(self) -> "BitfieldMask":
        """Create a copy of this mask."""
        result = BitfieldMask(self._size, dtype=self._dtype)
        result._data = self._data.copy()
        return result


# ---- Numba-accelerated implementations ----

if _HAVE_NUMBA:
    @nb.njit(cache=True, parallel=True)
    def _bitfield_to_boolean_numba(data: np.ndarray, size: int, bits_per_word: int) -> np.ndarray:
        """Convert bitfield to boolean array using Numba."""
        result = np.zeros(size, dtype=np.bool_)
        n_words = data.shape[0]
        for word_idx in nb.prange(n_words):
            word = data[word_idx]
            start_bit = word_idx * bits_per_word
            end_bit = min(start_bit + bits_per_word, size)
            for bit_idx in range(end_bit - start_bit):
                result[start_bit + bit_idx] = bool((word >> bit_idx) & 1)
        return result
    
    @nb.njit(cache=True, parallel=True)
    def _boolean_to_bitfield_numba(bool_array: np.ndarray, data: np.ndarray, bits_per_word: int) -> None:
        """Convert boolean array to bitfield using Numba (in-place)."""
        size = bool_array.shape[0]
        n_words = data.shape[0]
        for word_idx in nb.prange(n_words):
            word = 0
            start_bit = word_idx * bits_per_word
            end_bit = min(start_bit + bits_per_word, size)
            for bit_idx in range(end_bit - start_bit):
                if bool_array[start_bit + bit_idx]:
                    word |= (1 << bit_idx)
            data[word_idx] = word
    
    @nb.njit(cache=True, parallel=True)
    def _count_set_bits_numba(data: np.ndarray) -> int:
        """Count set bits using Numba with population count."""
        total = 0
        for i in nb.prange(data.shape[0]):
            word = data[i]
            count = 0
            while word:
                count += 1
                word &= word - 1  # Clear lowest set bit
            total += count
        return total
else:
    _bitfield_to_boolean_numba = None
    _boolean_to_bitfield_numba = None
    _count_set_bits_numba = None


# ---- Pure numpy fallbacks ----

def _bitfield_to_boolean_numpy(data: np.ndarray, size: int, bits_per_word: int) -> np.ndarray:
    """Convert bitfield to boolean array using pure numpy (slower fallback)."""
    result = np.zeros(size, dtype=bool)
    for i in range(size):
        word_idx = i // bits_per_word
        bit_idx = i % bits_per_word
        result[i] = bool((data[word_idx] >> bit_idx) & 1)
    return result


def _boolean_to_bitfield_numpy(bool_array: np.ndarray, data: np.ndarray, bits_per_word: int) -> None:
    """Convert boolean array to bitfield using pure numpy (slower fallback)."""
    size = len(bool_array)
    n_words = len(data)
    for word_idx in range(n_words):
        word = 0
        start_bit = word_idx * bits_per_word
        end_bit = min(start_bit + bits_per_word, size)
        for bit_idx in range(end_bit - start_bit):
            if bool_array[start_bit + bit_idx]:
                word |= (1 << bit_idx)
        data[word_idx] = word


# ---- High-level convenience functions ----

def create_mask_from_condition(
    data: np.ndarray,
    condition: Optional[np.ndarray] = None,
    use_bitfield: bool = True
) -> np.ndarray | BitfieldMask:
    """
    Create a mask from a boolean condition array.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to create mask for
    condition : np.ndarray, optional
        Boolean array of same length. If None, creates empty mask.
    use_bitfield : bool
        If True, return BitfieldMask (8x memory savings).
        If False, return standard numpy bool array.
    
    Returns
    -------
    mask : BitfieldMask or np.ndarray
        Mask representing the condition
    """
    if condition is None:
        condition = np.zeros(len(data), dtype=bool)
    
    if use_bitfield:
        return BitfieldMask.from_boolean(condition)
    return condition.astype(bool)


def combine_masks(
    masks: list[np.ndarray | BitfieldMask],
    operation: str = 'or'
) -> np.ndarray | BitfieldMask:
    """
    Combine multiple masks using logical operations.
    
    Parameters
    ----------
    masks : list
        List of masks (either BitfieldMask or numpy bool arrays)
    operation : str
        'or', 'and', or 'xor'
    
    Returns
    -------
    combined : BitfieldMask or np.ndarray
        Combined mask (type matches input)
    """
    if not masks:
        raise ValueError("Need at least one mask to combine")
    
    # Check if we're working with BitfieldMasks
    use_bitfield = isinstance(masks[0], BitfieldMask)
    
    if use_bitfield:
        result = masks[0].copy()
        for mask in masks[1:]:
            if operation == 'or':
                result = result | mask
            elif operation == 'and':
                result = result & mask
            else:
                raise ValueError(f"Unsupported operation for BitfieldMask: {operation}")
        return result
    else:
        result = masks[0].copy()
        for mask in masks[1:]:
            if operation == 'or':
                result |= mask
            elif operation == 'and':
                result &= mask
            elif operation == 'xor':
                result ^= mask
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        return result
