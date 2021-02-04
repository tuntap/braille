#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools as it
import math
import more_itertools as mit
import numpy as np
import unicodedata as ud
import sympy as sp


# STRING = '0123456789ab'
# STRING = range(2 * 6)
# STRING = '0123456789abcdefgh'
# STRING = range(3 * 6)
# START = np.array(list(STRING))


# row-wise intra-glyph

# 01 67 cd
# 23 89 ef
# 45 ab gh

# 1cr 1cc 1br 6bc 3gr 2gc

def row_wise_intra_glyph(array):
    return np.reshape(array, (-1, 3, 2))


def d_row_wise_intra_glyph(array):
    return np.ravel(array)


# column-wise intra-glyph

# 03 69 cf
# 14 7a dg
# 25 8b eh


def col_wise_intra_glyph(array):
    array = np.reshape(array, (-1, 2, 3))
    return np.transpose(array, (0, 2, 1))


def d_col_wise_intra_glyph(array):
    array = np.transpose(array, (0, 2, 1))
    return np.ravel(array)


# row-wise inter-glyph

# 01 23 45
# 67 89 ab
# cd ef gh

def row_wise_inter_glyph(array):
    array = np.reshape(array, (3, -1, 2))
    return np.transpose(array, (1, 0, 2))


def d_row_wise_inter_glyph(array):
    array = np.transpose(array, (1, 0, 2))
    return np.ravel(array)


# col-wise inter-glyph

# 09 3c 6f
# 1a 4d 7g
# 2b 5e 8h

def col_wise_inter_glyph(array):
    array = np.reshape(array, (2, -1, 3))
    return np.transpose(array, (1, 2, 0))


def d_col_wise_inter_glyph(array):
    array = np.transpose(array, (2, 0, 1))
    return np.ravel(array)


def catz_order(array):
    array = np.reshape(array, (1, 2, 3, -1, 2))
    return np.transpose(array, (0, 1, 3, 2, 4))


# (<row>x<column><name><order>){3}
def parse_spec(spec):
    position = {'c': 0, 'b': 1, 'g': 2}
    sizes = [None] * 6
    axes = [None] * 6

    for i, part in enumerate(spec.split(' ')):
        size, name, order = part[:-2], part[-2], part[-1]

        j = position[name]
        size = tuple(map(int, size.split('x')))
        size = size if order == 'r' else reversed(size)
        paxes = (2 * i, 2 * i + 1)
        paxes = paxes if order == 'r' else reversed(paxes)

        sizes[2 * i:2 * i + 2] = size
        axes[2 * j:2 * j + 2] = paxes

    return sizes, axes


# (<size><name>){6}
def parse_spec2(spec):
    position = {'cr': 0, 'cc': 1, 'br': 2, 'bc': 3, 'gr': 4, 'gc': 5}
    sizes = [None] * 6
    axes = [None] * 6

    for i, part in enumerate(spec.split(' ')):
        size, name = part[:-2], part[-2:]

        j = position[name]
        size = int(size)
        axis = i

        sizes[i] = size
        axes[j] = axis

    return sizes, axes


def unpack_braille(array, conf):
    sizes, axes = conf
    return np.transpose(np.reshape(array, sizes), axes)


def rev_conf(conf):
    sizes, axes = conf

    sizes2 = [None] * len(sizes)
    axes2 = [None] * len(axes)

    for i, j in enumerate(axes):
        sizes2[i] = sizes[j]
        axes2[i] = axes[j]

    return sizes2, axes2


def pack_braille(array, conf):
    _, axes = rev_conf(conf)
    return np.ravel(np.transpose(array, axes))


def test_unpack_pack(array, conf):
    unpacked = unpack_braille(array, conf)
    packed = pack_braille(unpacked, conf)

    return unpacked, packed


def test_num(conf, n=36):
    return test_unpack_pack(np.fromiter(range(n), int), conf)


import string

def test_str(conf, s=None):
    s = s or string.digits + string.ascii_lowercase
    return test_unpack_pack(np.fromiter(s, 'U1'), conf)


BRAILLE = {
    'a': '100000',
    'b': '101000',
    'c': '110000',
    'd': '110100',
    'e': '100100',
    'f': '111000',
    'g': '111100',
    'h': '101100',
    'i': '011000',
    'j': '011100',
    'k': '100010',
    'l': '101010',
    'm': '110010',
    'n': '110110',
    'o': '100110',
    'p': '111010',
    'q': '111110',
    'r': '101110',
    's': '011010',
    't': '011110',
    'u': '100011',
    'v': '101011',
    'w': '011101',
    'x': '110011',
    'y': '110111',
    'z': '100111',
}
BRAILLE_REV = {v: k for k, v in BRAILLE.items()}


def make_braille_chars():
    chars = {}

    for letter, pattern in BRAILLE.items():
        nums = np.reshape(np.array(list(pattern)), (3, 2))
        nums = np.transpose(nums)
        nums = np.reshape(nums, (6,))
        nums, = np.where(nums == '1')
        nums = list(map(lambda n: n + 1, nums))
        char = ud.lookup('Braille pattern dots-{}'.format(''.join(map(str, nums))))

        chars[letter] = char

    return chars


BRAILLE_CHARS = make_braille_chars()
BRAILLE_CHARS_REV = {v: k for k, v in BRAILLE_CHARS.items()}


def encode_row_wise_intra_glyph(string):
    string = it.chain.from_iterable(map(lambda l: BRAILLE[l], string))
    array = np.reshape(np.fromiter(string, 'U1'), (-1, 3, 2))
    return ''.join(d_row_wise_intra_glyph(array))


def decode_row_wise_intra_glyph(string):
    # TODO: use fromiter
    array = np.reshape(row_wise_intra_glyph(np.array(list(string))), (-1, 6))
    return ''.join(map(lambda g: BRAILLE_REV[''.join(g)], array))


def encode_col_wise_intra_glyph(string):
    string = it.chain.from_iterable(map(lambda l: BRAILLE[l], string))
    array = np.reshape(np.fromiter(string, 'U1'), (-1, 3, 2))
    return ''.join(d_col_wise_intra_glyph(array))


def decode_col_wise_intra_glyph(string):
    array = np.reshape(col_wise_intra_glyph(np.array(list(string))), (-1, 6))
    return ''.join(map(lambda g: BRAILLE_REV[''.join(g)], array))


def encode_row_wise_inter_glyph(string):
    string = it.chain.from_iterable(map(lambda l: BRAILLE[l], string))
    array = np.reshape(np.fromiter(string, 'U1'), (-1, 3, 2))
    return ''.join(d_row_wise_inter_glyph(array))


def decode_row_wise_inter_glyph(string):
    array = np.reshape(row_wise_inter_glyph(np.array(list(string))), (-1, 6))
    return ''.join(map(lambda g: BRAILLE_REV[''.join(g)], array))


def encode_col_wise_inter_glyph(string):
    string = it.chain.from_iterable(map(lambda l: BRAILLE[l], string))
    array = np.reshape(np.fromiter(string, 'U1'), (-1, 3, 2))
    return ''.join(d_col_wise_inter_glyph(array))


def decode_col_wise_inter_glyph(string):
    array = np.reshape(col_wise_inter_glyph(np.array(list(string))), (-1, 6))
    return ''.join(map(lambda g: BRAILLE_REV[''.join(g)], array))


def block_params(n):
    factors = list(it.chain.from_iterable(
        [p] * m for p, m in sp.factorint(n).items()))

    partitions = it.takewhile(lambda p: len(p) <= 4, mit.partitions(factors))
    partitions = set(tuple(sorted(map(math.prod, partition)))
                     for partition in partitions)
    partitions = (tuple(partition) + (1,) * (4 - len(partition))
                  for partition in partitions)
    partitions = set(it.chain.from_iterable(map(it.permutations, partitions)))

    yield from partitions


# (packed) bitstring '0101010101010'
# (packed) bitvector ['0' '1' '0' ...]
# (unpacked) bitgrid [[[['1' '0' ...]]]]
# (unpacked) chargrid [[['h' ...]]]
# (packed) string


def try_decode(array, conf):
    array = unpack_braille(array, conf)
    decoded = np.zeros(array.shape[:-2], 'U1')

    try:
        for i in np.ndindex(decoded.shape):
            decoded[i] = BRAILLE_REV[''.join(np.ravel(array[i]))]
    except KeyError:
        return None

    return decoded


def brute(string):
    n = len(string)
    assert n % 6 == 0

    array = np.fromiter(string, 'U1')

    for cr, cc, br, bc in block_params(n // 6):
        sizes = (cr, cc, br, bc, 3, 2)
        axes = tuple(range(6))

        for sizes, axes in zip(it.permutations(sizes), it.permutations(axes)):
            res = try_decode(array, (sizes, axes))

            if res is not None:
                yield ''.join(np.ravel(res))


def something(string):
    return set(brute(string))


# print(decode_row_wise_intra_glyph(encode_row_wise_intra_glyph('hello')))
# print(decode_col_wise_intra_glyph(encode_col_wise_intra_glyph('hello')))
# print(decode_row_wise_inter_glyph(encode_row_wise_inter_glyph('hello')))
# print(decode_col_wise_inter_glyph(encode_col_wise_inter_glyph('hello')))

PACKED = np.fromiter('101010110110000010101001100111101010100101011011001010', 'U1')

# 10 10 10 10 10 01 10 01 01
# 11 01 10 10 01 11 01 10 11
# 00 00 10 10 10 10 00 10 10

# 01 23 45 67 89 ab
# cd ef gh ij kl mn
# op qr st uv wx yz

# 01 23 45 ij kl mn
# 67 89 ab op qr st
# cd ef gh uv wx yz

# reversed(gcol, bcol, brow, grow, ccol, crow)
# crow ccol grow brow bcol gcol
# '1cr 2cc 3gr 1br 3bc 2gc'

import string

STRING = string.digits + string.ascii_lowercase

# row, glyph, col
# array = np.reshape(array, (3, -1, 2))
# glyph, row, col
# return np.transpose(array, (1, 0, 2))

RES = np.fromiter(STRING, 'U1')
# brow, bcolumn, row, glyph, column
RES = np.reshape(RES, (1, 2, 3, -1, 2))
RES = np.transpose(RES, (0, 1, 3, 2, 4))
print(RES)
