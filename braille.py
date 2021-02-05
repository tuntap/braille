#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools as it
import math
import more_itertools as mit
import numpy as np
import unicodedata as ud
import sympy as sp


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


######################################################################
###
### Permutations


def make_perm(src, dst):
    return tuple(src.index(marker) for marker in dst)


def apply_perm(seq, perm):
    return tuple(seq[i] for i in perm)


def reverse_perm(perm):
    final = [None] * len(perm)

    for i, j in enumerate(perm):
        final[j] = i

    return tuple(final)


######################################################################
###
### Packing


def unpack_braille(array, conf):
    sizes, axes = conf
    return np.transpose(np.reshape(array, sizes), reverse_perm(axes))


def pack_braille(array, conf):
    _, axes = conf
    return np.ravel(np.transpose(array, axes))


######################################################################
###
### Encoding and decoding


# (packed) bitstring '0101010101010'
# (packed) bitvector ['0' '1' '0' ...]
# (unpacked) bitgrid [[[['1' '0' ...]]]]
# (unpacked) chargrid [[['h' ...]]]
# (packed) string


def decode(array, conf):
    array = unpack_braille(array, conf)
    decoded = np.zeros(array.shape[:-2], 'U1')

    try:
        for i in np.ndindex(decoded.shape):
            decoded[i] = BRAILLE_REV[''.join(np.ravel(array[i]))]
    except KeyError as e:
        return None

    return decoded


def decode_str(string, conf):
    return decode(np.fromiter(string, 'U1'), conf)


######################################################################
###
### Brute-forcing


def filter_seen(iterable):
    seen = set()

    for thing in iterable:
        if thing not in seen:
            seen.add(thing)
            yield thing


def multiplicative_partitions(n, k=None):
    factors = it.chain.from_iterable(
        [p] * m for p, m in sp.factorint(n).items())

    # TODO: I'd like to avoid the filter_seen step by generating distinct
    # partitions if possible.
    #
    # Otherwise, perhaps I should implement the partitioning from scratch as in
    # https://stackoverflow.com/questions/8558292/how-to-find-multiplicative-partitions-of-any-integer.

    ps = mit.partitions(factors)
    ps = it.takewhile(lambda p: len(p) <= k, ps) if k else ps
    ps = map(lambda p: tuple(sorted(map(math.prod, p))), ps)
    ps = filter_seen(ps)
    ps = it.chain.from_iterable(map(mit.distinct_permutations, ps))

    yield from ps


def all_confs(string):
    n = len(string)
    assert n % 6 == 0

    for partition in multiplicative_partitions(n // 6):
        sizes = (*partition, 3, 2)
        axes = tuple(range(len(partition) + 2))

        for sizes, axes in zip(it.permutations(sizes), it.permutations(axes)):
            yield sizes, axes


def brute(string):
    array = np.fromiter(string, 'U1')

    for conf in all_confs(string):
        res = decode(array, conf)

        if res is not None:
            yield res, conf


def brute1(string):
    return set(''.join(np.ravel(decoded)) for decoded, _ in brute(string))


######################################################################
###
### Utility


# TODO: test whether there exists a > 3-tier structure that can't be captured
# by a <= 3-tier structure

# 1cr 1cc 3br 3bc 3gr 2gc
# 1br 1bc 3cr 3cc 3gr 2gc


def html(array, conf):
    def rec(array, depth):
        if type(array) is not np.ndarray:
            return str(array)

        row = []

        for i in range(len(array)):
            elem = rec(array[i], depth + 1)
            row.append('<td class="depth{}">{}</td>'.format(depth, elem))

        tr = '<tr class="depth{}">{}</tr>'.format(depth, '\n'.join(row))
        return '<table class="depth{}">{}</table>'.format(depth, tr)

    return rec(array, 0)


def braille_html(array, conf):
    assert array.ndim >= 3

    def rec(array, depth):
        if array.ndim == 2:
            nr, nc = array.shape
            table = []

            for r in range(nr):
                row = []

                for c in range(nc):
                    row.append('<td class="depth{}">{}</td>'.format(
                        depth, array[r][c]))

                table.append('<tr class="depth{}">{}</tr>'.format(
                    depth, '\n'.join(row)))

            return '<table class="depth{}">{}</table>'.format(
                depth, '\n'.join(table))

        row = []

        for i in range(len(array)):
            elem = rec(array[i], depth + 1)
            row.append('<td class="depth{}">{}</td>'.format(depth, elem))

        tr = '<tr class="depth{}">{}</tr>'.format(depth, '\n'.join(row))
        return '<table class="depth{}">{}</table>'.format(depth, tr)

    return rec(array, 0)


def doit(string):
    packed = np.fromiter(string, 'U1')
    array = np.fromiter(range(len(string)), int)

    with open('braille.html', 'w') as f:
        f.write('''<!doctype html>
<html>
<head>
<style>
    table {
        border: 2px solid;
        padding: 10px;
        border-radius: 0px;
        text-align: center;
    }

    table.depth0 {
        border-color: #1cc970;
    }

    table.depth1 {
        border-color: #e3db3d;
    }

    table.depth2 {
        border-color: #1c87c9;
    }
</style>
</head>
<body>
''')

        f.write('''<h1>String</h1>
{}</br>
{}'''.format(packed, array))

        for i, (_, conf) in enumerate(brute(string)):
            array = unpack_braille(array, conf)
            print(i, array.shape, conf[1])
            sizes, axes = conf
            f.write('<h1>Table {} ({}, {})</h1>'.format(i + 1, sizes, axes))
            axes = tuple(a for a in axes if a not in (4, 5)) + (4, 5)
            f.write(braille_html(array, conf))
            f.write('\n')

        f.write('''</body>
</html>
''')


PACKED1 = np.fromiter('101010110110000010101001100111101010100101011011001010', 'U1')
PACKED2 = np.fromiter('111000010010111101111000101011001000011011001011101010001000010111101000101101011010011010', 'U1')
PACKED3 = np.fromiter('111000010010111101111000101011001000011011001011101010001000010111101000101101011010', 'U1')

# congratulations

# ---

# 110000
# 100110
# 110110
# 111100
# 101110
# 100000
# 011110
# 100011
# 101010
# 100000
# 011110
# 011000
# 100110
# 110110
# 011010

# ---

# 110000 100110
# 110110 111100
# 101110 100000
# 011110 100011
# 101010 100000
# 011110 011000
# 100110 110110
# 011010

# ---

# 11 10
# 00 01
# 00 10

# 11 11
# 01 11
# 10 00

# 10 10
# 11 00
# 10 00

# 01 10
# 11 00
# 10 11

# 10 10
# 10 00
# 10 00

# 01 01
# 11 10
# 10 00

# 10 11
# 01 01
# 10 10

# 01 xx
# 10 xx
# 10 xx

# ---

# 11 10 00 01 00 10 11 11 01 11 10 00 10 10 11 00 10 00 01 10 11 00 10 11
# 10 10 10 00 10 00 01 01 11 10 10 00 10 11 01 01 10 10 01 xx 10 xx 10 xx
