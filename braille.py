#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools as it
import math
import more_itertools as mit
import numpy as np
import unicodedata as ud
import sympy as sp
import yattag


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
    shape, axes = conf
    return np.transpose(np.reshape(array, apply_perm(shape, axes)),
                        reverse_perm(axes))


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

    # TODO: Try to avoid the filter_seen step by generating distinct partitions
    # directly if possible.
    #
    # Otherwise, perhaps multiplicative partitioning should be implemented from
    # scratch
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
        shape = (*partition, 3, 2)
        axes = tuple(range(len(partition) + 2))

        for axes in it.permutations(axes):
            yield shape, axes


def brute(string):
    array = np.fromiter(string, 'U1')

    for conf in all_confs(string):
        res = decode(array, conf)

        if res is not None:
            yield res, conf


######################################################################
###
### Utility


def brute1(string):
    return set(''.join(np.ravel(decoded)) for decoded, _ in brute(string))


def format_array_html(array):
    doc, tag, text = yattag.Doc().tagtext()

    def rec(array, depth):
        class_ = 'depth{}'.format(depth)

        if type(array) is not np.ndarray:
            with tag('table', klass=class_):
                with tag('tr', klass=class_):
                    with tag('td', klass=class_):
                        text(str(array))
        else:
            with tag('table', klass=class_):
                with tag('tr', klass=class_):
                    for i in range(len(array)):
                        with tag('td', klass=class_):
                            rec(array[i], depth + 1)

        return doc

    return rec(array, 0).getvalue()


def format_braille_html(array):
    assert array.ndim >= 3
    doc, tag, text = yattag.Doc().tagtext()

    def rec(array, depth):
        class_ = 'depth{}'.format(depth)

        if array.ndim == 2:
            nr, nc = array.shape

            with tag('table', klass=class_):
                for r in range(nr):
                    with tag('tr', klass=class_):
                        for c in range(nc):
                            with tag('td', klass=class_):
                                text(str(array[r][c]))
        else:
            with tag('table', klass=class_):
                with tag('tr', klass=class_):
                    for i in range(len(array)):
                        with tag('td', klass=class_):
                            rec(array[i], depth + 1)

        return doc

    return rec(array, 0).getvalue()


STYLE = '''
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

  table.depth3 {
    border-color: #ff0000;
  }

  table.depth4 {
    border-color: #000000;
  }
</style>
'''


def brute2(string):
    array = np.fromiter(string, 'U1')
    idx = np.fromiter(range(len(string)), int)

    doc, tag, text = yattag.Doc().tagtext()
    doc.asis('<!doctype html>')

    with tag('html'):
        with tag('head'):
            doc.asis(STYLE)

        with tag('body'):
            for i, (res, conf) in enumerate(brute(string)):
                shape, _ = conf

                with tag('h1'):
                    text('Solution {} {}'.format(i + 1, conf))

                doc.asis(format_array_html(array))
                doc.stag('br')

                doc.asis(format_array_html(idx))
                doc.stag('br')

                doc.asis(format_array_html(np.reshape(array, shape)))
                doc.stag('br')

                doc.asis(format_array_html(np.reshape(idx, shape)))
                doc.stag('br')

                doc.asis(format_braille_html(unpack_braille(idx, conf)))
                doc.stag('br')

                doc.asis(format_braille_html(unpack_braille(array, conf)))
                doc.stag('br')

                doc.asis(format_array_html(res))
                doc.stag('br')

    with open('braille.html', 'w') as f:
        f.write(yattag.indent(doc.getvalue()))


def str_array(iterable):
    return np.fromiter(iterable, 'U1')


def num_array(iterable):
    return np.fromiter(iterable, int)


HELLOTEST = '101010110110000010101001100111101010100101011011001010'
CONGRATULATIONS = '111000010010111101111000101011001000011011001011101010001000010111101000101101011010011010'
CONGRATULATION = '111000010010111101111000101011001000011011001011101010001000010111101000101101011010'
