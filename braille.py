#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# NOTE: The regular scheme doesn't capture reversals or permutations of items
# within a group.

# NOTE: Duplicate solutions arise when a correct shape's sizes are further
# broken down into their factors while preserving the ordering of the axes. We
# can't throw away the broken down shape completely though because they might
# yield new solutions when coupled with different orderings.

# NOTE: It is possible for a single bitstring to decode into different
# solutions that are just permutations (anagrams) of each other.

# NOTE: Is it possible for a single bitstring to decode into different
# solutions that contain different letters? So far no examples have been found.

# NOTE: A Braille array is an ndarray such that: (1) its dtype is U1, (2) it
# has 3 or more dimensions, (3) the last 2 dimensions are the rows and columns
# (in that order) of a Braille glyph, and (4) it contains only the characters
# '0' and '1'. This is the "internal representation" of a sequence of Braille
# glyphs.

# NOTE: A char array is an ndarray of dtype U1.

# NOTE: A string array is a 1-dimensional char array.

# NOTE: A bitstring array is string array containing only the characters '0'
# and '1'.

# NOTE: A bitstring is a string containing only the characters '0' and '1'.


import collections
import itertools as it
import math
import more_itertools as mit
import numpy as np
import unicodedata as ud
import sympy as sp
import yattag


BRAILLE_SHAPE = (3, 2)
BRAILLE_NDIM = len(BRAILLE_SHAPE)
BRAILLE_SIZE = math.prod(BRAILLE_SHAPE)


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
BRAILLE_REV['??????'] = '?'


def make_braille_chars():
    chars = {}

    for letter, pattern in BRAILLE.items():
        nums = np.reshape(np.array(list(pattern)), (3, 2))
        nums = np.transpose(nums)
        nums = np.reshape(nums, (6,))
        nums, = np.where(nums == '1')
        nums = list(map(lambda n: n + 1, nums))
        char = ud.lookup('Braille pattern dots-{}'.format(
            ''.join(map(str, nums))))

        chars[letter] = char

    return chars

if BRAILLE_SHAPE == (3, 2):
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
### Conversions


def unpack(array, conf):
    shape, axes = conf
    return np.transpose(np.reshape(array, apply_perm(shape, axes)),
                        reverse_perm(axes))


def pack(array, conf):
    shape, axes = conf
    return np.ravel(np.transpose(np.reshape(array, shape), axes))


# Convert a Braille array to a char array of similar shape.
def decode_braille(array, braille_rev=BRAILLE_REV):
    decoded = np.zeros(array.shape[:-BRAILLE_NDIM], 'U1')

    try:
        for i in np.ndindex(decoded.shape):
            decoded[i] = braille_rev[''.join(np.ravel(array[i]))]
    except KeyError as e:
        return None

    return decoded


# Convert a char array to a Braille array of similar shape.
def encode_braille(array, braille=BRAILLE):
    encoded = np.zeros((*array.shape, *BRAILLE_SHAPE), 'U1')

    try:
        for i in np.ndindex(array.shape):
            encoded[i, ...] = np.reshape(
                np.fromiter(braille[array[i]], 'U1'), BRAILLE_SHAPE)
    except KeyError:
        return None

    return encoded


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


def all_confs(n):
    assert n % BRAILLE_SIZE == 0

    for partition in multiplicative_partitions(n // BRAILLE_SIZE):
        shape = (*partition, *BRAILLE_SHAPE)
        axes = tuple(range(len(partition) + BRAILLE_NDIM))

        for axes in it.permutations(axes):
            yield shape, axes


def brute_array(array):
    assert array.ndim == 1
    yield from ((decode_braille(unpack(array, conf)), conf)
                for conf in all_confs(len(array)))


def brute_array_valid(array):
    yield from ((decoded, conf) for decoded, conf in brute_array(array)
                if decoded is not None)


def brute_array_padding(array, padding):
    assert array.ndim == 1
    assert padding > 0

    idx = np.fromiter(range(len(array) + BRAILLE_SIZE * padding), int)

    for conf in all_confs(len(idx)):
        insert = sorted(unpack(idx, conf).flat[-(BRAILLE_SIZE * padding):])

        padded = array
        for i, j in enumerate(insert):
            padded = np.insert(padded, j, '?')

        yield decode_braille(unpack(padded, conf)), padded, conf


def brute_array_padding_valid(array, padding):
    yield from ((decoded, padded, conf) for decoded, padded, conf
                in brute_array_padding(array, padding)
                if decoded is not None)


######################################################################
###
### Utility


def str_array(iterable):
    return np.fromiter(iterable, 'U1')


def num_array(iterable):
    return np.fromiter(iterable, int)


def decode(string, conf=((-1, *BRAILLE_SHAPE), (0, 1, 2))):
    assert len(string) % BRAILLE_SIZE == 0
    return ''.join(np.ravel(decode_braille(unpack(str_array(string), conf))))


def encode(string, conf=((-1, *BRAILLE_SHAPE), (0, 1, 2))):
    return ''.join(np.ravel(pack(encode_braille(str_array(string)), conf)))


def encode_visual(string):
    array = np.transpose(encode_braille(str_array(string)), (1, 0, 2))
    return '\n'.join(' '.join(''.join(g) for g in row) for row in array)


def brute(string, padding=None):
    assert len(string) % BRAILLE_SIZE == 0
    array = str_array(string)

    if not padding:
        return set(''.join(np.ravel(decoded))
                   for decoded, _ in brute_array_valid(array))

    paddings = (range(1, len(string) // BRAILLE_SIZE)
                if padding == -1 else [padding])

    return set(it.chain.from_iterable(
        (''.join(np.ravel(decoded)).replace('?', '') for decoded, _, _
         in brute_array_padding_valid(array, padding))
        for padding in paddings))


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


def format_braille_html(array, hfunc=None):
    assert array.ndim > 2
    doc, tag, text, line = yattag.Doc().ttl()

    def rec(array, depth):
        class_ = 'depth{}'.format(depth)

        if array.ndim == 2:
            nr, nc = array.shape

            with tag('table', klass=class_):
                if hfunc:
                    with tag('tr'):
                        line('th', hfunc(array), klass=class_, colspan=nc)
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


def brute_visualize(string):
    assert len(string) % BRAILLE_SIZE == 0
    array = str_array(string)
    idx = num_array(range(len(string)))

    doc, tag, text, line = yattag.Doc().ttl()
    doc.asis('<!doctype html>')

    res = list(brute_array_valid(str_array(string)))
    count = collections.defaultdict(int)

    for decoded, _ in res:
        count[''.join(np.ravel(decoded))] += 1

    with tag('html'):
        with tag('head'):
            doc.asis(STYLE)

        with tag('body'):
            for i, (decoded, conf) in enumerate(res):
                shape, axes = conf

                line('h1', 'Solution {} ({}, {}, {} ({}))'.format(
                    i + 1, *conf, ''.join(np.ravel(decoded)),
                    count[''.join(np.ravel(decoded))]))

                line('h2', 'Braille bitstring')
                doc.asis(format_array_html(array))

                line('h2', 'Shape')
                doc.asis(format_array_html(np.reshape(array, shape)))

                line('h2', 'Unpacked ordering')
                doc.asis(format_braille_html(unpack(idx, conf)))

                line('h2', 'Unpacked bitstring')
                doc.asis(format_braille_html(
                    unpack(array, conf),
                    hfunc=lambda a: BRAILLE_REV[''.join(np.ravel(a))]))

    with open('braille.html', 'w') as f:
        f.write(yattag.indent(doc.getvalue()))


def brute_visualize_padding(string, padding):
    assert len(string) % BRAILLE_SIZE == 0
    array = str_array(string)

    doc, tag, text, line = yattag.Doc().ttl()
    doc.asis('<!doctype html>')

    paddings = (range(1, len(string) // BRAILLE_SIZE)
                if padding == -1 else [padding])
    res = list(it.chain.from_iterable(
        brute_array_padding_valid(array, padding) for padding in paddings))
    count = collections.defaultdict(int)

    for decoded, _, _ in res:
        count[''.join(np.ravel(decoded))] += 1

    with tag('html'):
        with tag('head'):
            doc.asis(STYLE)

        with tag('body'):
            for i, (decoded, padded, conf) in enumerate(res):
                shape, axes = conf
                idx = num_array(range(len(padded)))

                line('h1', 'Solution {} ({}, {}, {} ({}))'.format(
                    i + 1, *conf, ''.join(np.ravel(decoded)),
                    count[''.join(np.ravel(decoded))]))

                line('h2', 'Braille bitstring')
                doc.asis(format_array_html(array))

                line('h2', 'Padded bitstring')
                doc.asis(format_array_html(padded))

                line('h2', 'Shape')
                doc.asis(format_array_html(np.reshape(padded, shape)))

                line('h2', 'Unpacked ordering')
                doc.asis(format_braille_html(unpack(idx, conf)))

                line('h2', 'Unpacked bitstring')
                doc.asis(format_braille_html(
                    unpack(padded, conf),
                    hfunc=lambda a: BRAILLE_REV[''.join(np.ravel(a))]))

    with open('braille.html', 'w') as f:
        f.write(yattag.indent(doc.getvalue()))
