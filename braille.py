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


def all_block_params(n):
    factors = list(it.chain.from_iterable(
        [p] * m for p, m in sp.factorint(n).items()))

    partitions = it.takewhile(lambda p: len(p) <= 4, mit.partitions(factors))
    partitions = set(tuple(sorted(map(math.prod, partition)))
                     for partition in partitions)
    partitions = (tuple(partition) + (1,) * (4 - len(partition))
                  for partition in partitions)
    # TODO: generate permutations that take into account repetition of elements
    partitions = set(it.chain.from_iterable(map(it.permutations, partitions)))

    yield from partitions


def all_confs(string):
    n = len(string)
    assert n % 6 == 0

    for cr, cc, br, bc in all_block_params(n // 6):
        sizes = (cr, cc, br, bc, 3, 2)
        axes = tuple(range(6))

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


# array = np.fromiter(range(54), int)
# sizes, axes = (3, 3, 3, 2, 1, 1), (0, 2, 4, 5, 3, 1)
# print(format_conf((sizes, axes)))
# print(np.reshape(array, sizes))
# print(np.transpose(np.reshape(array, sizes), axes))


# array = np.fromiter(range(54), int)
# 3cr 1cc 2gc 3br 1bc 3gr
# sizes, axes = (3, 1, 2, 3, 1, 3), (0, 1, 5, 2, 3, 4)
# print(unpack_braille(array, (sizes, axes)).shape)
# print(html(unpack_braille(array, (sizes, axes)), (sizes, axes)))
# shape = (3, 1, 1, 3, 2, 3)

# print(np.reshape(array, sizes))
# print(np.transpose(np.reshape(array, sizes), reverse_perm(axes)))

# def test(string):
#     n = len(string)
#     assert n % 6 == 0

#     for cr, cc, br, bc in all_block_params(n // 6):
#         sizes = (cr, cc, br, bc, 3, 2)
#         axes = tuple(range(6))

#         for sizes, axes in zip(it.permutations(sizes), it.permutations(axes)):
#             yield sizes, axes, apply_perm(sizes, axes)


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


def format_decoded(decoded):
    cr, cc, br, bc = decoded.shape
    grid = [[None] * (cc * bc) for _ in range(cr * br)]

    for i in np.ndindex(decoded.shape):
        ci, cj, bi, bj = i
        grid[ci * br + bi][cj * bc + bj] = decoded[i]

    return '\n'.join(map('\t'.join, grid))


# # (<row>x<column><name><order>){3}
# def parse_spec(spec):
#     position = {'c': 0, 'b': 1, 'g': 2}
#     sizes = [None] * 6
#     axes = [None] * 6

#     for i, part in enumerate(spec.split(' ')):
#         size, name, order = part[:-2], part[-2], part[-1]

#         j = position[name]
#         size = tuple(map(int, size.split('x')))
#         size = size if order == 'r' else reversed(size)
#         paxes = (2 * i, 2 * i + 1)
#         paxes = paxes if order == 'r' else reversed(paxes)

#         sizes[2 * i:2 * i + 2] = size
#         axes[2 * j:2 * j + 2] = paxes

#     return sizes, axes


# (<size><name>){6}
def parse_conf_spec(spec):
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


def format_conf(conf):
    names = ['cr', 'cc', 'br', 'bc', 'gr', 'gc']
    return ' '.join('{}{}'.format(size, names[axis])
                    for size, axis in zip(*conf))


PACKED = np.fromiter('101010110110000010101001100111101010100101011011001010', 'U1')

# 101010110110000010101001100111101010100101011011001010
#
# '1cr 3cc 3gr 1br 3bc 2gc'
#
# 01 23 45 ij kl mn AB CD EF
# 67 89 ab op qr st GH IJ KL
# cd ef gh uv wx yz MN OP QR
#
# 10 10 10 10 10 01 10 01 01
# 11 01 10 10 01 11 01 10 11
# 00 00 10 10 10 10 00 10 10

# reversed(gcol, bcol, brow, grow, ccol, crow)
# crow ccol grow brow bcol gcol
# '1cr 2cc 3gr 1br 3bc 2gc'
#
# 01 23 45 ij kl mn
# 67 89 ab op qr st
# cd ef gh uv wx yz

PACKED = np.fromiter('111000010010111101111000101011001000011011001011101010001000010111101000101101011010011010', 'U1')
PACKEDN = np.fromiter(range(90), int)

def encode(string):
    return ''.join(map(lambda c: BRAILLE[c], string))

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

# ---

# ###

# 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&/()=?*_[]{}<>@+-.:,;žć

# ---

# 01 67 23 89 45 ab cd ij ef kl gh mn op uv qr wx st yz AB GH CD IJ EF
# KL MN ST OP UV QR WX YZ %& !" /( #$ )= ?* }< _[ >@ ]{ +- .: ,; žć

# ###

# 01 67 23 89 45 ab cd ij ef kl gh mn op uv qr wx st yz AB GH CD IJ EF KL
# MN ST OP UV QR WX YZ %& !" /( #$ )= [] >_ {} -+ @< *? xx xx xx

# 01 67
# 23 89
# 45 ab

# cd ij
# ef kl
# gh mn

# op uv
# qr wx
# st yz

# AB GH
# CD IJ
# EF KL

# MN ST
# OP UV
# QR WX

# YZ %&
# !" /(
# #$ )=

# [] >_
# {} -+
# @< *?

# xx
# xx
# xx

# reversed('2gc 3gr 15bc 1br 1cr 1cc')
# '1cc 1cr 1br 15bc 3gr 2gc'
#
# 01 67 cd ij op uv AB GH MN ST YZ %& [] >_ xx
# 23 89 ef kl qr wx CD IJ OP UV !" /( {} -+ xx
# 45 ab gh mn st yz EF KL QR WX #$ )= @< *? xx

PACKED2 = np.fromiter('111000010010111101111000101011001000011011001011101010001000010111101000101101011010', 'U1')

# '7br 3gr 1cr 2bc 1cc 2gc'

# 01 23
# 45 67
# 89 ab

# cd ef
# gh ij
# kl mn

# op qr
# st uv
# wx yz

# AB CD
# EF GH
# IJ KL

# MN OP
# QR ST
# UV WX

# YZ !"
# #$ %&
# /( )=

# ?* _[
# ]{ }<
# >@ +-

# for d, _ in brute('111000010010111101111000101011001000011011001011101010001000010111101000101101011010'):
#     print(format_decoded(d))
