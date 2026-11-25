#!/usr/bin/env python3
"""Trace every number in `main.tex` and `appendix.tex` back to the released results.

The paper carries its tables inline, so a number in the text can drift from the
measurement it came from without anything failing. This script reads `main.tex`,
pulls out every numeric token, and looks for each one in the released results:
the per-item CSVs in `data/`, the analysis tables in `results/tables/`, and the
printed output of every command in `results/printed_outputs/`. It reports which
numbers it could not find, so each can be checked by hand.

    python3 check_text_numbers.py
    python3 check_text_numbers.py --tex ../../main.tex --verbose
    python3 check_text_numbers.py --out ../results/printed_outputs/check_text_numbers.txt

It is a tracing aid, not a test: a number may be absent from the haystack
because it is a page measurement, a citation year, a rounding of a figure that
is stored at full precision, or a quantity stated once in prose. The point is
that the unmatched list is short enough to read, and that nothing in it should
be an empirical claim nobody can locate.
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# numeric tokens that are typesetting, not measurement
SKIP_CONTEXT = re.compile(
    r'\\(fontsize|selectfont|setlength|tabcolsep|arraystretch|columnwidth|textwidth|'
    r'resizebox|hspace|vspace|cmidrule|multicolumn|multirow|documentclass|usepackage|'
    r'setcounter|renewcommand|definecolor|includegraphics|graphicspath|label|ref|cite|'
    r'citep|citet|url|begin\{tabular|topnumber|bottomnumber|totalnumber|topfraction|'
    r'bottomfraction|textfraction|floatpagefraction|leftmargin|itemsep|topsep|parsep)')

NUM = re.compile(r'(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?)(?![\w])')


def strip_latex_noise(line):
    """drop the parts of a line whose numbers are layout, not measurement"""
    if SKIP_CONTEXT.search(line):
        # keep the prose outside the command's braces where possible
        line = SKIP_CONTEXT.sub(' ', line)
        line = re.sub(r'\{[^{}]*\}', ' ', line)
    return line


def tex_numbers(paths):
    """every numeric token of the paper, with the file and line it sits on"""
    seen = {}
    for path in ([paths] if isinstance(paths, str) else paths):
      name = os.path.basename(path)
      for n, raw in enumerate(open(path, encoding='utf8'), 1):
        if raw.lstrip().startswith('%'):
            continue
        line = strip_latex_noise(raw)
        for m in NUM.finditer(line):
            tok = m.group(1).replace('{,}', '')
            if tok.rstrip('0').rstrip('.') == '':
                continue
            seen.setdefault(tok, ('%s:%d' % (name, n), raw.strip()))
    return seen


def haystack(data, tables, printed):
    """every number the released results contain, as strings"""
    blobs = []
    for d, pat in ((data, '*.csv'), (tables, '*.csv'), (printed, '*.txt')):
        for f in sorted(glob.glob(os.path.join(d, pat))):
            with open(f, encoding='utf8', errors='replace') as fh:
                blobs.append(fh.read())
    text = '\n'.join(blobs)
    full = set(re.findall(r'-?\d+(?:\.\d+)?', text))
    # a number in the paper is usually a rounding of a stored one, so index
    # every stored value by its 0-, 1-, 2- and 3-decimal rounding as well
    rounded = set()
    for v in full:
        try:
            x = float(v)
        except ValueError:
            continue
        for k in (0, 1, 2, 3):
            s = f'{abs(x):.{k}f}'
            rounded.add(s)
            rounded.add(s.rstrip('0').rstrip('.') if '.' in s else s)
    return full | rounded, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', nargs='+',
                    default=[os.path.join(HERE, '..', '..', 'main.tex'),
                             os.path.join(HERE, '..', '..', 'appendix.tex')])
    ap.add_argument('--data', default=os.path.join(HERE, '..', 'data'))
    ap.add_argument('--tables', default=os.path.join(HERE, '..', 'results', 'tables'))
    ap.add_argument('--printed', default=os.path.join(HERE, '..', 'results', 'printed_outputs'))
    ap.add_argument('--verbose', action='store_true', help='list the line of every miss')
    ap.add_argument('--out')
    a = ap.parse_args()

    out = []

    def say(*t):
        s = ' '.join(str(x) for x in t)
        out.append(s)
        print(s)

    nums = tex_numbers(a.tex)
    hay, text = haystack(a.data, a.tables, a.printed)

    hits, misses = [], []
    for tok, (line_no, line) in sorted(nums.items(), key=lambda kv: -len(kv[0])):
        found = tok in hay or tok in text
        if not found and '.' not in tok and len(tok) >= 4:
            found = f'{int(tok):,}' in text or tok in text.replace(',', '')
        (hits if found else misses).append((tok, line_no, line))

    say('paper               : %s' % ', '.join(os.path.basename(t) for t in a.tex))
    say(f'numeric tokens      : {len(nums)}')
    say(f'traced to results   : {len(hits)}')
    say(f'not located         : {len(misses)}')
    say('')
    if misses:
        say('Not located in data/, results/tables/ or results/printed_outputs/:')
        for tok, line_no, line in sorted(misses, key=lambda t: str(t[1])):
            say(f'  {tok:>12}  {line_no}')
            if a.verbose:
                say(f'               {line[:150]}')
    else:
        say('Every numeric token in the paper appears in the released results.')

    if a.out:
        with open(a.out, 'w', encoding='utf8') as f:
            f.write('\n'.join(out) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
