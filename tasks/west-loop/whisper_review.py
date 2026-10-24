"""Compare the Whisper variants of the West Loop run and build a review packet.

    python whisper_review.py WORKSPACE OUTPUT [--windows-per-session 12]

Writes OUTPUT/comparison.csv (one row per session and variant), OUTPUT/clips/*.wav and
OUTPUT/review.html: windows where variants disagree, plus agreeing windows as a control, each with
its audio and every variant's text. The page keeps answers in the browser and exports a CSV.
"""
import argparse
import html
import json
import random
import re
import subprocess
from pathlib import Path

import pandas as pd

VOCABULARY = {
    'curb ramp': r'\bcur\w*\s*(?:ramp|wrem|rem|rim)\w*', 'crosswalk': r'\bcross\s*walk',
    'pole': r'\bpoles?\b', 'obstacle': r'\bobstacles?\b|\bconstruction\b|\btrash\b|\bhydrant\b',
    'signal': r'\bsignals?\b|\bwait\b|\bhold\b',
}
NOT_LATIN = re.compile(r'[^\x00-\x7FÀ-ɏ‘-”]')  # Cyrillic, CJK...: invented in English speech


def flags(segment):
    text = segment['text']
    return dict(
        low_logprob=(segment.get('avg_logprob') or 0) < -1,
        high_no_speech=(segment.get('no_speech_prob') or 0) > 0.6,
        repetitive=(segment.get('compression_ratio') or 0) > 2.4,
        foreign_script=bool(NOT_LATIN.search(text)),
    )


def load(workspace):
    rows = []
    for path in sorted(Path(workspace, 'transcripts').glob('*/*.json')):
        record = json.loads(path.read_text())
        for segment in record['segments']:
            rows.append(dict(variant=path.parent.name, route=record['route'], start=segment['start'],
                             end=segment['end'], text=segment['text'], **flags(segment)))
    return pd.DataFrame(rows)


def comparison(segments):
    def summary(group):
        text = ' '.join(group.text).lower()
        any_flag = group[['low_logprob', 'high_no_speech', 'repetitive', 'foreign_script']].any(axis=1)
        return pd.Series(dict(
            segments=len(group), speech_min=(group.end - group.start).sum() / 60,
            low_logprob=int(group.low_logprob.sum()), repetitive=int(group.repetitive.sum()),
            foreign_script=int(group.foreign_script.sum()), any_flag=int(any_flag.sum()),
            **{f'said {term}': len(re.findall(rx, text)) for term, rx in VOCABULARY.items()},
        ))
    return segments.groupby(['route', 'variant']).apply(summary).reset_index()


def text_in(segments, variant, route, a, b):
    s = segments[(segments.variant == variant) & (segments.route == route) & (segments.end > a) & (segments.start < b)]
    return ' / '.join(s.text) or '—'


def windows(segments, per_session, rng):
    """Disagreements first (text in some variants only), then agreeing windows as a control."""
    picked = []
    variants = sorted(segments.variant.unique())
    for route, group in segments.groupby('route'):
        candidates = []
        for row in group.itertuples():
            a, b = max(0.0, row.start - 1), row.end + 1
            present = [v for v in variants if text_in(group, v, route, a, b) != '—']
            kind = 'agree' if len(present) == len(variants) else 'disagree'
            candidates.append((kind, round(a, 1), round(b, 1)))
        candidates = sorted(set(candidates))
        disagree = [c for c in candidates if c[0] == 'disagree']
        agree = [c for c in candidates if c[0] == 'agree']
        take = rng.sample(disagree, min(len(disagree), per_session * 2 // 3))
        take += rng.sample(agree, min(len(agree), per_session - len(take)))
        picked += [(route, kind, a, min(b, a + 20)) for kind, a, b in sorted(take, key=lambda c: c[1])]
    return picked, variants


def clip(video, a, b, path):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{a:.2f}', '-t', f'{b - a:.2f}', '-i', str(video),
                    '-vn', '-ac', '1', '-ar', '16000', str(path)], check=True)


def page(items, variants):
    rows = []
    for i, item in enumerate(items):
        texts = ''.join(
            f'<tr><th>{html.escape(v)}</th><td>{html.escape(item["texts"][v])}</td><td>'
            + ''.join(f'<label><input type="radio" name="{i}|{v}" value="{o}"> {o}</label> '
                      for o in ['correct', 'wrong words', 'invented', 'missed speech'])
            + '</td></tr>' for v in variants)
        rows.append(f'''<section><h2>{i + 1}. {html.escape(item["route"])} at {item["a"]:.0f}–{item["b"]:.0f} s</h2>
<audio controls preload="none" src="clips/{item["clip"]}"></audio>
<p>Is there speech? <label><input type="radio" name="{i}|speech" value="yes"> yes</label>
<label><input type="radio" name="{i}|speech" value="no"> no</label>
Notes <input name="{i}|notes" size="50"></p><table>{texts}</table></section>''')
    meta = json.dumps([dict(route=x['route'], a=x['a'], b=x['b'], kind=x['kind']) for x in items])
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Whisper review</title>
<style>body{{font:15px system-ui;margin:24px;max-width:1100px}}section{{border-top:1px solid #ccc;padding:8px 0}}
th{{text-align:left;padding-right:12px;white-space:nowrap}}td{{padding:2px 8px}}audio{{width:420px}}</style></head><body>
<h1>Whisper review</h1><p>Listen, then mark each variant's text. Answers stay in this browser; press Export when done.
Variants are listed in a fixed order; the window kind (disagreement or control) is hidden until export.</p>
<button onclick="exportCsv()">Export CSV</button>{''.join(rows)}<button onclick="exportCsv()">Export CSV</button>
<script>
const META = {meta};
const KEY = 'whisper-review';
let saved = {{}}; try {{ saved = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
for (const el of document.querySelectorAll('input')) {{
  const v = saved[el.name];
  if (el.type === 'radio') el.checked = v === el.value; else if (v) el.value = v;
  el.addEventListener('change', () => {{ saved[el.name] = el.value; try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}} }});
}}
function exportCsv() {{
  const lines = ['item,route,start_s,end_s,kind,field,answer'];
  for (const [name, value] of Object.entries(saved)) {{
    const [i, field] = name.split('|'); const m = META[+i];
    lines.push([+i + 1, m.route, m.a, m.b, m.kind, field, JSON.stringify(value)].join(','));
  }}
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n')], {{type: 'text/csv'}}));
  a.download = 'whisper_review.csv'; a.click();
}}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace')
    parser.add_argument('output')
    parser.add_argument('--windows-per-session', type=int, default=12)
    args = parser.parse_args()

    out = Path(args.output)
    (out / 'clips').mkdir(parents=True, exist_ok=True)
    segments = load(args.workspace)
    comparison(segments).round(2).to_csv(out / 'comparison.csv', index=False)

    picked, variants = windows(segments, args.windows_per_session, random.Random(0))
    items = []
    for route, kind, a, b in picked:
        name = f'{route}_{a:07.1f}.wav'
        clip(Path(args.workspace, 'data', route, 'video.mp4'), a, b, out / 'clips' / name)
        items.append(dict(route=route, kind=kind, a=a, b=b, clip=name,
                          texts={v: text_in(segments, v, route, a, b) for v in variants}))
    (out / 'review.html').write_text(page(items, variants))
    print(f'{len(items)} windows, {len(variants)} variants -> {out}')


if __name__ == '__main__':
    main()
