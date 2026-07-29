#!/usr/bin/env node
// Generate the tekken tokenizer oracle pack from the pinned llama.cpp
// `llama-tokenize`, for `tokenizer::tests::tekken_tokenizer_matches_pinned_oracle`.
//
//   node scripts/gen-tekken-tokenizer-oracle.mjs \
//     --gguf <Mistral-Nemo-*.gguf> --bin <llama-tokenize.exe> --commit <sha> \
//     > qa/prompt-packs/mistral-nemo-tekken-tokenizer-oracle-v1.json
//
// The cases deliberately concentrate on the two places tekken differs from
// gpt-4o (no contraction group; single-digit grouping) plus the branches they
// share, since a pre-tokenizer bug is silently-different tokens, not a crash.

import { execFileSync } from 'node:child_process';
import { Buffer } from 'node:buffer';

const args = process.argv.slice(2);
const arg = (name) => {
  const i = args.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= args.length) {
    throw new Error(`missing --${name}`);
  }
  return args[i + 1];
};

const gguf = arg('gguf');
const bin = arg('bin');
const commit = arg('commit');

const CASES = [
  // --- tekken difference 1: no contraction group in the word alternatives ---
  ['contraction-s', "John's book"],
  ['contraction-t', "don't stop"],
  ['contraction-re', "we're here"],
  ['contraction-ve', "they've gone"],
  ['contraction-ll', "it'll work"],
  ['contraction-d', "I'd like"],
  ['contraction-upper', "JOHN'S BOOK"],

  // --- tekken difference 2: digits are one segment each ---
  ['digits-run', '1234567890'],
  ['digits-embedded', 'abc123def'],
  ['digits-mixed', 'v2.1.0 build 4567'],
  ['digits-leading-space', ' 42'],

  // --- shared: case-run word grammar (prefix? UPPER* LOWER+ | UPPER+ LOWER*) ---
  ['camel-case', 'HelloWorld'],
  ['camel-acronym', 'getHTTPResponse'],
  ['all-caps', 'HELLO WORLD'],
  ['single-upper', 'A b C d'],
  ['leading-space-word', ' hello'],
  ['punct-prefix-word', '(hello'],

  // --- shared: punctuation branch, whose tail is [\r\n/]* (note the slash) ---
  ['slash-path', 'a/b/c'],
  ['url', 'https://example.com/path?q=1'],
  ['punct-run', '!!!???'],
  ['punct-newline', 'x!\ny'],

  // --- shared: whitespace branches ---
  ['multi-space', 'a   b'],
  ['trailing-space', 'end   '],
  ['newlines', 'x\n\ny'],
  ['tab', 'a\tb'],
  ['leading-newline', '\nstart'],

  // --- unicode: the ASCII case classes make non-ASCII letters both upper- and
  // lower-like, which is the subtle part of this dialect ---
  ['accented-lower', 'élan vital'],
  ['accented-upper', 'ÉLAN VITAL'],
  ['accented-mixed', 'Café Zürich'],
  ['cyrillic', 'Привет мир'],
  ['greek', 'Ελληνικά'],
  ['cjk', '日本語のテキスト'],
  ['korean', '한국어 텍스트'],
  ['emoji', 'hi 👋 there 🎉'],
  ['combining', 'égalité'],

  // --- prose and code, the realistic shapes ---
  ['prose', 'The quick brown fox jumps over the lazy dog.'],
  ['code', 'fn main() { let x = vec![1, 2, 3]; }'],
  ['json', '{"key": "value", "n": 42}'],
  ['mixed', "Mistral's Nemo-12B: 128K ctx, released 2024/07/18!"],
];

const cases = CASES.map(([id, input]) => {
  // --no-bos     : the vector is the pre-tokenizer + BPE result alone, with no
  //                BOS policy folded in.
  // --no-escape  : REQUIRED. Without it llama-tokenize interprets `\n`/`\t`
  //                sequences in the argument, so any case containing a literal
  //                backslash would be re-interpreted and the oracle would not
  //                describe the bytes we actually feed Camelid.
  // --log-disable: keep model-load chatter off stderr.
  const out = execFileSync(
    bin,
    ['-m', gguf, '-p', input, '--ids', '--no-bos', '--no-escape', '--log-disable'],
    { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
  );
  // llama-tokenize --ids prints a bracketed list, e.g. "[1, 2, 3]".
  const match = out.match(/\[[^\]]*\]/);
  if (!match) {
    throw new Error(`could not parse ids for case ${id}: ${out}`);
  }
  const ids = JSON.parse(match[0]);
  return {
    id,
    input,
    input_utf8_hex: Buffer.from(input, 'utf8').toString('hex'),
    expected_token_ids: ids,
  };
});

process.stdout.write(
  `${JSON.stringify(
    {
      oracle: {
        command: `llama-tokenize -m <gguf> -p <input> --ids --no-bos --no-escape --log-disable`,
        commit,
        model: 'Mistral-Nemo-Instruct-2407-Q4_K_M.gguf',
        pre_tokenizer: 'tekken',
      },
      cases,
    },
    null,
    2,
  )}\n`,
);
