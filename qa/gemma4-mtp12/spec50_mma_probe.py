#!/usr/bin/env python3
"""No-Cargo Metal proof/benchmark of the production 12B K8 head kernels."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=262144)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    root = Path(__file__).resolve().parents[2]
    source = (root / 'src/metal/spec50_head.rs').read_text()
    shader = source.split('const SPEC50_HEAD_SHADER: &str = r#"', 1)[1].split('"#;', 1)[0]
    with tempfile.TemporaryDirectory(prefix='spec50-mma-') as temp:
        temp = Path(temp)
        baseline = temp / 'baseline.metal'
        baseline.write_text('#define SPEC50_ROWS_PER_SG 8\n#define SPEC50_ROWS_PER_STEP 2\n'
                            '#define SPEC50_SG_PER_TG 4\n#define SPEC50_FLAT 1\n'
                            '#define SPEC50_YFMT 2\n' + shader)
        candidate = temp / 'mma.metal'
        candidate.write_text((root / 'src/metal/spec50_mma_head.metal').read_text())
        variants = temp / 'variants.txt'
        # Each line gives rows/simdgroup and simdgroups/threadgroup. MMA's
        # 4 groups cooperate on one 8-row tile, so its grid parameter is 2.
        variants.write_text(f'{baseline} 8 4\n{candidate} 2 4\n{baseline} 8 4\n')
        binary = temp / 'probe'
        subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O2', '-fobjc-arc',
                        '-framework', 'Foundation', '-framework', 'Metal',
                        str(Path(__file__).with_suffix('.mm')), '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(variants), str(args.rows), '8'], check=True)


if __name__ == '__main__':
    main()
