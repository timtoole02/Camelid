#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


sys.dont_write_bytecode = True
RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
SCRIPT = RUNNER / "summarize_cap16_abba.py"
SPEC = importlib.util.spec_from_file_location("summarize_cap16_abba", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZER)


def round_line(index: int, wall_ms: str, accepted: int, width: int) -> str:
    return (
        f"[mtp round] #{index} wall={wall_ms}ms "
        f"(assistant=1.00ms, verifier=2.00ms) accepted={accepted}/{width}"
    )


class Cap16AbbaThroughputTest(unittest.TestCase):
    def parse(self, lines: list[str]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return SUMMARIZER.decode_throughput(path)

    def test_exact_four_round_fixture_computes_48_token_throughput(self) -> None:
        kind, rounds, throughput = self.parse(
            [
                round_line(0, "250.00", 13, 14),
                round_line(1, "250.00", 12, 13),
                round_line(2, "250.00", 13, 14),
                round_line(3, "250.00", 6, 7),
            ]
        )
        self.assertEqual(kind, "mtp")
        self.assertEqual(rounds, 4)
        self.assertEqual(throughput, Decimal("48"))

    def test_missing_duplicate_or_spec_rounds_fail_closed(self) -> None:
        valid = [
            round_line(0, "250.00", 13, 14),
            round_line(1, "250.00", 12, 13),
            round_line(2, "250.00", 13, 14),
            round_line(3, "250.00", 6, 7),
        ]
        for lines in (valid[:3], valid + [valid[-1]], [line.replace("[mtp", "[spec") for line in valid]):
            with self.subTest(lines=lines), self.assertRaises(SUMMARIZER.ReceiptError):
                self.parse(lines)

    def test_width_or_emitted_token_drift_fails_closed(self) -> None:
        wrong_width = [
            round_line(0, "250.00", 13, 14),
            round_line(1, "250.00", 12, 13),
            round_line(2, "250.00", 13, 15),
            round_line(3, "250.00", 6, 7),
        ]
        wrong_emitted = [
            round_line(0, "250.00", 12, 14),
            round_line(1, "250.00", 12, 13),
            round_line(2, "250.00", 13, 14),
            round_line(3, "250.00", 6, 7),
        ]
        for lines in (wrong_width, wrong_emitted):
            with self.subTest(lines=lines), self.assertRaises(SUMMARIZER.ReceiptError):
                self.parse(lines)


if __name__ == "__main__":
    unittest.main()
