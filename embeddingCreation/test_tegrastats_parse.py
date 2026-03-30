from __future__ import annotations

import json
import unittest

from tegrastats_parse import parse_tegrastats_line


class TegrastatsParseTests(unittest.TestCase):
    def test_parse_realistic_line_with_bracket_gr3d(self) -> None:
        line = (
            "03-30-2026 14:16:54 RAM 3320/7620MB (lfb 163x4MB) SWAP 42/3810MB "
            "CPU [15%@1510,10%@1510,22%@1510,9%@1510,off,off] "
            "EMC_FREQ 36%@1600 GR3D_FREQ 72%@[624] VDD_IN 6430mW/6088mW "
            "cpu@58.5C gpu@56.0C tj@60.2C"
        )
        row = parse_tegrastats_line(line)
        self.assertEqual(row["ram_used_mb"], "3320")
        self.assertEqual(row["ram_total_mb"], "7620")
        self.assertEqual(row["swap_used_mb"], "42")
        self.assertEqual(row["swap_total_mb"], "3810")
        self.assertEqual(row["emc_pct"], "36")
        self.assertEqual(row["emc_mhz"], "1600")
        self.assertEqual(row["gr3d_pct"], "72")
        self.assertEqual(row["gr3d_mhz"], "624")
        self.assertEqual(row["vdd_in_mw"], "6430")
        self.assertEqual(row["vdd_in_avg_mw"], "6088")
        self.assertEqual(row["temp_cpu_c"], "58.5000")
        self.assertEqual(row["temp_gpu_c"], "56.0000")
        self.assertEqual(row["temp_tj_c"], "60.2000")
        self.assertEqual(row["cpu0_pct"], "15")
        self.assertEqual(row["cpu1_pct"], "10")
        self.assertEqual(row["cpu2_pct"], "22")
        self.assertEqual(row["cpu3_pct"], "9")
        cpu_pairs = json.loads(row["cpu_usages_json"])
        self.assertEqual(len(cpu_pairs), 4)
        self.assertEqual(cpu_pairs[0]["mhz"], 1510)

    def test_parse_realistic_line_with_alt_gr3d_format(self) -> None:
        line = (
            "03-30-2026 14:17:04 RAM 3479/7620MB SWAP 43/3810MB "
            "CPU [32%@1510,28%@1510,31%@1510,30%@1510] EMC_FREQ 44%@2133 "
            "GR3D_FREQ 99%@918 VDD_IN 7812mW/7001mW cpu@62.0C gpu@61.5C"
        )
        row = parse_tegrastats_line(line)
        self.assertEqual(row["gr3d_pct"], "99")
        self.assertEqual(row["gr3d_mhz"], "918")
        self.assertEqual(row["temp_tj_c"], "")
        self.assertTrue(row["cpu_mean_pct"])
        temps = json.loads(row["temps_json"])
        self.assertIn("cpu", temps)
        self.assertIn("gpu", temps)


if __name__ == "__main__":
    unittest.main()
