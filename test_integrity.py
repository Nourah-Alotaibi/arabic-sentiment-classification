import tempfile, unittest
from pathlib import Path
import pandas as pd
from train import normalize, prepare


class SentimentIntegrity(unittest.TestCase):
    def test_normalization_keeps_negation_and_emoji(self):
        text = normalize("أَنَا لا أحب هذا 😞 https://example.com @someone")
        self.assertIn("لا", text)
        self.assertIn("😞", text)
        self.assertIn("URL", text)
        self.assertNotIn("example.com", text)

    def test_duplicates_and_conflicting_labels_cannot_leak(self):
        rows = [
            {
                "Text": f"نص فريد {i} تجربة",
                "Sentiment": ["negative", "neutral", "positive"][i % 3],
            }
            for i in range(90)
        ]
        rows.extend(
            [
                rows[0].copy(),
                {"Text": "نص مختلف متعارض", "Sentiment": "positive"},
                {"Text": "نص مختلف متعارض", "Sentiment": "negative"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "results"
            out.mkdir()
            pd.DataFrame(rows).to_csv(root / "input.csv", index=False)
            d, dev, tr, va, te, audit = prepare(root / "input.csv", out)
            self.assertEqual(audit["conflicting_label_rows_removed"], 2)
            self.assertEqual(len(d), 90)
            self.assertFalse(d.normalized.duplicated().any())
            for a, b in [(tr, va), (tr, te), (va, te)]:
                self.assertFalse(set(d.iloc[a].normalized) & set(d.iloc[b].normalized))


if __name__ == "__main__":
    unittest.main()
