import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze import CumulativeOrdinalLogistic, parse_labels, read_session


class ParseLabelsTests(unittest.TestCase):
    def test_session_uses_last_rating_event_even_when_it_is_unanswered(self):
        fields = ["participant", "event", "trial", "condition", "rep", "phase", "audio_saved", "note"]
        rows = [
            {"participant": "P01", "event": "session_start", "audio_saved": "true"},
            {"participant": "P01", "event": "trial_start", "trial": "1", "condition": "free", "rep": "1"},
            {"participant": "P01", "event": "rating", "trial": "1", "condition": "free", "note": "4"},
            {"participant": "P01", "event": "rating", "trial": "1", "condition": "free", "note": ""},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "session.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            trials, participant, audio_saved = read_session(path)

        self.assertEqual(participant, "P01")
        self.assertTrue(audio_saved)
        self.assertIsNone(parse_labels(trials[0])["ease"])

    def test_uses_final_rating_and_keeps_stuck_as_separate_binary_label(self):
        trial = {
            "trial": "3",
            "condition": "free",
            "rating": {"note": "4"},
            "affect_rating": {
                "affect_valence": "1",
                "affect_arousal": "-1",
                "affect_label": "calm_pleasant",
            },
            "stuck_count": 2,
        }

        labels = parse_labels(trial)

        self.assertEqual(labels["ease"], 4)
        self.assertEqual(labels["affect_label"], "calm_pleasant")
        self.assertEqual(labels["stuck_count"], 2)
        self.assertEqual(labels["stuck"], 1)

    def test_stuck_label_is_not_assigned_outside_free_trials(self):
        trial = {
            "trial": "1",
            "condition": "read",
            "rating": None,
            "affect_rating": None,
            "stuck_count": 0,
        }

        self.assertIsNone(parse_labels(trial)["stuck"])

    def test_rejects_affect_label_that_disagrees_with_numeric_values(self):
        trial = {
            "trial": "2",
            "condition": "free",
            "rating": None,
            "affect_rating": {
                "affect_valence": "-1",
                "affect_arousal": "1",
                "affect_label": "pleasant",
            },
            "stuck_count": 0,
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_labels(trial)


class CumulativeOrdinalLogisticTests(unittest.TestCase):
    def test_predictions_use_observed_ordered_classes(self):
        x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.asarray([1, 1, 2, 3, 3, 3])
        model = CumulativeOrdinalLogistic(c=1.0).fit(x, y)

        predicted = model.predict(x)

        self.assertTrue(set(predicted).issubset({1, 2, 3}))
        self.assertEqual(len(predicted), len(y))


if __name__ == "__main__":
    unittest.main()
