import unittest

from finetune_sentiment import (
    FineTuneConfig,
    classification_metrics,
    inference_messages,
    label_name,
    normalize_prediction,
    training_messages,
)


class FineTuneConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = FineTuneConfig()
        self.assertIs(config.validated(), config)

    def test_split_sizes_must_leave_training_data(self):
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            FineTuneConfig(test_size=0.5, validation_size=0.5).validated()


class PromptFormattingTest(unittest.TestCase):
    def test_numeric_labels_match_financial_phrasebank_order(self):
        self.assertEqual(
            [label_name(index) for index in range(3)],
            ["negative", "neutral", "positive"],
        )

    def test_training_messages_include_exact_answer(self):
        messages = training_messages(" Revenue increased. ", 2)

        self.assertEqual(messages[-1], {"role": "assistant", "content": "positive"})
        self.assertEqual(messages[1]["content"], "Revenue increased.")

    def test_inference_messages_do_not_include_an_answer(self):
        messages = inference_messages("Revenue was unchanged.")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_normalize_prediction_rejects_ambiguous_output(self):
        self.assertEqual(normalize_prediction("Positive."), "positive")
        self.assertEqual(normalize_prediction("positive or neutral"), "unknown")
        self.assertEqual(normalize_prediction("unclear"), "unknown")


class MetricsTest(unittest.TestCase):
    def test_metrics_include_accuracy_macro_f1_and_unknowns(self):
        metrics = classification_metrics(
            ["positive", "neutral", "negative"],
            ["positive", "unknown", "negative"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["unknown_predictions"], 1)
        self.assertGreater(metrics["macro_f1"], 0)


if __name__ == "__main__":
    unittest.main()
