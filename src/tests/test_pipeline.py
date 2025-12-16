import unittest

from core.signal_buffer import SignalBuffer
from core.translator import Translator


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.buffer = SignalBuffer(
            buffer_size=5,
            confidence_threshold=0.6
        )

        self.translator = Translator()

    def test_pipeline_translation(self):
        """
        Simula um fluxo completo:
        sinal bruto -> buffer -> tradução
        """

        simulated_signals = [
            ("A", 0.92),
            ("A", 0.93),
            ("A", 0.91),
            ("A", 0.95),
            ("A", 0.94),
        ]

        stable_signal = None

        for signal, confidence in simulated_signals:
            result = self.buffer.add(signal, confidence)
            if result:
                stable_signal = result

        self.assertIsNotNone(stable_signal)
        self.assertEqual(stable_signal, "A")

        translated = self.translator.translate(stable_signal)

        self.assertIsInstance(translated, str)
        self.assertEqual(translated.lower(), "a")


if __name__ == "__main__":
    unittest.main()
