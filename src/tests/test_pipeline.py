import unittest, sys, os

#Solução para chamar os outros modulos fora de test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', 'src')))

from core.signal_buffer import SignalBuffer
from core.translator import Translator


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.buffer = SignalBuffer(size=5, min_confidence=0.6)

        self.translator = Translator()

    def test_pipeline_translation(self):
        simulated_signals = [
        "A", "A", "A", "A", "A"
        ]

        stable_signal = None

        for signal in simulated_signals:
            result = self.buffer.update(signal)
            if result:
                stable_signal = result

        self.assertIsNotNone(stable_signal)
        self.assertEqual(stable_signal, "A")

        translated = self.translator.translate(stable_signal)

        self.assertIsInstance(translated, str)
        self.assertEqual(translated.lower(), "a")


if __name__ == "__main__":
    unittest.main()
