"""
Test inference with / without CoT
"""
import sys
import unittest
sys.path.append('..')
from transformers import AutoTokenizer
from utils import ChatPromptProcessor, Dialogue



class TestInference(unittest.TestCase):
    def testInferenceWithoutCot(self):
        pass

    def testInferenceWithCot(self):
        pass


if __name__ == "__main__":
    unittest.main()
