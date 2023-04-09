"""
Test inference with / without CoT
"""
import sys
import unittest
sys.path.append('..')

import torch
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from utils import ChatPromptProcessor, Dialogue

CONTEXT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.'
tokenizer = AutoTokenizer.from_pretrained("/data/scratch/LLaMa-7B")
USE_8BIT = False

samples = [
    ([
        Dialogue(
            instruction='Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?',
            response=""
        ),
    ], 128
    ),
]

class TestInference(unittest.TestCase):
    def testInferenceWithoutCot(self):
        processor = ChatPromptProcessor(tokenizer, CONTEXT, 256)
        model = LlamaForCausalLM.from_pretrained(
                    "/data/scratch/LLaMa-7B",
                    load_in_8bit=USE_8BIT,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        model.eval()
        for history, max_new_tokens in samples:
            prompt = processor.preprocess_prompt(history, max_new_tokens)
            inputs = {k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt").items()}
            generate_ids = model.generate(**inputs, max_length = 128)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(output)

    def testInferenceWithCot(self):
        pass


if __name__ == "__main__":
    unittest.main()
