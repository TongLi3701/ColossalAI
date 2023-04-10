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
tokenizer = AutoTokenizer.from_pretrained("/data3/data/Coati-last6")
USE_8BIT = False

samples = [
    ([
        Dialogue(
            instruction="A pet store had 64 puppies. In one day they sold 28 of them and put the rest into cages with 4 in each cage. How many cages did they use ？ Let's think step by step",
            response=""
        ),
    ], 128
    ),
]

class TestInference(unittest.TestCase):
    def testInferenceWithoutCot(self):
        processor = ChatPromptProcessor(tokenizer, CONTEXT, 256)
        model = LlamaForCausalLM.from_pretrained(
                    "/data3/data/Coati-last6",
                    load_in_8bit=USE_8BIT,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        model.eval()
        for history, max_new_tokens in samples:
            prompt = processor.preprocess_prompt(history, max_new_tokens)
            inputs = {k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt").items()}
            prompt_len = inputs['input_ids'].size(1)
            output = model.generate(**inputs, max_length = 256)
            response = output[0, prompt_len:]
            out_string = tokenizer.decode(response, skip_special_tokens=True)
            out_string = processor.postprocess_output(out_string)

    def testInferenceWithCot(self):
        pass


if __name__ == "__main__":
    unittest.main()
