import re

from ...smp import *


FAIL_MSG = 'Failed to obtain answer via API.'

# Datasets that use zero-shot extraction (no few-shot examples) + regex <answer> tag extraction
ZEROSHOT_EXTRACT_DATASETS = ['DocVQA', 'InfoVQA', 'TextVQA', 'ChartQA']


def extract_answer_from_tag(prediction):
    """Try to extract answer from <answer></answer> tag using regex."""
    match = re.search(r'<answer>(.*?)</answer>', str(prediction), re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_vqa_ICE():
    example_1 = """
Question: What is the title of the chart?\n
Model response: Looking at the image, I can see that the chart is titled "Population Growth Rate". The chart shows demographic trends over several decades with clear labeling at the top.\n
Extracted answer: Population Growth Rate
"""

    example_2 = """
Question: What is the value shown for 2020?\n
Model response: Let me analyze the chart carefully. The value for 2020 appears to be 42.5%, which represents a significant increase compared to the previous year.\n
Extracted answer: 42.5%
"""

    example_3 = """
Question: What color is the largest segment?\n
Model response: The largest segment in the pie chart is colored blue. It takes up approximately 45% of the total area.\n
Extracted answer: blue
"""

    example_4 = """
Question: Who is the author of this document?\n
Model response: Based on the document header, the author is Dr. John Smith from the Department of Computer Science.\n
Extracted answer: Dr. John Smith
"""

    example_5 = """
Question: How many people are in the image?\n
Model response: I can count the people in this image. There are 5 people visible - three standing in the front row and two in the back.\n
Extracted answer: 5
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_vqa_extract_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_vqa_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Question: ' + question + '\n'
    prompt += 'Model response: ' + prediction + '\n'
    prompt += 'Extracted answer:'
    return prompt


def build_vqa_extract_prompt_zeroshot(line):
    question = line['question']
    prediction = str(line['prediction'])
    prompt = 'Extract the concise answer (a single word or short phrase) from the model response.\n\n'
    prompt += 'Question: ' + question + '\n'
    prompt += 'Model response: ' + prediction + '\n'
    prompt += 'Extracted answer:'
    return prompt


def VQA_auxeval(model, line, dataset=None):
    prediction = str(line['prediction'])

    # For specified datasets, first try regex extraction from <answer> tag
    use_zeroshot = dataset and listinstr(ZEROSHOT_EXTRACT_DATASETS, dataset)
    if use_zeroshot:
        tag_answer = extract_answer_from_tag(prediction)
        if tag_answer:
            return dict(log='Extracted from <answer> tag', res=tag_answer)

    # Build prompt: zero-shot for specified datasets, few-shot for others
    if use_zeroshot:
        prompt = build_vqa_extract_prompt_zeroshot(line)
    else:
        prompt = build_vqa_extract_prompt(line)

    log = ''
    retry = 5
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {line["prediction"]}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=prediction)
