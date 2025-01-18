import re
import random
import datasets

def gsm8k_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        question = doc["question"]
        target = doc["answer"].split("####")[-1].strip()
        leading = "Question: "
        return {
            "question": leading + question,
            "target": target
        }
    return dataset.map(_process_doc)

def gpqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
            doc["Correct Answer"],
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Correct Answer"])
        question = f"""
Question: {doc['Question']}
Choices:
(A) {choices[0]}
(B) {choices[1]}
(C) {choices[2]}
(D) {choices[3]}"""
        target = f"({chr(65 + correct_answer_index)})"
        return {
            "question": question,
            "target": target
        }
    return dataset.map(_process_doc)

def aqua_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    target format : (A)
    """
    def _process_doc(doc):
        q = doc["question"]
        options = doc["options"]
        options = [f"({opt}" for opt in options]##(A) option.....
        options = '\n'.join(options)
        target = doc["correct"]
        target = f"({target})"
        question = f"""
Question: {q}
Choices:
{options}"""
        return {
            "question": question,
            "target": target
        }
    return dataset.map(_process_doc)
