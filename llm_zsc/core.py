from transformers import LlamaForCausalLM, LlamaTokenizerFast
from typing import List, Union, Optional
import torch.nn.functional as F
from llm_zsc.utils import string_to_variations_list
from llm_zsc.manual_parameters import TRUE_STRING, FALSE_STRING
import torch

true_variations = string_to_variations_list(TRUE_STRING)
false_variations = string_to_variations_list(FALSE_STRING)


class CausalLMZeroShotClassificationPipeline:

    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, type_of_sequence: str = 'comment',
                 type_of_class='theme', override_system_prompt=None, override_context_messages=None,
                 use_system_prompt=True, llm_response_preamble=None):
        self.model = model

        # update generation config so we are not doing sampling.
        self.model.generation_config.do_sample = None
        self.model.generation_config.temperature = None
        self.model.generation_config.max_length = 2000
        self.model.generation_config.top_p = None

        self.tokenizer = tokenizer
        self.llm_response_preamble = llm_response_preamble
        self.type_of_sequence = type_of_sequence
        self.type_of_class = type_of_class

        if self.llm_response_preamble is not None:
            self.model.generation_config.token_healing = True  # ensure we heal the last token.

        if override_system_prompt is None:
            self.system_prompt = f"Your role is to classify whether a given text string belongs to a specific '{type_of_class}'. Simply respond with either 'True' or 'False' (without quotes). You should only return 'True' if it specifically mentions that {type_of_class}."
        else:
            self.system_prompt = override_system_prompt

        if override_context_messages is None:
            self.example_messages = [
                {
                    "role": "user",
                    "content": f'Please respond with True or False to whether or not the following comment from an employee is expressing the theme "My organisation has a clear strategy.":\n"The direction of the organization is so well communicated now! My teams have a new direction and they are far more motivated to get their work done."'
                },
                {
                    "role": "assistant",
                    "content": TRUE_STRING,
                },
            ]
        else:
            self.example_messages = override_context_messages

        system_prompt_messages = [{"role": "system", "content": self.system_prompt}] if use_system_prompt else []

        self.context_messages = [
            *system_prompt_messages,
            *self.example_messages
        ]

        self.all_true_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in true_variations]
        self.all_false_ids = [self.tokenizer.encode(f, add_special_tokens=False)[0] for f in false_variations]

        self.unique_true_ids = list(set(self.all_true_ids) - set(self.all_false_ids))
        self.unique_false_ids = list(set(self.all_false_ids) - set(self.all_true_ids))
        self.all_true_or_false_ids = list(set(self.unique_true_ids + self.unique_false_ids))

    def classify_zero_shot(self, string_to_classify: str,
                           classes_list: List[str], multi_label=False):

        scores = []
        for class_value in classes_list:
            messages = [
                *self.context_messages,
                {
                    "role": "user",
                    # TODO: Add ability to customise this last message, and in general revisit prompts to see if we need to make them be more specific.
                    "content": f'Great! That was perfect. This is exactly what I want. Now please respond with either True or False for whether the following {self.type_of_sequence} should be categorised as belonging to the {self.type_of_class} "{class_value}":\n"{string_to_classify}"'
                },
            ]

            base_prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if self.llm_response_preamble is not None:
                base_prompt_str += self.llm_response_preamble

            inputs = self.tokenizer(base_prompt_str, return_tensors='pt').to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]

            true_false_logits = logits[0, self.all_true_or_false_ids]
            true_false_probs = F.softmax(true_false_logits, dim=0)

            true_probs = true_false_probs[[self.all_true_or_false_ids.index(id) for id in self.unique_true_ids]]
            false_probs = true_false_probs[[self.all_true_or_false_ids.index(id) for id in self.unique_false_ids]]

            true_prob = true_probs.sum().item()
            false_prob = false_probs.sum().item()

            total_prob = true_prob + false_prob
            true_prob = true_prob / total_prob

            max_logit_id = torch.argmax(logits).item()
            if max_logit_id not in self.all_true_or_false_ids:
                most_likely_next_token = self.tokenizer.decode(max_logit_id)
                raise ValueError(
                    f"The most likely next token is neither True nor False. It was `{most_likely_next_token}`. "
                    f"Check prompt and quality of model.")

            scores.append(true_prob)

        if not multi_label:
            # Apply softmax to normalize scores across all classes
            scores = F.softmax(torch.tensor(scores), dim=0).tolist()

        combined = sorted(zip(classes_list, scores), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_scores = zip(*combined)

        return {
            'labels': list(sorted_labels),
            'scores': list(sorted_scores),
            'sequence': string_to_classify
        }

    def __call__(self, sequences: Union[str, List[str]], candidate_labels: Union[str, List[str]],
                 multi_label: Optional[bool] = False):
        '''
            sequences (`str` or `List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.
        '''
        if isinstance(candidate_labels, str):
            classes_list = [candidate_labels]
        elif isinstance(candidate_labels, list):
            classes_list = candidate_labels
        else:
            raise NotImplementedError('Parameter `candidate_labels` must be either a str or a lst of str.')

        if len(classes_list) == 1:
            multi_label = True  # If only one class, they must be independent probabilities

        if isinstance(sequences, str):
            return self.classify_zero_shot(string_to_classify=sequences,
                                           classes_list=classes_list,
                                           multi_label=multi_label)
        elif isinstance(sequences, list):
            results_list = []
            for sequence in sequences:
                result = self.classify_zero_shot(string_to_classify=sequence,
                                                 classes_list=classes_list,
                                                 multi_label=multi_label)
                results_list.append(result)
            return results_list
        else:
            raise NotImplementedError('Parameter `sequences` must be either a str or a lst of str.')


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = 'cuda:0'

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)

    from pprint import pprint

    themes = ['I am not empowered to do my job.', 'I am empowered to do my job',
              'The organization\'s broader strategy (meaning the higher level direction of the organisation, not operational issues) is not clear.']
    verbatim = ['There is no accountability. People are doing shit they shouldn\'t.']

    classifier = CausalLMZeroShotClassificationPipeline(model=model, tokenizer=tokenizer)

    probabilities = classifier(verbatim, themes, multi_label=True)

    pprint(probabilities)
