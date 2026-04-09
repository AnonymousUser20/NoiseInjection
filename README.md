# NoiseInjection

### Project Structure

```text
NoiseInjection/
│── Error Categorization/
│── Samples
│   ├── gsm8k_original/
│   ├── gsm_symbolic/
│   └── math_500/
│   └── omni_math/
│── Clean_Script_for_Execution.py
│── README.md
```
- The Samples directory contains all samples from each dataset, that we have considered for our experiment.
- The ```Clean_Script_for_Execution.py``` python contains the entire pipeline for evaluation.

Datasets used :
- GSM8K
- GSM-Symbolic
- MATH-500
- Omni-MATH

Prompting Strategies Used :
- Normal
- Chain-of-Thought (CoT)
- Denoise CoT
- Macro Action
- Process Reward Model (PRM)

Noise Levels :
- Original
- Low-noise
- Medium Noise
- High Noise

<br><br><br><br>

<img width="880" height="578" alt="image" src="https://github.com/user-attachments/assets/5a8ac7b9-331d-46e9-8178-367ad58abbef" />

<br><br><br><br>


The difficulty rating distribution of the samples considered for our empirical evaluation is given below :
<br>
<img width="1000" height="600" alt="difficulty_bell_curves_by_dataset" src="https://github.com/user-attachments/assets/7a8d6915-112c-463d-aeb3-d0720ad88e43" />

The difficulty ratings have been calculated using the prompts and hierarchy tree in Omni-MATH dataset, after masking names of the dataset from which each sample was selected.
```
@article{gao2024omni,
  title={Omni-math: A universal olympiad level mathematic benchmark for large language models},
  author={Gao, Bofei and Song, Feifan and Yang, Zhe and Cai, Zefan and Miao, Yibo and Dong, Qingxiu and Li, Lei and Ma, Chenghao and Chen, Liang and Xu, Runxin and others},
  journal={arXiv preprint arXiv:2410.07985},
  year={2024}
}
```
<br><br><br><br>

# Normal Prompting
```
if method == "normal":
            return f"""You are a math expert. Solve the following math problem and give the final answer only.
                    Respond in JSON format with the final answer wrapped in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""
```
# Chain-of-Thoughts Prompting

The model is prompted to generate a reasoning for its answer.

```
elif method == "cot":
            return f"""You are a math expert. Solve the following math problem step-by-step.

                    Respond in JSON format with:
                    - "reasoning": your step-by-step explanation.
                    - "final_answer": the final answer wrapped in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "reasoning": "Let's solve it step by step...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""
```
# Denoise Chain-of-Though Prompting

The model is prompted to denoise the perturbed question, before generating a solution.

```
elif method == "denoise_cot":
            return f"""You are a math expert. The following math question may contain irrelevant or noisy information.

                    Your task:
                    1. Identify irrelevant/noisy sentences (that are not needed to solve the math problem).
                    2. Provide a cleaned version of the question without the noise.
                    3. Solve the cleaned question step by step.

                    Respond in JSON format using the following keys:
                    - "noisy_sentences": Wrap the list in <noisy_sentences></noisy_sentences> tags.
                    - "denoised_question": Wrap the cleaned question in <denoised_question></denoised_question> tags.
                    - "solution_steps": The step-by-step reasoning.
                    - "final_answer": Wrap the final answer in <answer></answer> tags.

                    Now analyze and solve:
                    {question}

                    Answer:
                    {{
                      "noisy_sentences": "<noisy_sentences>LIST_NOISY_SENTENCES</noisy_sentences>",
                      "denoised_question": "<denoised_question>CLEANED_QUESTION_HERE</denoised_question>",
                      "solution_steps": "Step-by-step reasoning here...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""

```
# Macro-action Prompting

The model is prompted to generate the reasoning, foloowing a templat, consisting of detailed steps.

```
elif method == "macro_action":
            return f"""You are a math expert. The question below may contain irrelevant or noisy information.

                    Your task:
                    - Assume: Identify quantities and facts that are likely relevant for solving the problem.
                    - Simplify: Rewrite the problem by removing irrelevant details.
                    - Verify: Check that the simplified problem still preserves the original question's intent.
                    - Solve: Answer the problem step-by-step.

                    Follow these four stages exactly and clearly label your output at each stage.

                    Respond in JSON format using the following keys:
                    - "assumptions": Relevant quantities and facts.
                    - "simplified_question": The rewritten problem statement.
                    - "verification": Verification notes.
                    - "solution_steps": The detailed step-by-step reasoning.
                    - "final_answer": Wrap the final answer in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "assumptions": "Relevant facts here...",
                      "simplified_question": "Simplified question here...",
                      "verification": "Verification notes here...",
                      "solution_steps": "Step-by-step reasoning here...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""
```
# PRM

```
# === 3.  Process-Reward-Modeling (PRM) =====================================
    def query_llm_prm(self, question: str, n_candidates: int = 6) -> str:
        prm_prompt = self.create_prompt(question, method="denoise_cot")

        # First pass: obtain cleaned question once (deterministic call)
        cleaned_resp = self.query_llm(prm_prompt.replace("temperature\": 0", "temperature\": 0"))
        try:
            cleaned_json = json.loads(cleaned_resp)
            cleaned_q_text = cleaned_json.get("denoised_question", "")
            # FIX: Ensure cleaned_q_text is a string
            if isinstance(cleaned_q_text, list):
                cleaned_q_text = ' '.join(str(item) for item in cleaned_q_text)
            elif not isinstance(cleaned_q_text, str):
                cleaned_q_text = str(cleaned_q_text)
        except Exception:
            cleaned_q_text = question  # fallback

        # Second pass: sample diverse candidate solutions
        candidates = []
        for _ in range(n_candidates):
            cand_resp = self._query_llm_sample(prm_prompt, temperature=0.9, top_p=0.9)
            candidates.append(cand_resp)

        # Score & select
        scored = [(cand, self._prm_score(cand, cleaned_q_text)) for cand in candidates]
        best = max(scored, key=lambda x: x[1])[0]
        return best
```

- Uses the denoise-CoT template to generate multiple candidates.
- Scores each candidate with heuristic PRM filters.
- Returns the highest-scoring answer.

### PRM Score

```
def _prm_score(self, response: str, cleaned_q: str) -> float:
        """Heuristic score ⇒ higher = better."""
        try:
            resp_json = json.loads(response)
        except Exception:
            return 0.0

        nums_in_q = set(self._extract_numbers(cleaned_q))

        solution_steps = resp_json.get("solution_steps", "")
        nums_in_ans = set(self._extract_numbers(solution_steps))

        num_check = 1.0 if nums_in_ans <= nums_in_q else 0.0
        arith_check = min(self._arith_consistency_score(resp_json) / 10.0, 1.0)
        jitter = random.random() * 1e-3
        return num_check + arith_check + jitter
```

- if numeric values in solution are <= numeric values in question : num_check = ``` True ```
- arithmatic score ranges from ``` 0-1 ``` calculated using the following function
  ```
  def _arith_consistency_score(self, response_json: dict) -> float:
        
        steps = response_json.get("solution_steps", "")
        # FIX: Ensure steps is a string
        if isinstance(steps, list):
            steps = ' '.join(str(item) for item in steps)
        elif not isinstance(steps, str):
            steps = str(steps)

        if not steps:
            return 0.0
        eq = steps.count("=")
        bad = steps.lower().count("/0")
        return max(eq - 3 * bad, 0)
  ```
  eq represents equalities. Thus, for every equality the score is increased but for every penalty
  is three times. The lowest value is always 0.
- ```jitter``` is added as a tie-breaker (very small value).

