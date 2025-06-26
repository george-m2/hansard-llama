# hansard-Llama

A LLM fine-tuning project focused on parliamentary records and debates from the 1990s and 2000s.


| Item                | Details                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------|
| **Base model**      | Llama-3 8B with 4-bit quantization (`unsloth/llama-3-8b-bnb-4bit`)                        |
| **Compute** | 1x RTX 4090 24GB, AMD EPYC 7C13, 64GB RAM                                                            |
| **Training length** | ~7 hours, 5700 steps                                                                    |
| **Dataset**    | 1,517,832 samples                                                                         |
| **Parameters**| 0.52% of total base parameters                                                                 |

## Comparison
```hansard-llm``` provides more concise, policy-focused justification and prose found in Parliamentary debates, while the base LLaMA-3-8B model gives a broader explanation.

The following comparison was perfomed using the ```inference_comparison.ipynb``` notebook on a single RTX 3090.
Responses have been truncated. 

| Question | ```Llama-3-8B-Instruct``` | ```hansard-llm``` |
|----------|----------------------|------------------|
| **What were the main justifications provided by the government for the implementation of the furlough scheme during early COVID-19 debates?** | <br><br><ul><li><strong>Economic necessity</strong>: The furlough scheme was seen as a necessary measure to prevent mass job losses…</li><li><strong>Job preservation</strong>: The scheme was designed to help employers keep their staff on the payroll…</li><li><strong>Support for the self-employed</strong>: The furlough scheme also included a separate system for self-employed individuals…</li><li><strong>Protection for vulnerable workers</strong>: The scheme was also designed to protect vulnerable workers…</li><li><strong>Fairness and equity</strong>: The government emphasized the need for fairness and equity…</li><li><strong>Flexibility and adaptability</strong>: The scheme was designed to be flexible and adaptable…</li></ul> | <br><br><ul><li><strong>Protecting jobs and businesses</strong>: The furlough scheme was designed to help employers and employees weather the economic storm by keeping staff on the payroll and maintaining business continuity.</li><li><strong>Supporting workers’ incomes</strong>: Employees continued to receive 80% of their salary (up to £2,500 per month).</li><li><strong>Preventing a jobs crisis</strong>: Seen as a key measure to avert a spike in unemployment.</li><li><strong>Supporting the most vulnerable</strong>: Targeted assistance for those most at risk.</li><li><strong>Encouraging business investment</strong>: Framed as a way to sustain confidence and future investment by businesses.</li></ul> |
