# NoiseInjection

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

