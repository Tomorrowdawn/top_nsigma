> [!NOTE]
> Our latest evaluation code is maintained in `/dev` branch. Please check it out for the latest updates. This README is outdated.
> This page is subject to change as our paper enters the review process.

# Top-NSigma Sampling

This is the official repository for the [Top-nσ sampling](https://arxiv.org/pdf/2411.07641) algorithm. The repository aims to provide a working implementation of the algorithm and collect empirical data with help from the community. We encourage you to try it out and share your feedback!

TLDR:

```python
M, sigma=logits.max(keep_dim=True, dim=-1), logits.std(keepdim=True, dim=-1)
logits[logits < M-n*sigma] = float('-inf')
```

---

## Overview

Top-nσ is a novel sampling method for language models that truncates the probability distribution based on standard deviations from the maximum logit value. It exhibits superior performance in terms of quality and diversity compared to existing sampling methods, particularly in high temperature settings. Basically, you don't need to worry about temperature wielding top-nsigma.

### Usage

We provide two versions of the implementation:

1. HuggingFace version, see in `src/hf/hf_nsigma.py`. You can use it out of the box with HuggingFace Transformers.
2. VLLM version, see in `src/vllm/sampler.py`. You need to apply the ugly hack in `src/vllm/hack.py`. Put it simply, here is how you do it:

```python
import vllm
from hack import hack_vllm, recover_sampler
from sampler import FacadeSampler
model = vllm.LLM(model=path)
hack_vllm(model, FacadeSampler(nsigma, device))
```

> [!NOTE] 
> Due to vLLM's current architecture, we have to use this temporary hack.


## Contributing

We strongly welcome contributions from the community! 

A key question is: what's the best value for $n$? While this parameter serves as an alternative to temperature for controlling diversity, its optimal value isn't fully settled yet. The community suggests a range of 0-2, though this is quite broad. In my own experience, any value between 0.3 and 1.5 could work well. If you prefer conservative sampling, use a lower value like 0.7; for more diversity, try 1.3.

### Top-nsigma Usage Tracking

https://github.com/aphrodite-engine/aphrodite-engine/pull/825


https://github.com/SillyTavern/SillyTavern/pull/3094


https://github.com/ggerganov/llama.cpp/pull/11223

## Citation

If you find this work useful, please consider citing:

```
@misc{tang2024topnsigmalogitsneed,
      title={Top-$n\sigma$: Not All Logits Are You Need}, 
      author={Chenxia Tang and Jianchun Liu and Hongli Xu and Liusheng Huang},
      year={2024},
      eprint={2411.07641},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.07641}, 
}
```
