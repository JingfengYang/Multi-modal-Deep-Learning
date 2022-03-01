# Multi-modal-Deep-Learning

Recent Multi-modal Deep Learning Advances (list of papers and highlights).

----
## Introduction 

### Prelude

There are many advances of using unified models (e.g. Transformer) to create representations for multiple modalities. Some of them even enable fusion of multiple modalities to make different modalities help each other. Here, multiple modalities not only include natural language, vision and speech, but also include formal language (e.g. code) and (semi-)structured knowledge (e.g. table, KG etc.). This is a list of recent important papers in this field. Welcome to contribute.


- [Introduction](#introduction)
  - [Prelude](#prelude)
- [Resources](#resources)
- [Natural Language](#natural-language)
- [Vision](#vision)
  - [Supervised Vision Tasks](#supervised-vision-tasks)
  - [Unsupervised Vision Representation Learning](#unsupervised-vision-representation-learning)
- [Speech](#speech)
  - [Unsupervised Speech Representation Learning](#unsupervised-speech-representation-learning)
  - [Unsupervised Automatic Speech Recognition (ASR)](#unsupervised-automatic-speech-recognition)
- [Formal Language / Code](#formal-language)
- [Structured Knowledge](#structured-knowledge)
  - [Table](#table)
  - [Knowledge Graph](#knowledge-graph)
  - [Retrieval Paragraphs as Knowledge](#retrieval-paragraphs-as-knowledge)
- [Modality Fusion](#modality-fusion)
  - [Vision and Natural Language](#vision-and-natural-language)


## Resources
* [Microsoft UniLM series](https://github.com/microsoft/unilm/)

## Natural Language

* BERT, RoBERTa, BART, SpanBERT, UniLM, PEGASUS, ELECTRA, T5, GPT-k, FLAN, InstructGPT etc.

* [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555.pdf), arxiv Feb 2022.

## Vision

### Supervised Vision Tasks

* [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020.

* [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf), ICLR 2021.

* [DeiT: Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf), arxiv Dec 2020.

* [MoCo-V3: An Empirical Study of Training Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf), ICCV 2021.

* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf), arxiv Aug 2021.

### Unsupervised Vision Representation Learning

* [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf), arxiv April 2021.

* [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254), arxiv Jun 2021

* [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/pdf/2111.09886.pdf), arxiv Nov 2021.

* [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf), arxiv Nov 2021.

* [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555.pdf), arxiv Feb 2022.

## Speech

### Unsupervised Speech Representation Learning

* [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf), arxiv Jun 2020.

* [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/pdf/2106.07447.pdf), arxiv Jun 2021.

* [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/pdf/2110.13900.pdf), arxiv Oct 2021.

* [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555.pdf), arxiv Feb 2022.

### Unsupervised Automatic Speech Recognition

* [wav2vec-U: Unsupervised Speech Recognition](https://arxiv.org/pdf/2105.11084.pdf), arxiv May 2021.

## Formal Language

* [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf), EMNLP 2020 (Findings).

* [Codex: Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf), arxiv Jul 2021.

* [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/pdf/2009.08366.pdf), ICLR 2021.

* [AlphaCode: Competition-Level Code Generation with AlphaCode](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf).

## Structured Knowledge

* [UNIFIEDSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models](https://arxiv.org/abs/2201.05966), arxiv Jan 2022.

### Table

* [TABERT: Pretraining for Joint Understanding of Textual and Tabular Data](https://arxiv.org/pdf/2005.08314.pdf), ACL 2020.

* [GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing](https://arxiv.org/pdf/2009.13845.pdf), ICLR 2021.

* [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/pdf/2004.02349.pdf), ACL 2020.

* [STRUG: Structure-Grounded Pretraining for Text-to-SQL](https://arxiv.org/pdf/2010.12773.pdf), NAACL 2021.

* [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/pdf/2107.07653.pdf), ICLR 2022.

* [TableFormer: Robust Transformer Modeling for Table-Text Encoding](https://openreview.net/pdf?id=EHzvRqy6kD), ACL 2022.

### Knowledge Graph

* [COMET: Commonsense Transformers for Automatic Knowledge Graph Construction](https://arxiv.org/pdf/1906.05317.pdf), ACL 2019.

* [(COMET-)ATOMIC-2020: On Symbolic and Neural Commonsense Knowledge Graphs](https://arxiv.org/pdf/2010.05953.pdf), arxiv Oct 2020.

* [Knowledge is Power: Symbolic Knowledge Distillation, Commonsense Morality, & Multimodal Script Knowledge](https://dl.acm.org/doi/abs/10.1145/3488560.3500242), WSDM 2022.

### Retrieval Paragraphs as Knowledge

* [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/pdf/2002.08909.pdf), arxiv Feb 2020.

* [MERGE: Pre-training via Paraphrasing](https://proceedings.neurips.cc/paper/2020/file/d6f1dd034aabde7657e6680444ceff62-Paper.pdf), NeuralPS 2020.

* [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550.pdf), EMNLP 2020.

* [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeuralPS 2020.

* [End-to-End Training of Neural Retrievers for Open-Domain Question Answering](https://arxiv.org/pdf/2101.00408.pdf), ACL 2021.

* [Condenser: a Pre-training Architecture for Dense Retrieval](https://arxiv.org/pdf/2104.08253.pdf), EMNLP 2021.

* [Spider: Learning to Retrieve Passages without Supervision](http://www.cs.tau.ac.il/~oriram/spider.pdf), arxiv Dec 2021.

## Modality Fusion

### Vision and Natural Language

* [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://papers.nips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf), NeuralPS 2019.

* [LXMERT: Learning Cross-Modality Encoder Representations](https://arxiv.org/pdf/1908.07490.pdf), EMNLP 2019.

* [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557.pdf), ACL 2020.

* [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/pdf/1908.06066.pdf), arxiv Dec 2019.

* [UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/pdf/1909.11740.pdf), arxiv July 2020.

* [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf), ECCV 2020.

* [VILLA: Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/pdf/2006.06195.pdf), NeuralPS 2020.

* [ViLBERT-MT: 12-in-1: Multi-Task Vision and Language Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf), CVPR 2020.

* [Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/pdf/2004.00849.pdf), arxiv April 2020.

* [U-VisualBERT: Unsupervised Vision-and-Language Pre-training Without Parallel Images and Captions](https://arxiv.org/pdf/2010.12831.pdf), NAACL 2021.

* [M6: A Chinese Multimodal Pretrainer](https://arxiv.org/pdf/2103.00823.pdf), arxiv March 2021.

* [DALL·E: Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092.pdf), arxiv Feb 2021.

* [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf), arxiv Feb 2021.

* [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/pdf/2012.15409.pdf), ACL 2021.

* [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334.pdf), ICML 2021.

* [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/pdf/2108.10904.pdf), arxiv Aug 2021.

* [ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/pdf/2107.07651.pdf), arxiv July 2021.

* [VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/pdf/2101.00529.pdf), CVPR 2021.

* [LAFITE: Towards Language-Free Training for Text-to-Image Generation](https://arxiv.org/pdf/2111.13792.pdf), arxiv Nov 2021.

* [VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/pdf/2111.02358.pdf), arxiv Nov 2021.

* [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/pdf/2112.10741.pdf) arxiv Dec 2021.

* [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/pdf/2112.04482.pdf), arxiv Dec 2021.

* [Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks](https://arxiv.org/pdf/2112.01522.pdf), arxiv Dec 2021.

* [CM3: A Causal Masked Multimodal Model of the Internet](https://arxiv.org/pdf/2201.07520.pdf), arxiv Jan 2022.

* [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/pdf/2202.03052.pdf), arxiv Feb 2022.

