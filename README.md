# Aprimoramento e Comparação Científica de SLMs

Este repositório contém um workflow completo em Jupyter Notebook para aprimorar e avaliar cientificamente um Small Language Model (SLM). O projeto segue um plano estruturado para aplicar técnicas de **Destilação de Conhecimento (KD)** e **Engenharia de Prompt Avançada**, permitindo uma comparação robusta do impacto de cada técnica, mesmo em ambientes com recursos limitados como o Google Colab.

## 🚀 Principais Características

* **Destilação de Conhecimento**: Implementação de múltiplas técnicas de KD, incluindo Logit Matching e a mais robusta, Destilação de Explicações.
* **Engenharia de Prompt**: Implementação das principais estratégias de prompting, como Zero-Shot, Few-Shot ICL, Chain-of-Thought (CoT) e Auto-Consistência.
* **Pipeline Completo**: Um fluxo de trabalho de ponta a ponta, desde a configuração de dados e treinamento de modelos até a avaliação sistemática e geração de relatórios.
* **Otimizado para Colab**: O código foi ajustado e testado para funcionar dentro das limitações de memória das GPUs gratuitas do Google Colab (ex: T4), com alternativas de modelos mais leves.
* **Análise Científica**: Gera uma matriz de resultados e gráficos para comparar o desempenho (F1-Score / Exact Match) de cada combinação de modelo e técnica de prompt.

## 📂 Estrutura do Projeto

Toda a lógica está contida em um único Jupyter Notebook:

* `SLMs_Workflow_Completo.ipynb`: O notebook principal que contém todas as funções, o pipeline de orquestração e as análises. Ele é dividido em seções que correspondem às fases do plano original.
