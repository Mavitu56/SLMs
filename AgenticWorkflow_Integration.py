# AgenticWorkflow Integration with SLM_Final Notebook
# This file contains the agentic AI extensions that integrate with existing classes

import torch
import numpy as np
from tqdm import tqdm
import re

class AgenticAIFramework:
    """
    Framework Agentic AI avançado para SLMs - Versão Integrada
    Baseado em: https://arxiv.org/abs/2505.10468
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.agent_registry = {}
        self.conversation_memory = []
        self.tool_registry = {}
        
    def register_agent(self, agent_name, agent_config):
        """Registra um agente no framework"""
        self.agent_registry[agent_name] = {
            'role': agent_config.get('role', 'general'),
            'specialization': agent_config.get('specialization', []),
            'prompt_template': agent_config.get('prompt_template', ''),
            'max_iterations': agent_config.get('max_iterations', 3),
            'confidence_threshold': agent_config.get('confidence_threshold', 0.7)
        }
        
    def register_tool(self, tool_name, tool_func, tool_description):
        """Registra uma ferramenta que os agentes podem usar"""
        self.tool_registry[tool_name] = {
            'function': tool_func,
            'description': tool_description
        }

class AdvancedAgenticPromptEngineering:
    """
    Extensão das capacidades de engenharia de prompt com Agentic AI
    Projetado para integrar com as classes existentes do SLM_Final
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.agentic_framework = AgenticAIFramework(config, logger)
        self._setup_agents()
        self._setup_tools()
    
    def _setup_agents(self):
        """Configura agentes especializados baseados no paper Agentic AI"""
        
        # Agente Planejador
        self.agentic_framework.register_agent('planner', {
            'role': 'task_decomposition',
            'specialization': ['planning', 'reasoning', 'task_breakdown'],
            'prompt_template': '''Você é um agente planejador especializado em decomposição de tarefas.

Contexto: {context}
Pergunta: {question}

Crie um plano estruturado para responder esta pergunta:
1. Identifique as informações-chave necessárias
2. Determine a estratégia de busca no contexto
3. Identifique potenciais armadilhas ou ambiguidades
4. Planeje a estrutura da resposta
5. Defina critérios de validação

Plano:''',
            'max_iterations': 1,
            'confidence_threshold': 0.8
        })
        
        # Agente Pesquisador
        self.agentic_framework.register_agent('researcher', {
            'role': 'information_extraction',
            'specialization': ['search', 'extraction', 'fact_finding'],
            'prompt_template': '''Você é um agente pesquisador especializado em extração de informações.

Plano recebido: {plan}
Contexto para pesquisa: {context}
Pergunta específica: {question}

Extraia TODAS as informações relevantes do contexto:
- Fatos diretos relacionados à pergunta
- Informações que podem ser inferidas
- Relações entre diferentes partes do texto
- Possíveis respostas alternativas

FATOS DIRETOS: [lista]
INFERÊNCIAS: [lista]
RELAÇÕES: [lista]
CANDIDATOS À RESPOSTA: [lista]''',
            'max_iterations': 2,
            'confidence_threshold': 0.75
        })
        
        # Agente Sintetizador
        self.agentic_framework.register_agent('synthesizer', {
            'role': 'synthesis_reasoning',
            'specialization': ['synthesis', 'reasoning', 'answer_generation'],
            'prompt_template': '''Você é um agente sintetizador especializado em raciocínio e síntese.

Plano original: {plan}
Informações extraídas: {research_results}
Pergunta: {question}

Sintetize as informações para gerar uma resposta precisa:
1. Avalie a relevância de cada informação extraída
2. Identifique a resposta mais provável
3. Considere informações conflitantes
4. Verifique consistência com o contexto

RACIOCÍNIO: [explique o processo]
RESPOSTA FINAL: [resposta concisa]''',
            'max_iterations': 1,
            'confidence_threshold': 0.8
        })
        
        # Agente Verificador
        self.agentic_framework.register_agent('verifier', {
            'role': 'quality_control',
            'specialization': ['verification', 'validation', 'quality_check'],
            'prompt_template': '''Você é um agente verificador especializado em controle de qualidade.

Pergunta original: {question}
Contexto: {context}
Resposta proposta: {proposed_answer}

Verifique a qualidade da resposta:
1. PRECISÃO: A resposta está correta com base no contexto?
2. COMPLETUDE: A resposta aborda totalmente a pergunta?
3. CONSISTÊNCIA: A resposta é internamente consistente?
4. RELEVÂNCIA: A resposta é diretamente relevante à pergunta?

PRECISÃO: [1-5] - [justificativa]
COMPLETUDE: [1-5] - [justificativa]  
CONSISTÊNCIA: [1-5] - [justificativa]
RELEVÂNCIA: [1-5] - [justificativa]

NOTA GERAL: [média]
RECOMENDAÇÃO: [aceitar/revisar/rejeitar]''',
            'max_iterations': 1,
            'confidence_threshold': 0.7
        })
    
    def _setup_tools(self):
        """Configura ferramentas que os agentes podem usar"""
        
        # Ferramenta de análise de similaridade semântica (com fallback)
        def semantic_similarity_tool(text1, text2):
            """Calcula similaridade semântica entre dois textos"""
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return f"Similaridade semântica: {similarity:.3f}"
            except ImportError:
                # Fallback: simple word overlap similarity
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0.0
                return f"Similaridade semântica (fallback): {similarity:.3f}"
            except Exception:
                return "Ferramenta de similaridade não disponível"
        
        self.agentic_framework.register_tool(
            'semantic_similarity',
            semantic_similarity_tool,
            'Calcula similaridade semântica entre dois textos'
        )
        
        # Ferramenta de extração de entidades nomeadas
        def named_entity_extraction_tool(text):
            """Extrai possíveis entidades nomeadas do texto"""
            patterns = {
                'DATAS': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text),
                'NÚMEROS': re.findall(r'\b\d+(?:\.\d+)?\b', text),
                'NOMES_PRÓPRIOS': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text),
                'LOCAIS': re.findall(r'\b(?:em|de|para|até)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
            }
            
            result = "ENTIDADES ENCONTRADAS:\n"
            for categoria, entidades in patterns.items():
                if entidades:
                    result += f"{categoria}: {list(set(entidades))}\n"
            
            return result
        
        self.agentic_framework.register_tool(
            'named_entity_extraction',
            named_entity_extraction_tool,
            'Extrai entidades nomeadas do texto (datas, números, nomes, locais)'
        )

# EXTENSÕES PARA AS CLASSES EXISTENTES DO SLM_FINAL

def add_agentic_methods_to_prompt_engineer():
    """
    Adiciona métodos agenticos à classe PromptEngineer existente
    Use esta função após importar as classes do SLM_Final
    """
    
    def agentic_multiagent_evaluation(self, model, tokenizer, eval_dataset, metrics_calc):
        """
        Avaliação usando framework multiagente avançado
        Integrado com a classe PromptEngineer existente
        """
        print("🤖 Avaliação Multiagente Agentic")
        
        # Inicializar framework agentic se não existir
        if not hasattr(self, 'agentic_framework'):
            self.agentic_prompt_engineer = AdvancedAgenticPromptEngineering(self.config, self.logger)
        
        model.eval()
        predictions = []
        references = []
        detailed_logs = []
        
        # Limitar para subset para eficiência
        eval_subset = eval_dataset.select(range(min(20, len(eval_dataset))))
        
        for idx, example in enumerate(tqdm(eval_subset, desc="Agentic Multi-Agent")):
            try:
                example_log = {
                    'example_id': example['id'],
                    'question': example['question'],
                    'agents_outputs': {},
                    'final_confidence': 0.0
                }
                
                # === FASE 1: PLANEJAMENTO ===
                planner_output = self._run_agent_with_tools(
                    model, tokenizer, 'planner', {
                        'context': example['context'][:1000],
                        'question': example['question']
                    }, example_log
                )
                
                # === FASE 2: PESQUISA ===
                research_output = self._run_agent_with_tools(
                    model, tokenizer, 'researcher', {
                        'plan': planner_output,
                        'context': example['context'],
                        'question': example['question']
                    }, example_log
                )
                
                # === FASE 3: SÍNTESE ===
                synthesis_output = self._run_agent_with_tools(
                    model, tokenizer, 'synthesizer', {
                        'plan': planner_output,
                        'research_results': research_output,
                        'question': example['question']
                    }, example_log
                )
                
                final_answer = self._extract_final_answer_from_synthesis(synthesis_output)
                
                # === FASE 4: VERIFICAÇÃO ===
                verification_output = self._run_agent_with_tools(
                    model, tokenizer, 'verifier', {
                        'question': example['question'],
                        'context': example['context'][:500],
                        'proposed_answer': final_answer
                    }, example_log
                )
                
                # Avaliar aceitação
                if 'aceitar' in verification_output.lower():
                    verified_answer = final_answer
                    example_log['verification_status'] = 'aceita'
                elif 'revisar' in verification_output.lower():
                    verified_answer = final_answer
                    example_log['verification_status'] = 'revisada'
                else:
                    verified_answer = ""
                    example_log['verification_status'] = 'rejeitada'
                
                confidence = self._extract_confidence_from_verification(verification_output)
                example_log['final_confidence'] = confidence
                
                predictions.append({
                    'id': example['id'],
                    'prediction_text': verified_answer
                })
                
                references.append({
                    'id': example['id'],
                    'answers': example['answers']
                })
                
                detailed_logs.append(example_log)
                
            except Exception as e:
                print(f"Erro no exemplo {idx}: {e}")
                predictions.append({'id': example['id'], 'prediction_text': ""})
                references.append({'id': example['id'], 'answers': example['answers']})
        
        # Calcular métricas
        results = metrics_calc.compute(predictions=predictions, references=references)
        
        # Adicionar métricas agenticas
        valid_logs = [log for log in detailed_logs if 'verification_status' in log]
        if valid_logs:
            results.update({
                'avg_confidence': np.mean([log['final_confidence'] for log in valid_logs]),
                'acceptance_rate': len([log for log in valid_logs if log.get('verification_status') == 'aceita']) / len(valid_logs),
                'rejection_rate': len([log for log in valid_logs if log.get('verification_status') == 'rejeitada']) / len(valid_logs)
            })
        
        print(f"✅ F1: {results['f1']:.2f}, EM: {results['exact_match']:.2f}")
        print(f"🎯 Confiança média: {results.get('avg_confidence', 0):.2f}")
        print(f"✓ Taxa de aceitação: {results.get('acceptance_rate', 0)*100:.1f}%")
        
        return results
    
    def _run_agent_with_tools(self, model, tokenizer, agent_name, inputs, example_log):
        """Executa um agente com acesso a ferramentas"""
        if not hasattr(self, 'agentic_prompt_engineer'):
            self.agentic_prompt_engineer = AdvancedAgenticPromptEngineering(self.config, self.logger)
        
        agent_config = self.agentic_prompt_engineer.agentic_framework.agent_registry[agent_name]
        prompt_template = agent_config['prompt_template']
        
        # Construir prompt
        prompt = prompt_template.format(**inputs)
        
        try:
            inputs_tensor = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.MAX_LENGTH * 2,
                truncation=True
            ).to(self.config.DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs_tensor,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            agent_output = response[len(prompt):].strip()
            
            example_log['agents_outputs'][agent_name] = agent_output
            return agent_output
            
        except Exception as e:
            error_msg = f"Erro no agente {agent_name}: {e}"
            example_log['agents_outputs'][agent_name] = error_msg
            return error_msg
    
    def _extract_final_answer_from_synthesis(self, synthesis_output):
        """Extrai resposta final do output do sintetizador"""
        patterns = [
            r'RESPOSTA FINAL:\s*(.+?)(?:\n|$)',
            r'resposta final[:\s]*(.+?)(?:\n|$)',
            r'RESPOSTA:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, synthesis_output, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: última linha não vazia
        lines = [line.strip() for line in synthesis_output.split('\n') if line.strip()]
        return lines[-1] if lines else ""
    
    def _extract_confidence_from_verification(self, verification_output):
        """Extrai nível de confiança do verificador"""
        nota_pattern = r'NOTA GERAL:\s*(\d+(?:\.\d+)?)'
        match = re.search(nota_pattern, verification_output, re.IGNORECASE)
        
        if match:
            nota = float(match.group(1))
            return nota / 5.0  # Normalizar para 0-1
        
        # Fallback baseado em palavras-chave
        if 'aceitar' in verification_output.lower():
            return 0.8
        elif 'revisar' in verification_output.lower():
            return 0.6
        elif 'rejeitar' in verification_output.lower():
            return 0.3
        
        return 0.5
    
    def self_reflection_evaluation(self, model, tokenizer, eval_dataset, metrics_calc):
        """
        Implementa auto-reflexão baseada no framework Agentic AI
        """
        print("🧠 Avaliação com Auto-Reflexão")
        
        model.eval()
        predictions = []
        references = []
        reflection_scores = []
        
        # Subset menor para eficiência
        eval_subset = eval_dataset.select(range(min(15, len(eval_dataset))))
        
        for example in tqdm(eval_subset, desc="Self-Reflection"):
            try:
                # === FASE 1: RESPOSTA INICIAL ===
                initial_prompt = f"""Contexto: {example['context']}
Pergunta: {example['question']}

Forneça uma resposta inicial:"""

                inputs = tokenizer(initial_prompt, return_tensors="pt", 
                                 max_length=self.config.MAX_LENGTH, truncation=True).to(self.config.DEVICE)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.7, 
                                           do_sample=True, pad_token_id=tokenizer.eos_token_id)
                
                initial_response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(initial_prompt):].strip()
                
                # === FASE 2: AUTO-REFLEXÃO ===
                reflection_prompt = f"""Contexto: {example['context']}
Pergunta: {example['question']}
Minha resposta inicial: {initial_response}

Reflexão crítica sobre minha resposta:
1. ANÁLISE: Minha resposta está correta e completa?
2. VERIFICAÇÃO: Baseei-me adequadamente no contexto?
3. MELHORIAS: Como posso melhorar?

Reflexão:"""

                reflection_inputs = tokenizer(reflection_prompt, return_tensors="pt",
                                           max_length=self.config.MAX_LENGTH*2, truncation=True).to(self.config.DEVICE)
                
                with torch.no_grad():
                    reflection_outputs = model.generate(**reflection_inputs, max_new_tokens=100,
                                                      temperature=0.5, do_sample=True, 
                                                      pad_token_id=tokenizer.eos_token_id)
                
                reflection = tokenizer.decode(reflection_outputs[0], skip_special_tokens=True)[len(reflection_prompt):].strip()
                
                # === FASE 3: RESPOSTA REFINADA ===
                refinement_prompt = f"""Contexto: {example['context']}
Pergunta: {example['question']}
Resposta inicial: {initial_response}
Minha reflexão: {reflection}

Com base na reflexão, forneço uma resposta final melhorada:"""

                refinement_inputs = tokenizer(refinement_prompt, return_tensors="pt",
                                            max_length=self.config.MAX_LENGTH*2, truncation=True).to(self.config.DEVICE)
                
                with torch.no_grad():
                    final_outputs = model.generate(**refinement_inputs, max_new_tokens=80,
                                                  temperature=0.3, do_sample=True,
                                                  pad_token_id=tokenizer.eos_token_id)
                
                final_response = tokenizer.decode(final_outputs[0], skip_special_tokens=True)[len(refinement_prompt):].strip()
                
                # Avaliar qualidade da reflexão
                reflection_score = self._evaluate_reflection_quality(reflection)
                reflection_scores.append(reflection_score)
                
                predictions.append({'id': example['id'], 'prediction_text': final_response})
                references.append({'id': example['id'], 'answers': example['answers']})
                
            except Exception as e:
                print(f"Erro na auto-reflexão: {e}")
                predictions.append({'id': example['id'], 'prediction_text': ""})
                references.append({'id': example['id'], 'answers': example['answers']})
                reflection_scores.append(0.0)
        
        # Calcular métricas
        results = metrics_calc.compute(predictions=predictions, references=references)
        results.update({
            'avg_reflection_quality': np.mean(reflection_scores),
            'std_reflection_quality': np.std(reflection_scores)
        })
        
        print(f"✅ F1: {results['f1']:.2f}, EM: {results['exact_match']:.2f}")
        print(f"🧠 Qualidade da reflexão: {results['avg_reflection_quality']:.2f}")
        
        return results
    
    def _evaluate_reflection_quality(self, reflection_text):
        """Avalia a qualidade da auto-reflexão"""
        score = 0.0
        reflection_lower = reflection_text.lower()
        
        # Presença de análise crítica
        if any(word in reflection_lower for word in ['análise', 'crítica', 'correto', 'incorreto']):
            score += 0.25
        
        # Verificação de fontes
        if any(word in reflection_lower for word in ['contexto', 'baseado', 'informação', 'fonte']):
            score += 0.25
        
        # Identificação de melhorias
        if any(word in reflection_lower for word in ['melhorar', 'melhoria', 'refinamento', 'ajustar']):
            score += 0.25
        
        # Estrutura organizada
        if any(pattern in reflection_lower for pattern in ['1.', '2.', '3.', 'primeiro', 'segundo']):
            score += 0.25
        
        return score
    
    # Retornar os métodos para serem adicionados à classe
    return {
        'agentic_multiagent_evaluation': agentic_multiagent_evaluation,
        '_run_agent_with_tools': _run_agent_with_tools,
        '_extract_final_answer_from_synthesis': _extract_final_answer_from_synthesis,
        '_extract_confidence_from_verification': _extract_confidence_from_verification,
        'self_reflection_evaluation': self_reflection_evaluation,
        '_evaluate_reflection_quality': _evaluate_reflection_quality
    }

def add_agentic_methods_to_evaluator():
    """
    Adiciona métodos agenticos à classe ScientificEvaluator existente
    """
    
    def comprehensive_evaluation_with_agentic(self, models_dict, datasets):
        """
        Avaliação científica incluindo técnicas Agentic AI
        """
        print("🤖 Iniciando Avaliação Científica com Agentic AI...")
        
        # Primeiro, executar avaliação padrão se disponível
        if hasattr(self, 'comprehensive_evaluation'):
            results_matrix, detailed_results = self.comprehensive_evaluation(models_dict, datasets)
        else:
            results_matrix, detailed_results = {}, {}
        
        # Adicionar avaliações Agentic AI
        agentic_strategies = {
            'Agentic Multi-Agent': 'agentic_multiagent_evaluation',
            'Agentic Self-Reflection': 'self_reflection_evaluation'
        }
        
        for model_name, (model, tokenizer) in models_dict.items():
            print(f"\n🤖 Avaliações Agentic para modelo: {model_name}")
            model.to(self.config.DEVICE)
            
            # Criar instância do prompt engineer agentico
            if not hasattr(self, 'agentic_prompt_engineer'):
                # Assumir que PromptEngineer já foi estendido com métodos agenticos
                self.agentic_prompt_engineer = type('MockPromptEngineer', (), {})()
                self.agentic_prompt_engineer.config = self.config
                self.agentic_prompt_engineer.logger = self.logger
                
                # Adicionar métodos agenticos
                agentic_methods = add_agentic_methods_to_prompt_engineer()
                for method_name, method_func in agentic_methods.items():
                    setattr(self.agentic_prompt_engineer, method_name, 
                           method_func.__get__(self.agentic_prompt_engineer))
            
            for strategy_name, method_name in agentic_strategies.items():
                print(f"  🔬 Estratégia: {strategy_name}")
                
                try:
                    eval_method = getattr(self.agentic_prompt_engineer, method_name)
                    results = eval_method(model, tokenizer, datasets['evaluation'], datasets['metrics_calculator'])
                    
                    # Adicionar aos resultados existentes
                    if model_name not in results_matrix:
                        results_matrix[model_name] = {}
                    if model_name not in detailed_results:
                        detailed_results[model_name] = {}
                    
                    results_matrix[model_name][strategy_name] = {
                        'f1': results['f1'],
                        'exact_match': results['exact_match']
                    }
                    detailed_results[model_name][strategy_name] = results
                    
                    print(f"    ✅ F1: {results['f1']:.2f}, EM: {results['exact_match']:.2f}")
                    
                    # Log métricas específicas
                    if 'avg_confidence' in results:
                        print(f"    🎯 Confiança: {results['avg_confidence']:.2f}")
                    if 'acceptance_rate' in results:
                        print(f"    ✓ Taxa de aceitação: {results['acceptance_rate']*100:.1f}%")
                    if 'avg_reflection_quality' in results:
                        print(f"    🧠 Qualidade reflexão: {results['avg_reflection_quality']:.2f}")
                    
                except Exception as e:
                    print(f"    ❌ Erro: {e}")
                    results_matrix[model_name][strategy_name] = {'f1': 0.0, 'exact_match': 0.0}
                    detailed_results[model_name][strategy_name] = {'error': str(e)}
        
        return results_matrix, detailed_results
    
    return {
        'comprehensive_evaluation_with_agentic': comprehensive_evaluation_with_agentic
    }

# FUNÇÃO DE INTEGRAÇÃO PRINCIPAL
def integrate_agentic_workflow():
    """
    Função principal para integrar AgenticWorkflow com SLM_Final
    
    Como usar:
    1. Execute todas as células do SLM_Final notebook
    2. Execute: exec(open('AgenticWorkflow_Integration.py').read())
    3. Execute: integrate_agentic_workflow()
    4. As classes PromptEngineer e ScientificEvaluator terão novos métodos agenticos
    """
    
    print("🚀 INTEGRANDO AGENTIC WORKFLOW COM SLM_FINAL")
    print("="*60)
    
    try:
        # Verificar se as classes base existem
        if 'PromptEngineer' in globals():
            print("✅ PromptEngineer encontrado - adicionando métodos agenticos...")
            
            # Adicionar métodos agenticos à classe PromptEngineer
            agentic_methods = add_agentic_methods_to_prompt_engineer()
            for method_name, method_func in agentic_methods.items():
                setattr(PromptEngineer, method_name, method_func)
            
            print("   ✓ agentic_multiagent_evaluation")
            print("   ✓ self_reflection_evaluation")
            print("   ✓ métodos auxiliares")
            
        else:
            print("⚠️  PromptEngineer não encontrado - certifique-se de executar as células do SLM_Final primeiro")
        
        if 'ScientificEvaluator' in globals():
            print("✅ ScientificEvaluator encontrado - adicionando métodos agenticos...")
            
            # Adicionar métodos agenticos à classe ScientificEvaluator
            evaluator_methods = add_agentic_methods_to_evaluator()
            for method_name, method_func in evaluator_methods.items():
                setattr(ScientificEvaluator, method_name, method_func)
            
            print("   ✓ comprehensive_evaluation_with_agentic")
            
        else:
            print("⚠️  ScientificEvaluator não encontrado - certifique-se de executar as células do SLM_Final primeiro")
        
        print("\n🎯 COMO USAR OS NOVOS MÉTODOS:")
        print("# Para avaliação multiagente:")
        print("prompt_engineer = PromptEngineer(config, logger)")
        print("results = prompt_engineer.agentic_multiagent_evaluation(model, tokenizer, eval_dataset, metrics_calc)")
        print("")
        print("# Para auto-reflexão:")
        print("results = prompt_engineer.self_reflection_evaluation(model, tokenizer, eval_dataset, metrics_calc)")
        print("")
        print("# Para avaliação completa com agentic:")
        print("evaluator = ScientificEvaluator(config, logger)")
        print("results_matrix, detailed = evaluator.comprehensive_evaluation_with_agentic(models_dict, datasets)")
        
        print("\n✅ INTEGRAÇÃO CONCLUÍDA COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        return False

# FUNÇÃO DE DEMONSTRAÇÃO RÁPIDA
def quick_agentic_demo():
    """
    Demonstração rápida das capacidades agenticas
    Executa apenas se as classes do SLM_Final estiverem disponíveis
    """
    print("🎯 DEMONSTRAÇÃO RÁPIDA AGENTIC AI")
    print("="*50)
    
    try:
        # Verificar disponibilidade das classes necessárias
        required_classes = ['ScientificExperimentRunner', 'PromptEngineer', 'ScientificEvaluator']
        missing_classes = [cls for cls in required_classes if cls not in globals()]
        
        if missing_classes:
            print(f"❌ Classes necessárias não encontradas: {missing_classes}")
            print("Execute primeiro todas as células do SLM_Final notebook")
            return None
        
        # Integrar métodos agenticos
        integrate_agentic_workflow()
        
        # Setup básico
        runner = ScientificExperimentRunner()
        config = runner.config
        logger = runner.logger
        
        # Configurações reduzidas para demo
        config.EVAL_SUBSET_SIZE = 5
        config.MAX_LENGTH = 256
        
        print("\n📊 Carregando dados para demonstração...")
        datasets, _ = runner.dataset_manager.load_and_analyze_data()
        
        print("🤖 Preparando modelo...")
        models_dict = runner._prepare_evaluation_models({})
        
        if not models_dict:
            print("⚡ Carregando modelo base...")
            base_model, base_tokenizer = runner._load_model_safe(config.STUDENT_MODEL_ID)
            if base_model:
                models_dict['SLM Base'] = (base_model, base_tokenizer)
        
        if not models_dict:
            print("❌ Não foi possível carregar modelo")
            return None
        
        print(f"✅ Modelo carregado: {list(models_dict.keys())}")
        
        # Demonstração com PromptEngineer
        print(f"\n{'='*40}")
        print("🤖 DEMO: MULTIAGENT EVALUATION")
        print(f"{'='*40}")
        
        prompt_engineer = PromptEngineer(config, logger)
        model_name, (model, tokenizer) = next(iter(models_dict.items()))
        
        results = prompt_engineer.agentic_multiagent_evaluation(
            model, tokenizer, 
            datasets['evaluation'].select(range(3)), 
            datasets['metrics_calculator']
        )
        
        print(f"\n📊 RESULTADOS DEMO:")
        print(f"F1-Score: {results['f1']:.3f}")
        print(f"Exact Match: {results['exact_match']:.3f}")
        if 'avg_confidence' in results:
            print(f"Confiança média: {results['avg_confidence']:.3f}")
        if 'acceptance_rate' in results:
            print(f"Taxa de aceitação: {results['acceptance_rate']*100:.1f}%")
        
        print("\n✅ Demonstração concluída!")
        return results
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 AGENTIC WORKFLOW INTEGRATION")
    print("="*40)
    print("Este arquivo integra capacidades Agentic AI com o SLM_Final notebook")
    print("")
    print("COMO USAR:")
    print("1. Execute todas as células do SLM_Final notebook")
    print("2. Execute: exec(open('AgenticWorkflow_Integration.py').read())")
    print("3. Execute: integrate_agentic_workflow()")
    print("4. Execute: quick_agentic_demo()  # para demonstração")