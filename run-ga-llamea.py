from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import GA_LLaMEA_Method
from iohblade.loggers import ExperimentLogger
import numpy as np
import os
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.0-flash"
    llm1 = Gemini_LLM(api_key, ai_model)
    llm2 = Ollama_LLM("codestral")
    budget = 100

    for llm in [llm1]:#, llm2]:
        # GA-LLAMEA uses Discounted Thompson Sampling to adaptively select operators
        # Instead of fixed mutation prompts, it learns which operator works best:
        # - mutation: refine single parent
        # - crossover: combine two parents  
        # - random_new: generate from scratch
        GA_LLaMEA_method1 = GA_LLaMEA_Method(llm, budget=budget, name="GA-LLAMEA-1", n_parents=4, n_offspring=8, elitism=True, discount=0.9, tau_max=0.2)

        methods = [GA_LLaMEA_method1]
        logger = ExperimentLogger("results/GA-LLAMEA")
        experiment = MA_BBOB_Experiment(methods=methods, runs=2, seeds=[4,7], dims=[5], budget_factor=2000, budget=100, eval_timeout=300, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, runs=5, dims=[2], budget_factor=1000) #quick run


