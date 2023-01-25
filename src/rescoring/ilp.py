import pulp
import pandas as pd
import src.registry as registry
from pulp import GLPK


@registry.register("ilp")
class ILPModule:
    # Adapted from reference: 
    # https://github.com/sildar/potara/blob/master/potara/summarizer.py
    def __init__(self, concepts, max_budget, min_budget=1, solver_path=None):
        self.concepts = registry.create(concepts)
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.solver_path = solver_path


    def select(self, initial_results:pd.DataFrame):
        sentences = initial_results["text"].tolist()

        concepts_by_sentence, weights_by_concept = self.concepts.extract(sentences)

        concepts = set(
            _concept
            for _concepts in concepts_by_sentence
            for _concept in _concepts
        )

        n_concepts = len(concepts)
        n_sentences = len(sentences)

        if (n_sentences <= self.min_budget) or \
           (n_concepts == 0):
           return initial_results
        
        ## problem setup
        problem = pulp.LpProblem('sentence selection problem', pulp.LpMaximize)

        ## variables to solve
        concepts_var = pulp.LpVariable.dicts(
            name='c',
            indexs=range(n_concepts),
            lowBound=0,
            upBound=1,
            cat='Integer'
        )

        sentences_var = pulp.LpVariable.dicts(
            name='s',
            indexs=range(n_sentences),
            lowBound=0,
            upBound=1,
            cat='Integer'
        )

        problem += (
            pulp.lpSum(
                [weights_by_concept.get(_concept, 0) * concepts_var[i]
                for i, _concept in enumerate(concepts)]
            ),
            'total score by w_i x c_i'
        )

        ## size constraints
        
        problem += pulp.lpSum(
            [sentences_var[i] * 1
             for i in range(n_sentences)]
        ) <= self.max_budget

        problem += pulp.lpSum(
            [sentences_var[i] * 1
             for i in range(n_sentences)]
        ) >= self.min_budget

        ## consistency constraints

        # if s_i is covered by S, then each concept of s_i is covered by S
        for j, _sentence in enumerate(concepts_by_sentence):
            for i, _concept in enumerate(concepts):
                if _concept in _sentence:
                    problem += sentences_var[j] <= concepts_var[i]

        # if concept i is covered by S, then at least one s_i is covered by S
        for i, _concept in enumerate(concepts):
            problem += pulp.lpSum(
                [sentences_var[j]
                 for j, _concepts in enumerate(concepts_by_sentence)
                 if _concept in _concepts]
            ) >= concepts_var[i]

        ## solve the problem
        problem.solve(GLPK(path=self.solver_path, msg=0))

        ## get the results
        indices = []
        for i in range(n_sentences):
            if sentences_var[i].varValue == 1:
                indices.append(i)
        return initial_results.iloc[indices,:]