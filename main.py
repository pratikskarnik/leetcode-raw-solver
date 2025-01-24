import os
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, Any, TypedDict, List
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the solution is valid")
    error_message: str = Field(
        description="Detailed error message if solution is invalid"
    )
    improvement_suggestions: List[str] = Field(
        description="Suggestions for improving the solution"
    )


class SolverState(TypedDict):
    problem_description: str
    search_context: List[dict]
    solution_strategy: str
    python_solution: str
    validation_result: bool
    validation_error: str


class LeetCodeSolverBot:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        # Initialize API clients
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.tavily_client = TavilyClient(tavily_api_key)

        # Create output parser for validation
        self.validation_parser = PydanticOutputParser(pydantic_object=ValidationResult)

        # Create LangGraph workflow
        self.graph = self._build_leetcode_solver_graph()

    def _search_problem_context(self, state: SolverState) -> Dict[str, Any]:
        """Use Tavily to search for problem-solving context"""
        search_results = self.tavily_client.search(
            query=state["problem_description"].split(":")[0], max_results=5
        )
        return {"search_context": search_results}

    def _generate_solution_strategy(self, state: SolverState) -> Dict[str, str]:
        """Generate solution strategy using LangChain OpenAI"""
        strategy_prompt = PromptTemplate.from_template(
            """
        Based on the problem description and search context:
        Problem: {problem_description}
        Search Context: {search_context}
        
        Develop a comprehensive solution strategy for the Two Sum problem:
        1. Optimal algorithm selection
        2. Appropriate data structures
        3. Detailed implementation steps
        4. Time and space complexity analysis
        """
        )

        chain = strategy_prompt | self.llm
        response = chain.invoke(
            {
                "problem_description": state["problem_description"],
                "search_context": state["search_context"],
            }
        )
        return {"solution_strategy": response.content}

    def _write_python_solution(self, state: SolverState) -> Dict[str, str]:
        """Convert solution strategy to Python code"""
        code_generation_prompt = PromptTemplate.from_template(
            """
        Create a production-ready Python solution for the Two Sum problem:
        Strategy: {solution_strategy}
        
        Detailed requirements:
        - Clean, Pythonic implementation
        - Include function definition with type hints
        - Handle edge cases
        - Add comprehensive docstring
        - Optimize for time and space complexity
        - Provide a working solution that passes basic test cases
        """
        )

        chain = code_generation_prompt | self.llm
        response = chain.invoke({"solution_strategy": state["solution_strategy"]})
        return {"python_solution": response.content}

    def _validate_solution(self, state: SolverState) -> Dict[str, Any]:
        """LLM-powered validation of generated Python solution"""
        validation_prompt = PromptTemplate.from_template(
            """
        Validate the following Python solution for the Two Sum problem:
        
        Problem Description: {problem_description}
        
        Python Solution:
        {python_solution}
        
        Generate test cases for this problem description: {problem_description}

        Validation Criteria:
        1. Correct algorithm implementation
        2. Handles edge cases
        3. Proper type hints and docstrings
        4. Optimal time and space complexity
        5. Passes all the generated test cases

        Provide a structured validation result including:
        - Validity of the solution
        - Detailed error message if invalid
        - Specific improvement suggestions

        {format_instructions}
        """
        )

        # Create the validation chain
        validation_chain = validation_prompt | self.llm | self.validation_parser

        try:
            # Invoke the validation chain
            validation_result = validation_chain.invoke(
                {
                    "problem_description": state["problem_description"],
                    "python_solution": state["python_solution"],
                    "format_instructions": self.validation_parser.get_format_instructions(),
                }
            )

            return {
                "validation_result": validation_result.is_valid,
                "validation_error": validation_result.error_message,
            }

        except Exception as e:
            return {
                "validation_result": False,
                "validation_error": f"Validation process failed: {str(e)}",
            }

    def _build_leetcode_solver_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(SolverState)

        # Add nodes with method references
        workflow.add_node("search", self._search_problem_context)
        workflow.add_node("strategy", self._generate_solution_strategy)
        workflow.add_node("code_gen", self._write_python_solution)
        workflow.add_node("validate", self._validate_solution)

        # Define workflow edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "strategy")
        workflow.add_edge("strategy", "code_gen")
        workflow.add_edge("code_gen", "validate")

        # Conditional routing based on validation
        def route_validation(state: SolverState):
            return END if state["validation_result"] else "strategy"

        workflow.add_conditional_edges("validate", route_validation)

        return workflow.compile()

    def solve_leetcode_problem(self, problem_description: str):
        """Main workflow for solving LeetCode problem"""
        initial_state = {"problem_description": problem_description}
        final_state = self.graph.invoke(initial_state)

        if final_state.get("validation_result", False):
            return final_state.get("python_solution", "Solution generation failed")
        else:
            return f"Solution generation failed: {final_state.get('validation_error', 'Unknown error')}"


# Example Usage
def main():
    bot = LeetCodeSolverBot(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )

    problem_description = """
    Merge k Sorted Lists: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    Merge all the linked-lists into one sorted linked-list and return it."""

    solution = bot.solve_leetcode_problem(problem_description)
    print(solution)


if __name__ == "__main__":
    main()
