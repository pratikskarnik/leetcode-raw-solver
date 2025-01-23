# LeetCode Problem Solver Bot

## Overview
LeetCode Problem Solver Bot is an advanced AI-powered tool designed to solve algorithmic coding problems from LeetCode. It utilizes cutting-edge Language Learning Models (LLMs), contextual search capabilities, and structured workflows to generate optimal solutions with step-by-step validation.

### Key Features:
- **Problem Context Retrieval:** Uses Tavily to search for relevant contextual data.
- **Solution Strategy Generation:** Develops an optimal approach using OpenAI's GPT-4 models.
- **Python Code Generation:** Converts the strategy into clean, production-ready Python code.
- **Solution Validation:** Employs LLM-based validation to ensure correctness, efficiency, and adherence to coding standards.
- **Iterative Workflow:** Automatically refines the solution if validation fails, ensuring robustness.

## Prerequisites
- Python 3.12+
- OpenAI API Key
- Tavily API Key

## Setup

### 1. API Keys Acquisition

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account
3. Generate API key in API Keys section

#### Tavily API Key
1. Go to [Tavily AI](https://tavily.com/)
2. Sign up 
3. Generate API key

#### LangSmith API Key (Optional)
1. Visit LangSmith [Langsmith Platform](https://www.langchain.com/langsmith/)
2. Sign up or log in
3. Go to API Keys section
4. Generate a new key

### 2. Environment Configuration

Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LEETCODE
PYTHONPATH=.
```

### 3. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Problem Description Format

Define a problem description in the following format in `main()`:

```python
problem_description = """
    Merge k Sorted Lists: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    Merge all the linked-lists into one sorted linked-list and return it."""
```

### 5. Execution

```bash
python main.py
```

### Optional: LangSmith Integration
If enabled, the LangSmith platform provides detailed tracing and debugging for the LLM-based workflows, including intermediate outputs and validation feedback.

## Workflow Architecture

The bot's architecture is built around a structured LangGraph workflow with the following key steps:

1. **Search for Problem Context**  
   Queries Tavily to gather relevant resources and insights based on the problem description.

2. **Generate Solution Strategy**  
   Uses OpenAI's GPT-4 models to develop a comprehensive solution plan, which includes:
   - Algorithm selection
   - Data structure choices
   - Implementation steps
   - Time and space complexity analysis

3. **Code Generation**  
   Converts the solution strategy into clean, production-ready Python code. The generated code includes:
   - Proper type hints
   - Comprehensive docstrings
   - Handling of edge cases
   - Optimized implementation

4. **Validate Solution**  
   Employs an LLM-powered validation process to:
   - Verify correctness against predefined criteria
   - Test handling of edge cases
   - Ensure adherence to coding standards (e.g., type hints, docstrings)
   - Assess time and space complexity

5. **Iterative Refinement**  
   If the validation fails, the workflow refines the solution by revisiting earlier steps, ensuring a robust final output.

### Workflow Diagram
The workflow progresses as follows:
- **Entry Point:** `search` (Problem Context Retrieval)
- **Flow:** `search → strategy → code_gen → validate`
- **Conditional Routing:**  
  - If validation passes, the workflow ends.  
  - If validation fails, it loops back to the `strategy` step for refinement.

This iterative process ensures that the final solution is accurate, efficient, and production-ready.


## Troubleshooting
- Ensure your API keys are correctly configured in the .env file.
- Verify the internet connection for accessing external APIs.
- Check Python version compatibility (Python 3.12+ is required).
- Reinstall dependencies using pip install -r requirements.txt.