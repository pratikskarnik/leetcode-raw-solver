# LeetCode Problem Solver Bot

## Overview
AI-powered bot solving LeetCode problems using advanced LLM techniques.

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

Use this format in `main()`:
```python
problem_description = """
    Problem Name: Detailed problem description 
    Provide comprehensive problem statement."""
```

#### Example
```python
problem_description = """
    Merge k Sorted Lists: You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    Merge all the linked-lists into one sorted linked-list and return it."""
```

### 5. Execution

```bash
python main.py
```

### 6. Langsmith output (Optional)
You can see your LLM traces in Langsmith and copy the output python code from last validate component

## Features
- AI-powered problem solving
- Adaptive solution generation
- LLM-based validation
- Context retrieval with Tavily

## Troubleshooting
- Verify API keys
- Check internet connection
- Ensure Python dependencies are installed