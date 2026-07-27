## Runnables

- A Runnable is any object that implements the `invoke()` interface (contract), meaning it accepts an input, performs some work, and returns an output.

```python
    class MultiplyByTwo:

    def invoke(self, number):
        return number * 2
```

- In the same way LangChain uses a lot of runnables, like `llm.invoke()`

## `|` Pipe Operator and the `__or__` in Python

- `|` is simply the `__or__`
  - Each object / class can have it own "meaning" for the `__or__` method

- The same way we use `+` as the `__add__` operator, and so on

## Operator Overloading

- In Python, we can define the built in methods - which is called operator overloading

## LangChain and LCEL

- **LCEL is syntax for composing multiple Runnable objects into a single Runnable**

- LangChain uses operator overloading, implementing the `__or__` is chaining methods

- Chaining methods stands for getting an input > use it as arg for a method > get the result as the input for a next method and so on

- thats why in Langchain we create _Runnable chains_

- **Important:** Also, LangChain convert regular python functions into Runnables lambdas when passed in a chain

```python

    retrieval_chain = (
        prompt_template  # prompt template
        | llm  # invokes the LLM with the formatted prompt
        | StrOutputParser()  # extract the content from the LLM response
    )

```

### RunnablePassthrough.assign (in our 9_RAG file):

- Purpose:
  - The input of the chain is initially:
    ```python
    {"question": "..."}
    ```
  - However, the `prompt_template` expects:
    ```python
    {
      "question": "...",
      "context": "..."
    }
    ```
  - `RunnablePassthrough.assign()` solves this by preserving the original input and assigning new key-value pairs to it.

- How:
  - `context=` receives another Runnable (or a function, automatically converted into a Runnable)
  - This Runnable is executed using the original input as its argument
  - Its result becomes the value of the `context` key
  - Conceptually, it is similar to:

    ```python
    context = context_chain.invoke(input)

    output = {
        **input,
        "context": context
    }
    ```
