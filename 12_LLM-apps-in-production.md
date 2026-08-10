## 12. LLM Applications in Production

- Runtime
  - We iterate with LLM calls, so it might affect runtime

- Context Window Limitation

- Hallucinations
  - Since we are iterating many calls, each call has a small probability of bringing a bad answer - which is exponentially increases with the iteration

- Fine Tuning
  - One of the ways of solving hallucinations

- Pricing
  - Semantic tools - As I understood, we can add something that looks close to a RAG to define the tools, so everything is more precise

- Response Validation
  - Proper format on the output - hard to define robust solutions for large scales app

- Security
  - Specially Prompt Injection
  - Eden Recommends LLM Guards

- "Over killing"
  - Avoid using agents for tasks that can be done with deterministic code

### LLM Applications Landscape

- Simple LLM Calls
  - Eg, a pre structured prompt template that creates children stories about inserted topics

- RAGs / Vector stores
  - To super specific domain related questions

- Agents
  - Leverage the LLM with programatic tools

- Agents + Vector stores

### Privacy and Data Retention

- Some things to be aware of:
  - Data being used for training, How long the data is retained, purposes, copyrights etc

- About data and privacy, each vendor has its own terms
  - Usually they promise to no keep data etc, but even for some companies like banks / insurance, its not enough - so they serve models from open source etc

### Generative UI/UX and CopilotKit

- Transparency - Displaying Tools, reasoning, sources, documents etc

- CopilotKit - OpenSource to FE solutions for AI
  - React Components and Hooks

### Official LangChain Academy Courses

### Using LLMs in Productions

- Open Source vs Vendors

- Open Source
  - Big advantage of privacy and data control
  - All the disadvantages of controlling scale, accuracy and consequently, costs

- Vendors:
  - Easier to use, supports, quality etc

**Obs.:** Eden mentioned about fine tuning, and mentioning he just don't think its needed

### Next Steps

- LLM Ops:
  - Prompt Management
  - Monitoring
  - Debugging
  - Evaluation

- Tools for it:
  - Langsmith
  - Pezzo is a open source for the same purpose
