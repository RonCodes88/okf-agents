# Support Agent Example

A support Q&A agent built from a directory of linked Markdown files
using `okf-agents`.

## What's included

```
knowledge-base/
├── index.md                    # bundle root index
└── concepts/
    ├── billing.md              # links to refunds, account-setup, troubleshooting
    ├── refunds.md              # links back to billing
    ├── account-setup.md        # links to billing, troubleshooting
    └── troubleshooting.md      # links to billing, account-setup
```

## Run

```bash
pip install okf-agents
python demo.py
```

No API keys needed — the tools, retriever, and router are fully
deterministic. To use the navigator subgraph with an LLM, install a
provider package (e.g. `pip install langchain-openai`) and pass the
model to `create_okf_navigator()`.
