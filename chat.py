"""Generic chat runner: python chat.py <agent_definition.yaml> [--once "question"]"""
import argparse
import sys

from forge.agent_runtime import Agent, load_definition
from forge.llm import LLMError


def show(trace) -> None:
    a = trace.answer
    print(f"\nagent> {a.answer}")
    if a.refused:
        print(f"       (refused: {a.refusal_reason})")
    for c, ok in zip(a.citations, trace.citations_verified):
        print(f"       [{'verified' if ok else 'UNVERIFIED'}] {c.doc}: \"{c.quote[:100]}\"")
    for t in trace.tool_calls:
        print(f"       [tool {t['tool']}({t['args']})] -> {t['result'][:100]}")
    if trace.error:
        print(f"       ! {trace.error}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("definition")
    ap.add_argument("--once", help="ask a single question and exit")
    args = ap.parse_args()
    try:
        agent = Agent(load_definition(args.definition))
    except Exception as e:  # noqa: BLE001
        print(f"Could not load agent: {e}")
        return 1
    print(f"Loaded '{agent.d.name}' ({len(agent.chunks)} chunks, tools: {list(agent.tools) or 'none'})")
    if args.once:
        try:
            show(agent.ask(args.once))
        except LLMError as e:
            print(e)
            return 1
        return 0
    history: list[tuple[str, str]] = []
    print("Type a question (empty line or Ctrl-C to quit).")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            tr = agent.ask(q, history)
        except LLMError as e:
            print(e)
            continue
        show(tr)
        history.append((q, tr.answer.answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
