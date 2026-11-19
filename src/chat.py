from search import search_prompt


def main():
    print("Chat iniciado. Digite 'sair' para encerrar.")
    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            break

        if question.casefold() in {"sair", "exit", "quit"}:
            print("Chat encerrado.")
            break
        if not question:
            continue

        answer = search_prompt(question)
        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()