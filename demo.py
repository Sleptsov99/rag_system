"""
Demo — no external files needed.
Ingests a small set of hard-coded texts and runs example queries.

Run:
  python demo.py                  # uses dummy LLM (shows retrieval only)
  python demo.py --provider ollama
  python demo.py --provider openai
"""

import argparse

from src.pipeline import RAGPipeline

DEMO_DOCS = {
    "ml_basics.txt": """
Машинное обучение (Machine Learning, ML) — это раздел искусственного интеллекта,
в котором системы обучаются на данных без явного программирования.
Основные типы ML:
  • Обучение с учителем (supervised): модель обучается на размеченных данных.
    Примеры: классификация, регрессия.
  • Обучение без учителя (unsupervised): данные не размечены.
    Примеры: кластеризация (K-Means), снижение размерности (PCA).
  • Обучение с подкреплением (reinforcement): агент учится через взаимодействие
    со средой, получая вознаграждения.
""",
    "nlp_basics.txt": """
NLP (Natural Language Processing) — обработка естественного языка.
Ключевые задачи NLP:
  • Токенизация — разбиение текста на слова/подслова.
  • Лемматизация — приведение слов к базовой форме (бежать → бежать).
  • NER — распознавание именованных сущностей (Москва → LOCATION).
  • Классификация текстов — спам-фильтр, анализ тональности.
  • Машинный перевод — seq2seq модели, трансформеры.

Современные подходы основаны на трансформерной архитектуре (BERT, GPT, T5).
Эмбеддинги — векторные представления слов и предложений в многомерном пространстве.
Word2Vec обучает эмбеддинги, предсказывая контекст слова.
FastText улучшает Word2Vec, разбивая слова на символьные n-граммы.
""",
    "rag_explained.txt": """
RAG (Retrieval-Augmented Generation) — архитектура, объединяющая:
  1. Retrieval (поиск): поиск релевантных фрагментов из базы знаний.
  2. Generation (генерация): LLM генерирует ответ на основе найденных фрагментов.

Зачем RAG?
  • Позволяет работать с актуальными или приватными данными.
  • Снижает «галлюцинации» модели: ответ опирается на реальные документы.
  • Не требует дообучения модели (fine-tuning) при обновлении базы знаний.

Компоненты:
  • Векторная база данных (ChromaDB, FAISS, Pinecone) — хранит эмбеддинги.
  • Embedding model — кодирует текст в вектор (sentence-transformers, OpenAI).
  • LLM — генерирует финальный ответ (GPT-4, LLaMA, Claude).

Метрика качества поиска: cosine similarity между вектором запроса
и векторами документов.
""",
    "transformers.txt": """
Трансформеры — архитектура нейросетей, предложенная в статье
«Attention Is All You Need» (Vaswani et al., 2017).

Ключевые идеи:
  • Self-Attention: каждый токен «смотрит» на все остальные, вычисляя,
    насколько они важны для его представления.
  • Multi-Head Attention: несколько голов attention параллельно.
  • Positional Encoding: добавляет информацию о позиции токенов.

BERT (Bidirectional Encoder Representations from Transformers):
  • Двунаправленный энкодер — учитывает контекст с обеих сторон.
  • Pre-training: MLM (Masked Language Modeling) + NSP.
  • Используется для classification, NER, QA.

GPT (Generative Pre-trained Transformer):
  • Авторегрессивная модель — предсказывает следующий токен.
  • Используется для генерации текста.
""",
}

EXAMPLE_QUERIES = [
    "Что такое RAG и зачем он нужен?",
    "Чем BERT отличается от GPT?",
    "Что такое лемматизация?",
    "Какие типы машинного обучения существуют?",
    "Как работает механизм Attention?",
]


def main():
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument(
        "--provider", default="dummy",
        choices=["ollama", "openai", "dummy"],
        help="LLM provider",
    )
    parser.add_argument("--mmr", action="store_true", help="Use MMR retrieval")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG System Demo")
    print(f"  LLM provider: {args.provider}")
    print("=" * 60)

    pipeline = RAGPipeline(llm_provider=args.provider)
    pipeline.clear_index()

    # Ingest demo documents
    print("\n--- Ingesting demo documents ---")
    for name, text in DEMO_DOCS.items():
        n = pipeline.ingest_text(text.strip(), source=name)
        print(f"  {name}: {n} chunks")

    print(f"\nTotal chunks in index: {pipeline.document_count}")

    # Run example queries
    print("\n--- Running example queries ---\n")
    for question in EXAMPLE_QUERIES:
        print(f"Q: {question}")
        result = pipeline.query(
            question,
            top_k=3,
            use_mmr=args.mmr,
            return_sources=True,
        )
        print(f"A: {result['answer']}")
        print("Sources:", ", ".join(
            f"{s['source']} (score={s['score']:.2f})"
            for s in result["sources"]
        ))
        print("-" * 50)

    # Interactive mode after demo
    print("\nEnter your own questions (Ctrl+C to exit):")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not q:
            continue
        result = pipeline.query(q, top_k=3, return_sources=True)
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
