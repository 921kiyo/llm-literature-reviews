import os

# Set directory for target documents
os.environ['ROOT_DIRECTORY'] = 'backend/src/question_answer_pipeline/test'
import json
from backend.src.question_answer_pipeline.src.utils import initialize_docstore, get_answers
from backend.src.question_answer_pipeline.src.embedding import embed_questions
from dotenv import load_dotenv
from backend.src.question_answer_pipeline.src.utils import from_arxiv_docstore

load_dotenv()


def main(question, force_rebuild=False, use_modal=True):
    """
    q and a on pdf documents
    :param question:
    :param force_rebuild:
    :param use_modal:
    :return:
    """

    # create a docstore that stays updated with the filesystem
    # it is rebuilt if pdfs are deleted and items are added when new files are detected
    docs = initialize_docstore(force_rebuild=force_rebuild)

    queries = [question]

    print('embedding')
    question_embeddings = embed_questions(queries, use_modal=os.environ['MODAL'])

    print('getting answers')
    answers = get_answers(docs, queries, question_embeddings)

    for answer in answers:
        print(answer.formatted_answer)


def qa_abstracts(question, arxiv_results=None):
    # create a docstore that stays updated with the filesystem
    # it is rebuilt if pdfs are deleted and items are added when new files are detected

    docs = from_arxiv_docstore(arxiv_results)

    queries = [question]

    print('embedding question')
    question_embeddings = embed_questions(queries, use_modal=os.environ['MODAL'])

    print('getting answers')
    answers = get_answers(docs, queries, question_embeddings)

    for answer in answers:
        print(answer.formatted_answer)


if __name__ == '__main__':
    import argparse

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Question and answer")
    # Define the command-line arguments using the add_argument() method
    parser.add_argument("question", type=str, help="Your name")

    args = parser.parse_args()
    print(f'Question: {args.question}')

    file = './backend/src/question_answer_pipeline/example_arxiv_result.json'
    arxiv_results = json.load(open(file, 'r'))
    qa_abstracts(question=args.question, arxiv_results=arxiv_results)
