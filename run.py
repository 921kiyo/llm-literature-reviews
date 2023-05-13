from embedding import embed_document
from dotenv import load_dotenv


load_dotenv()


if __name__ == '__main__':
    """
        1) test doc parallelization
        2) test split parallelization
    """
    import os
    os.environ['ROOT_DIRECTORY'] = ROOT_DIRECTORY = 'backend/src/question_answer_pipeline/test'

    from backend.src.question_answer_pipeline.qa_utils.readers import parse_pdf
    import datetime

    import argparse
    f_path = '/Users/hectorlopezhernandez/PycharmProjects/tribe-hackathon/backend/src/question_answer_pipeline/test/pdfs/1411.4116v1.pdf'

    splits, metadata = parse_pdf(f_path, key='', citation='', chunk_chars=1100, overlap=100)
    doc_splits = [splits, splits, splits]
    metadatas = [metadata, metadata, metadata]
    print('-'*50)
    print('Embedding')
    print(f'Splits should be of length: {len(splits)}')
    start = datetime.datetime.now()
    print(start.strftime("%H:%M:%S"))
    embed_document(doc_splits, use_modal='true')
    end = datetime.datetime.now()
    print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end-start).total_seconds():.3}')
    print('-'*50)

