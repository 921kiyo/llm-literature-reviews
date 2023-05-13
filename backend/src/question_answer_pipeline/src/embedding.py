import modal
from modal import Image, SharedVolume, Stub
import os
from tqdm import tqdm

MODAL_DEPLOYMENT = 'PubFind'

CACHE_PATH = '/root/volume'

stub = Stub(MODAL_DEPLOYMENT)

# *create_package_mounts(["ControlNet"]),
volume = SharedVolume().persist(MODAL_DEPLOYMENT + '_volume')
pdfs_cache = '/root/volume'
MODEL_FOLDER = 'instructorXL'
LOCAL_MODEL_FOLDER = './question_answer_pipeline/test/embedding_models'


def download_models():
    from InstructorEmbedding import INSTRUCTOR
    from transformers import AutoTokenizer
    INSTRUCTOR('hkunlp/instructor-xl', cache_folder=MODEL_FOLDER)
    AutoTokenizer.from_pretrained('hkunlp/instructor-xl',
                                  cache_dir=MODEL_FOLDER)


PubFind_image = Image.debian_slim() \
    .pip_install("transformers", "InstructorEmbedding", "torch", "sentence-transformers") \
    .run_function(download_models)


# add_pdfs_to_embed(['/Users/hectorlopezhernandez/PycharmProjects/pub_find/appel/pdfs/Appel_ACIE_2012.pdf'])
@stub.local_entrypoint()
def run():
    import os
    os.environ['ROOT_DIRECTORY'] = 'backend/src/question_answer_pipeline/test'

    from backend.src.question_answer_pipeline.qa_utils.readers import parse_pdf
    import datetime

    import argparse
    f_path = '/Users/hectorlopezhernandez/PycharmProjects/tribe-hackathon/backend/src/question_answer_pipeline/test/pdfs/1411.4116v1.pdf'

    splits, metadata = parse_pdf(f_path, key='', citation='', chunk_chars=1100, overlap=100)
    doc_splits = [splits, splits]
    print('-' * 50)
    print('Embedding')
    print(f'Splits should be of length: {len(splits)}')
    start = datetime.datetime.now()
    print(start.strftime("%H:%M:%S"))
    embed_document(doc_splits, use_modal='true', map_splits=False)
    end = datetime.datetime.now()
    print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
    print('-' * 50)


@stub.function(shared_volumes={pdfs_cache: volume})
def peruse_volume(start_path='./'):
    print('*' * 50)
    for root, dirs, files in os.walk(start_path):
        # Calculate the depth of the current directory
        depth = root[len(start_path):].count(os.sep)
        # Indent based on the depth
        indent = ' ' * 4 * depth
        # Print the current directory
        print(f'{indent}{os.path.basename(root)}/')
        # Indent for files
        file_indent = ' ' * 4 * (depth + 1)
        # Print each file in the current directory
        for file in files:
            print(f'{file_indent}{file}')


def embed_questions(queries, use_modal='false'):
    """
    Wrapper that will embed questions remotely or locally
    use_modal: Str (comes from env variable)
    """

    if use_modal.lower() == 'true':
        print('Using MODAL')
        model_root = MODEL_FOLDER
        f = modal.Function.lookup(MODAL_DEPLOYMENT, "get_question_embedding")
        question_embeddings = f.call(queries, model_root)
    else:
        print('Using Local Machine')
        model_root = LOCAL_MODEL_FOLDER
        question_embeddings = get_question_embedding(queries, model_root)

    return question_embeddings


@stub.function(gpu="T4",
               image=PubFind_image,
               shared_volumes={CACHE_PATH: volume}
               )
def get_question_embedding(queries, model_root):
    from InstructorEmbedding import INSTRUCTOR
    import torch.cuda
    print(f"GPU access is {'available' if torch.cuda.is_available() else 'Not Available'}")
    model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)
    print('Have model')

    question_embeddings = []

    for question in queries:
        question = 'Represent the scientific query for retrieving supporting documents; Input: ' + question
        embeds = model.encode(question)
        question_embeddings.append(embeds)

    print('Have embeddings')
    return question_embeddings


def embed_document(doc_splits, use_modal='false', map_splits=False):
    """

    :param map_splits:
    :param doc_splits:
    :param use_modal:
    :return:
    """
    doc_embeddings = []

    if use_modal.lower() == 'true':
        # Here we split per document

        print('Embedding Document')
        f = modal.Function.lookup(MODAL_DEPLOYMENT, "embed_file_splits")
        # f = embed_file_splits
        iterable_ = [(splits, map_splits) for splits in doc_splits]

        # map over documents
        for file_embeddings, num_tokens in f.starmap(iterable_):
            doc_embeddings.append({'file_embeddings': file_embeddings, 'num_tokens': num_tokens})
    else:
        print('Using Local Machine')
        # instantiate model only once when running on local machine
        from InstructorEmbedding import INSTRUCTOR
        from transformers import AutoTokenizer
        import torch.cuda
        model_root = LOCAL_MODEL_FOLDER
        print(f"GPU access is {'available' if torch.cuda.is_available() else 'Not Available'}")
        model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)
        tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl',
                                                  cache_dir=model_root)  # initialize the INSTRUCTOR tokenizer
        # process items
        for splits in tqdm(doc_splits):
            file_embeddings, num_tokens = embed_file_splits(splits, use_modal=False, map_splits=False, model=model,
                                                            tokenizer=tokenizer)
            doc_embeddings.append({'file_embeddings': file_embeddings, 'num_tokens': num_tokens})

    return doc_embeddings


@stub.function(
    gpu='T4',
    image=PubFind_image,
    shared_volumes={CACHE_PATH: volume}
)
def embed_file_splits(splits, map_splits=False, use_modal=True,  model=None, tokenizer=None):
    """
    Wrapper that will embed questions remotely or locally
    use_modal: Str (comes from env variable)
    """

    file_embeddings = []
    num_tokens = []
    print(f'In embed_file_chunks: map_splits: {map_splits}')
    if map_splits is True:  # use_modal.lower() == 'true':
        print('Mapping Splits with Modal')
        f = get_file_chunks_embeds

        for embeds, tokens in f.map(splits):
            num_tokens.append(tokens)
            file_embeddings.append(embeds)
    else:
        print('Iterating through splits')
        if model is None:
            # if we're loading a container for each model
            model_root = MODEL_FOLDER if use_modal else LOCAL_MODEL_FOLDER
            print(f'Using model at: {model_root}')
            from InstructorEmbedding import INSTRUCTOR
            from transformers import AutoTokenizer
            import torch.cuda

            print(f"GPU access is {'available' if torch.cuda.is_available() else 'not available'}")
            model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)
            tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl',
                                                      cache_dir=model_root)  # initialize the INSTRUCTOR tokenizer

        for split in tqdm(splits, mininterval=2):
            embeds, tokens = get_file_chunks_embeds(split, model=model, tokenizer=tokenizer)
            num_tokens.append(tokens)
            file_embeddings.append(embeds)

    return file_embeddings, num_tokens


@stub.function(gpu="T4",
               image=PubFind_image,
               shared_volumes={CACHE_PATH: volume}
               )
def get_file_chunks_embeds(split, model=None, tokenizer=None):
    if model is None:
        # if we're loading a container for EVERY split
        model_root = MODEL_FOLDER
        from InstructorEmbedding import INSTRUCTOR
        from transformers import AutoTokenizer
        import torch.cuda

        print(f"GPU access is {'available' if torch.cuda.is_available() else 'not available'}")
        model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)
        tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl',
                                                  cache_dir=model_root)  # initialize the INSTRUCTOR tokenizer
    # encode each chunk
    model_max_seq_length = model.max_seq_length
    split = 'Represent the scientific paragraph for retrieval; Input: ' + split

    embeds = model.encode(split)
    tokenized_text = tokenizer(split)['input_ids']
    num_tokens = len(tokenized_text)

    if num_tokens > model_max_seq_length:
        print(f'TRUNCATING SPLIT')
        # collect embeddings for all documents

    return embeds, num_tokens


# TODO: This doesnt add them correctly
def add_pdfs_to_embed(file_paths):
    vol = modal.SharedVolume.lookup(MODAL_DEPLOYMENT + '_volume')
    for file_path in file_paths:
        print(file_path)
        vol.add_local_file(local_path=file_path, remote_path='pdfs_to_embed/')


def delete_pdfs_after_embedding():
    vol = modal.SharedVolume.lookup(MODAL_DEPLOYMENT + '_volume')
    vol.remove_file(path='/pdfs_to_embed/', recursive=True)
