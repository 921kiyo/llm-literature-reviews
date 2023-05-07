import modal
from modal import Image, SharedVolume, Stub
import os

MODAL_DEPLOYMENT = 'PubFind'

CACHE_PATH = '/root/volume'

stub = Stub(MODAL_DEPLOYMENT)

# *create_package_mounts(["ControlNet"]),
volume = SharedVolume().persist(MODAL_DEPLOYMENT + '_volume')

PubFind_image = Image.debian_slim() \
    .pip_install("transformers", "InstructorEmbedding", "torch", "sentence-transformers")

pdfs_cache = '/root/volume'


# add_pdfs_to_embed(['/Users/hectorlopezhernandez/PycharmProjects/pub_find/appel/pdfs/Appel_ACIE_2012.pdf'])
@stub.local_entrypoint()
def run():
    pass
    # useful when using ```modal run```


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
        model_root = CACHE_PATH
        f = modal.Function.lookup(MODAL_DEPLOYMENT, "get_question_embedding")
        question_embeddings = f.call(queries, model_root)
    else:
        print('Using Local Machine')
        model_root = 'embedding_models'
        question_embeddings = get_question_embedding(queries, model_root)

    return question_embeddings


@stub.function(gpu="A10G",
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


def embed_file_chunks(splits, use_modal='false'):
    """
    Wrapper that will embed questions remotely or locally
    use_modal: Str (comes from env variable)
    """

    if use_modal.lower() == 'true':
        print('Using MODAL')
        model_root = CACHE_PATH
        f = modal.Function.lookup(MODAL_DEPLOYMENT, "get_file_chunks_embeds")
        file_embeddings, num_tokens = f.call(splits, model_root)
    else:
        print('Using Local Machine')
        model_root = 'embedding_models'
        file_embeddings, num_tokens = get_file_chunks_embeds(splits, model_root)

    return file_embeddings, num_tokens


@stub.function(gpu="A10G",
               image=PubFind_image,
               shared_volumes={CACHE_PATH: volume}
               )
def get_file_chunks_embeds(splits, model_root):
    from InstructorEmbedding import INSTRUCTOR
    from transformers import AutoTokenizer
    import torch.cuda
    from tqdm import tqdm

    print(f"GPU access is {'available' if torch.cuda.is_available() else 'Not Available'}")
    model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)
    model_max_seq_length = model.max_seq_length

    tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl',
                                              cache_dir=model_root)  # initialize the INSTRUCTOR tokenizer

    # collect embeddings for all documents
    file_embeddings = []
    num_tokens = []

    # encode each chunk
    for split in tqdm(splits, mininterval=2):
        split = 'Represent the scientific paragraph for retrieval; Input: ' + split

        embeds = model.encode(split)
        tokenized_text = tokenizer(split)['input_ids']
        tokenized_length = len(tokenized_text)

        if tokenized_length > model_max_seq_length:
            print(f'TRUNCATING SPLIT')

        num_tokens.append(tokenized_length)
        file_embeddings.append(embeds)

    return file_embeddings, num_tokens


# TODO: This doesnt add them correctly
def add_pdfs_to_embed(file_paths):
    vol = modal.SharedVolume.lookup(MODAL_DEPLOYMENT + '_volume')
    for file_path in file_paths:
        print(file_path)
        vol.add_local_file(local_path=file_path, remote_path='pdfs_to_embed/')


def delete_pdfs_after_embedding():
    vol = modal.SharedVolume.lookup(MODAL_DEPLOYMENT + '_volume')
    vol.remove_file(path='/pdfs_to_embed/', recursive=True)
