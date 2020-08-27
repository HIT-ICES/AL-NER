# Python Standard Modules
import os


def create_dir_if_not_exits(dir_path, logger):
    """
    confirm whether dir_path is a directory
    If it is a file, an error is reported.
    If the directory does not exist, create a new one
    :param dir_path
    :param logger
    """
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            error_tips = f"Path: {dir_path} exits, but is not a directory!"
            logger.error(error_tips)
            raise ValueError(error_tips)
    else:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Complete the confirmation of the output folder.\n")
    return


def flatten_lists(lists):
    """
    flatten lists to 1d
    """
    flatten_list = []
    for temp in lists:
        if type(temp) == list:
            flatten_list += temp
        else:
            flatten_list.append(temp)
    return flatten_list


def vec_to_tags(tags, vecs, max_seq_len=64):
    """
    change vector to tags
    """
    idx_to_tag = {key: idx for key, idx in enumerate(tags)}
    tags = []

    for vec in vecs:
        tag = [idx_to_tag.get(idx) for idx in vec[:max_seq_len] if idx >0]
        tags.append(tag)

    return tags

def tagseq_to_entityseq(tags: list) -> list:
    """
    Convert tags format:
    [ "B-LOC", "I-LOC", "O", B-PER"] -> [(0, 2, "LOC"), (3, 4, "PER")]
    """
    entity_seq = []
    tag_name = ""
    start, end = 0, 0
    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if tag_name != "":
                end = index
                entity_seq.append((start, end, tag_name))
            tag_name = tag[2:]
            start = index
        elif tag.startswith("I-"):
            if tag_name == "" or tag_name == tag[2:]:
                continue
            else:
                end = index
                entity_seq.append((start, end, tag_name))
                tag_name = ""
        else:  # "O"
            if tag_name == "":
                continue
            else:
                end = index
                entity_seq.append((start, end, tag_name))
                tag_name = ""
    return entity_seq
