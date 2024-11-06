def string_to_variations_list(next_str):
    final_list = []
    # for s in {next_str, next_str.title(), next_str.upper(), next_str.lower()}:  # get case variations of string
    for s in {next_str,}:  # get case variations of string
        final_list += [s]  # add leading space variation for some tokenizers
    return final_list
