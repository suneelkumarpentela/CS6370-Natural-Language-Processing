# Add your import statements here



# Add any utility functions here
def ground_truth_metrics(query_rels,query_ids):
    query_count = len(query_ids)
    query_ids = sorted(query_ids,key = lambda x : int(x))
    ground_truth_list = [ [] for i in range(query_count)]
    for query_info in query_rels:
        query_val = (query_info["query_num"])
        if query_val in query_ids:
            query_i = query_ids.index(query_val)
            ground_truth_list[query_i].append(int(query_info["id"]))

    return ground_truth_list


def ground_truth_relevance_metrics(query_rels,query_ids):
    query_count = len(query_ids)
    query_ids = sorted(query_ids,key = lambda x : int(x))
    ground_truth_list = [ [ [0,0] ] for i in range(query_count)]
    for query_info in query_rels:
        query_val = (query_info["query_num"])
        if query_val in query_ids:
            query_i = query_ids.index(query_val)
            ground_truth_list[query_i][0].append( int(query_info["id"]) )
            ground_truth_list[query_i][1].append( int(query_info["position"]) )

    return ground_truth_list