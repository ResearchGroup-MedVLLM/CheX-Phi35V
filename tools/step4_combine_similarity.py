import os
import h5py
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def compute_hamming_distances(query_bin, gallery_bin):
    query_bin = query_bin.astype(bool)
    gallery_bin = gallery_bin.astype(bool)
    dists = np.zeros((len(query_bin), len(gallery_bin)))
    for i, q in enumerate(query_bin):
        dists[i] = np.mean(q != gallery_bin, axis=1)
    return dists

def multi_modal_similarity_search(query_feats, gallery_feats, metrics, weights=None, top_k=10):
    num_query = query_feats[list(query_feats.keys())[0]].shape[0]
    num_gallery = gallery_feats[list(gallery_feats.keys())[0]].shape[0]
    total_dist = np.zeros((num_query, num_gallery))

    for modality in query_feats:
        q_feat = query_feats[modality]
        g_feat = gallery_feats[modality]
        weight = weights.get(modality, 1.0) if weights else 1.0

        if metrics[modality] == 'cosine':
            dist = cosine_distances(q_feat, g_feat)
        elif metrics[modality] == 'hamming':
            dist = compute_hamming_distances(q_feat, g_feat)
        else:
            raise ValueError(f"Unsupported distance metric: {metrics[modality]}")

        total_dist += weight * dist

    topk_indices = [np.argsort(d_row)[:top_k].tolist() for d_row in total_dist]
    return topk_indices

def compute_top_k_accuracy(query_indices, labels, top_k_range=100):
    acc_results = {f'top_{k}': 0 for k in range(1, top_k_range + 1)}
    for k in range(1, top_k_range + 1):
        total_k = k * len(query_indices)
        count_sft_acc_0 = sum(sum(labels[idx]['sft_acc'] == 0 for idx in top_k[:k]) for top_k in query_indices)
        acc_results[f'top_{k}'] = count_sft_acc_0 / total_k
    return acc_results

def load_features(file_path):
    with h5py.File(file_path, 'r') as f:
        return np.array(f['features'])

def load_labels(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def main():

    root = './data/features/'
    label_file = "./data/analyze/train_random_20k.json"
    retrieval_results = {}

    # 加载特征
    query_features_q = load_features(os.path.join(root, "question_features_retrieval_100.h5"))
    query_features_e = load_features(os.path.join(root, "explanation_features_retrieval_100.h5"))
    query_features_v = load_features(os.path.join(root, "image_features_retrieval_100.h5"))
    query_features_d = load_features(os.path.join(root, "disease_features_retrieval_100.h5"))

    gallery_features_q = load_features(os.path.join(root, "question_features_train_random_20k.h5"))
    gallery_features_e = load_features(os.path.join(root, "explanation_features_train_random_20k.h5"))
    gallery_features_v = load_features(os.path.join(root, "image_features_train_random_20k.h5"))
    gallery_features_d = load_features(os.path.join(root, "disease_features_train_random_20k.h5"))

    labels = load_labels(label_file)

    # 1. q + e
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'e': query_features_e},
        gallery_feats={'q': gallery_features_q, 'e': gallery_features_e},
        metrics={'q': 'cosine', 'e': 'cosine'}
    )
    retrieval_results["q_e"] = compute_top_k_accuracy(indices, labels)

    # 2. q + e + disease (hamming)
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'e': query_features_e, 'd': query_features_d},
        gallery_feats={'q': gallery_features_q, 'e': gallery_features_e, 'd': gallery_features_d},
        metrics={'q': 'cosine', 'e': 'cosine', 'd': 'hamming'}
    )
    retrieval_results["q_e_disease"] = compute_top_k_accuracy(indices, labels)

    # 3. q + e + image (cosine)
    indices = multi_modal_similarity_search(
        query_feats={'e': query_features_e, 'v': query_features_v},
        gallery_feats={ 'e': gallery_features_e, 'v': gallery_features_v},
        metrics={'e': 'cosine', 'v': 'cosine'}
    )
    retrieval_results["e_v"] = compute_top_k_accuracy(indices, labels)

    # 3. q + v
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'v': query_features_v},
        gallery_feats={'q': gallery_features_q, 'v': gallery_features_v},
        metrics={'q': 'cosine',  'v': 'cosine'}
    )
    retrieval_results["q_v"] = compute_top_k_accuracy(indices, labels)



    # 4. q + e + disease + image
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'e': query_features_e, 'd': query_features_d, 'v': query_features_v},
        gallery_feats={'q': gallery_features_q, 'e': gallery_features_e, 'd': gallery_features_d, 'v': gallery_features_v},
        metrics={'q': 'cosine', 'e': 'cosine', 'd': 'hamming', 'v': 'cosine'}
    )
    retrieval_results["q_e_disease_image"] = compute_top_k_accuracy(indices, labels)

    # 5. q + e + disease - image（等价于加入 image 后赋负权重）
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'e': query_features_e, 'd': query_features_d, 'v': query_features_v},
        gallery_feats={'q': gallery_features_q, 'e': gallery_features_e, 'd': gallery_features_d, 'v': gallery_features_v},
        metrics={'q': 'cosine', 'e': 'cosine', 'd': 'hamming', 'v': 'cosine'},
        weights={'q': 1, 'e': 1, 'd': 1, 'v': -1}
    )
    retrieval_results["q_e_disease_minus_image"] = compute_top_k_accuracy(indices, labels)

    # 6. q + e - image
    indices = multi_modal_similarity_search(
        query_feats={'q': query_features_q, 'e': query_features_e, 'v': query_features_v},
        gallery_feats={'q': gallery_features_q, 'e': gallery_features_e, 'v': gallery_features_v},
        metrics={'q': 'cosine', 'e': 'cosine', 'v': 'cosine'},
        weights={'q': 1.0, 'e': 1.0, 'v': -1.0}
    )
    retrieval_results["q_e_minus_image"] = compute_top_k_accuracy(indices, labels)


    # 输出所有结果
    for name, result in retrieval_results.items():
        print(f"\nResults for {name}:")
        for k, acc in result.items():
            print(f"{k}: {acc:.4f}")

if __name__ == "__main__":
    main()
