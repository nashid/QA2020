import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import argparse

def min_max(arr):
    mi = np.min(arr)
    ma = np.max(arr)
    res = []
    for x in arr:
        res.append( float(x - mi) / (ma - mi) )
    return res


def calc_auc(initrank_file, pred_file):
    # read ground truth file
    q_answers = {}
    with open(initrank_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line == "":
                continue

            arr = line.split(" ")

            groud_truth = arr[0]
            qid = arr[1].split(":")[-1]
            did = arr[-1]

            if groud_truth == "1":
                if qid in q_answers.keys():
                    q_answers[qid].append(did)
                else:
                    q_answers[qid] = [did]

    # read pred file
    q_pred = {}
    with open(pred_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue

            arr = line.split(" ")
            qid = arr[0]
            did = arr[2]
            rank = int(arr[3])
            score = float(arr[4])
            if qid in q_pred.keys():
                q_pred[qid].append([did, score])
            else:
                q_pred[qid] = [[did, score]]

    auc_list = []

    for qid in q_pred.keys():
        if int(qid) >= 637 and int(qid) <= 641:
            continue
        y_true = []
        y_score = []

        ans = q_answers[qid]

        for did, score in q_pred[qid]:
            if did in ans:
                y_true.append(1)
            else:
                y_true.append(0)

            y_score.append(score)

        y_score = min_max(y_score)
        y_score = np.array(y_score)
        y_true = np.array(y_true)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)

        auc_list.append(auc)
        print("qid: %s  -  auc:%.4f" % (qid, auc))



    final_auc = np.mean(auc_list)
    print("=== Evaluation results ===")
    print("Test total:", len(auc_list))
    print("Prediction file:", pred_file)
    # print("len(pred.keys):", len(q_pred.keys()))
    print("auc =", final_auc)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type',  type=str, default='twitter')
    parser.add_argument('--model_type', help='model_type',  type=str, default='lambdaMART')
    args = parser.parse_args()

    data_type = args.data_type
    print("data_type:", data_type)

    model_type = args.model_type
    test_file_path = "for_ltr/ltr_%s_test_v2.txt" % data_type
    pred_file_path = "ltr/predicts/%s_%s_MAP_pred.txt" % (data_type, model_type)

    calc_auc(test_file_path, pred_file_path)
