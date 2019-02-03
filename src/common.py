import numpy as np


def train_model(model, tp, tn):
    ph = [x[0] for x in tp]
    pr = [x[1] for x in tp]
    pt = [x[2] for x in tp]
    nh = [x[0] for x in tn]
    nt = [x[2] for x in tn]
    model.train(ph, pr, pt, nh, nt)


def test_model(model, triple):
    h = [x[0] for x in triple]
    r = [x[1] for x in triple]
    t = [x[2] for x in triple]
    batch_size = 512
    num_step = int(len(triple) / batch_size)
    if len(triple) > num_step * batch_size:
        num_step += 1
    k_def = [1, 5, 10, 50]
    hits_k = [0.0 for _ in k_def]
    mr = 0.0
    mrr = 0.0
    for i in range(num_step):
        batch_h = h[i*batch_size:(i+1)*batch_size]
        batch_r = r[i*batch_size:(i+1)*batch_size]
        batch_t = t[i*batch_size:(i+1)*batch_size]
        score_h = model.predict_head(batch_r, batch_t)
        score_t = model.predict_tail(batch_h, batch_r)
        batch_rank_h = np.argsort(np.argsort(score_h, axis=1), axis=1)
        batch_rank_t = np.argsort(np.argsort(score_t, axis=1), axis=1)
        for j in range(len(batch_h)):
            rank_h = batch_rank_h[j, batch_h[j]] + 1
            rank_t = batch_rank_t[j, batch_t[j]] + 1
            mr += rank_h
            mr += rank_t
            mrr += 1 / rank_h
            mrr += 1 / rank_t
            for k in range(len(k_def)):
                if rank_h <= k_def[k]:
                    hits_k[k] += 1
                if rank_t <= k_def[k]:
                    hits_k[k] += 1
    for k in range(len(k_def)):
        hits_k[k] = hits_k[k] / (len(triple) * 2)
    mr = mr / (len(triple) * 2)
    mrr = mrr / (len(triple) * 2)
    print("hits@{}={}, MR={}, MRR={}".format(k_def, hits_k, mr, mrr))
