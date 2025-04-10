#!/usr/bin/env python3
"""
VERIFY_PATCH_ULTRATINY_WITH_FORWARD_9_OPTIMIZE_MINIMAL_SAVE.PY

1) Loads a tf.posit16 ultra-tiny MLP from 'posit16_ultratinymlp.ckpt'.
2) Picks digit=9 from MNIST test set, does forward pass => label=9.
3) Defines a large bottom patch => rows=20..27, cols=0..27, ±0.1
   BUT uses Z3.Optimize to minimize sum(|x_new - x_orig|) in that region.
   So the solver changes only the needed subset of that big patch.
4) If sat => prints minimal cost + *also saves a color-coded image* 
   showing which pixels changed.

Author: ChatGPT
"""

import sys, os
import numpy as np
import tensorflow as tf
from z3 import Optimize, Real, Bool, Or, And, Implies, Not, sat

import matplotlib
matplotlib.use("Agg")  # so we can save without needing a GUI
import matplotlib.pyplot as plt

###############################################################################
# 1) Helper: Convert Z3 value -> float
###############################################################################
def z3_value_to_float(z3_val):
    """
    If Z3 returns '3/2' => 1.5, or '1.234?' => 1.234
    """
    val_str = str(z3_val)
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num)/float(den)
    else:
        if '?' in val_str:
            val_str = val_str.split('?')[0]
        return float(val_str)


def main():
    ############################################################################
    # A) Parse command line => "posit16"
    ############################################################################
    if len(sys.argv) > 1:
        data_t = sys.argv[1]
    else:
        data_t = "float32"
    print("data_t =", data_t)

    CKPT_NAME = data_t + "_ultratinymlp.ckpt"
    CKPT_PATH = os.path.join(".", CKPT_NAME)
    print("Checkpoint path:", CKPT_PATH)

    ############################################################################
    # B) Build a graph => ultra-tiny MLP in tf.posit16
    ############################################################################
    tf.reset_default_graph()

    x_ph = tf.placeholder(tf.posit16, shape=(None,28,28,1), name="x_ph")

    def flatten_28x28(x):
        return tf.reshape(x, [-1,784])

    def ultra_tiny_net(x):
        # variable names must match your checkpoint's (Variable, Variable_1, etc.)
        W1 = tf.get_variable("Variable",   shape=[784,8], dtype=tf.posit16)
        b1 = tf.get_variable("Variable_1", shape=[8],     dtype=tf.posit16)
        lin1= tf.matmul(flatten_28x28(x), W1) + b1
        fc1 = tf.nn.relu(lin1)

        W2 = tf.get_variable("Variable_2", shape=[8,10],  dtype=tf.posit16)
        b2 = tf.get_variable("Variable_3", shape=[10],    dtype=tf.posit16)
        logits = tf.matmul(fc1, W2) + b2
        return logits, (W1,b1,W2,b2)

    logits_tf, var_tuple = ultra_tiny_net(x_ph)
    logits_float = tf.cast(logits_tf, tf.float32)
    pred_tf = tf.argmax(logits_float, axis=1, output_type=tf.int32)

    # gather main variables ignoring Adam
    main_vars = {}
    for v in tf.global_variables():
        nm = v.name.split(":")[0]
        if nm in ["Variable","Variable_1","Variable_2","Variable_3"]:
            main_vars[nm] = v

    saver = tf.train.Saver(var_list=main_vars)

    ############################################################################
    # C) Load MNIST => pick digit=9
    ############################################################################
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    idx_9 = None
    for i,lab in enumerate(y_test):
        if lab == 9:
            idx_9 = i
            break
    if idx_9 is None:
        print("No digit=9 in test set!")
        return

    print("Test image => idx=", idx_9, " label=", y_test[idx_9])

    orig_img_28 = X_test[idx_9].astype(np.float32)
    # normalize to [-1..1]
    orig_img_28 = (orig_img_28 - 127.5)/127.5
    test_img = orig_img_28[None,:,:,None]

    ############################################################################
    # D) TF session => restore => forward pass => read var
    ############################################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, CKPT_PATH)

        out_logits, out_pred = sess.run([logits_float, pred_tf],
                                        feed_dict={x_ph: test_img})
        print("Forward pass logits:", out_logits)
        print("Predicted label =>", out_pred[0])

        W1_val = sess.run(var_tuple[0])
        b1_val = sess.run(var_tuple[1])
        W2_val = sess.run(var_tuple[2])
        b2_val = sess.run(var_tuple[3])

    ############################################################################
    # E) Build Z3.Optimize => entire bottom band [20..27,0..27], eps=0.1
    #    then define cost = sum(|x_new - orig|) in that region
    #    classification => logit(k)>logit(9) for some k!=9
    ############################################################################
    from z3 import Optimize, Real, Bool, Or, And, Implies, Not

    opt = Optimize()
    x_vars = [ Real(f"x_{i}") for i in range(784) ]

    r0, r1 = 20, 27
    c0, c1 = 0, 27
    eps = 0.1

    abs_diffs = []
    def flatten_idx(r,c): return r*28 + c

    for r in range(28):
        for c in range(28):
            idx = flatten_idx(r,c)
            orig_val = float(orig_img_28[r,c])

            if r0 <= r <= r1 and c0 <= c <= c1:
                # allow [orig_val-eps, orig_val+eps]
                opt.add(x_vars[idx] >= orig_val - eps)
                opt.add(x_vars[idx] <= orig_val + eps)
                # define absdiff var
                dvar = Real(f"absdiff_{idx}")
                opt.add(dvar >= x_vars[idx] - orig_val,
                        dvar >= orig_val - x_vars[idx])
                abs_diffs.append(dvar)
            else:
                # fix outside patch
                opt.add(x_vars[idx] == orig_val)

    # hidden => real domain
    hidden_vars = []
    for i in range(8):
        lin_expr = float(b1_val[i])
        for j in range(784):
            lin_expr += float(W1_val[j,i])* x_vars[j]
        hv = Real(f"hidden_{i}")
        # piecewise ReLU
        rb = Bool(f"relu_{i}")
        opt.add((lin_expr >= 0) == rb)
        opt.add(Implies(rb, hv == lin_expr))
        opt.add(Implies(Not(rb), hv == 0))
        opt.add(hv >= 0)
        opt.add(hv >= lin_expr)

        hidden_vars.append(hv)

    # final => 10 reals
    logits_vars = []
    for d in range(10):
        val_d = float(b2_val[d])
        for i in range(8):
            val_d += float(W2_val[i,d])* hidden_vars[i]
        lv = Real(f"logit_{d}")
        opt.add(lv == val_d)
        logits_vars.append(lv)

    # misclass => ∃ k!=9 => logit[k]>logit(9)
    logit9 = logits_vars[9]
    or_list = []
    for k in range(10):
        if k!=9:
            or_list.append(logits_vars[k] > logit9)
    opt.add(Or(or_list))

    # cost => sum(abs_diffs)
    cost = Real("cost")
    opt.add(cost == sum(abs_diffs))

    handle = opt.minimize(cost)

    print(f"\nOptimize => searching minimal shift in [20..27,0..27], eps={eps}")
    res = opt.check()
    print("Solver result:", res)
    if res == sat:
        print("Found minimal adversarial => digit=9 misclassified.")
        print("Minimal L1 shift =>", opt.lower(handle))

        m = opt.model()
        adv_img_28 = np.zeros((28,28), dtype=np.float32)
        for idx in range(784):
            zval = m[x_vars[idx]]
            val_f = z3_value_to_float(zval)
            rr, cc = divmod(idx, 28)
            adv_img_28[rr,cc] = val_f

        # Now we make a (28,28) "diff_mask" where:
        # 0 => not changed, 1 => changed by > diff_thresh
        diff_thresh = 1e-5
        diff_mask = np.zeros((28,28), dtype=np.float32)
        for rr in range(28):
            for cc in range(28):
                if abs(adv_img_28[rr,cc] - orig_img_28[rr,cc]) > diff_thresh:
                    diff_mask[rr,cc] = 1.0

        # We'll save this diff_mask as a color-coded image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4,4))
        plt.imshow(diff_mask, cmap='jet', vmin=0, vmax=1, origin='upper')
        plt.colorbar(label="changed pixel=1, unchanged=0")
        plt.title("Changed Pixels in Minimal Adversarial")
        outfile = "adv_diff.png"
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Saved changed-pixel mask to '{outfile}'")

    else:
        print("No solution =>", res)
        print("Try bigger eps or different patch.")


if __name__=="__main__":
    main()
