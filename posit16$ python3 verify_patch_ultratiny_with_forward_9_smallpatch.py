#!/usr/bin/env python3
"""
VERIFY_PATCH_ULTRATINY_WITH_FORWARD_9_SMALLPATCH_VISUAL.PY

Same as your working code for digit=9, patch => [10..17],[10..17], eps=0.1,
but adds final lines to:
 - retrieve the sat assignment from Z3,
 - reconstruct the new adversarial image,
 - display an ASCII view showing which pixels changed.

No changes to the logic. The patch, digit=9, etc. are all the same.
"""

import sys, os
import numpy as np
import tensorflow as tf
from z3 import Solver, Real, Bool, Or, And, Implies, Not, sat

###################################
# Helper for Z3->float conversion
###################################
def z3_value_to_float(z3_val):
    val_str = str(z3_val)
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num)/float(den)
    else:
        if '?' in val_str:
            val_str = val_str.split('?')[0]
        return float(val_str)

###################################
# Helper for ASCII visualization
###################################
def ascii_adversarial_with_changes(orig_img, adv_img, diff_thresh=1e-5):
    """
    Mark 'X' where |adv - orig| > diff_thresh, else brightness char from .:-=+*#%@
    Both orig_img, adv_img shape=(28,28)
    """
    chars = " .:-=+*#%@"
    img_min, img_max = adv_img.min(), adv_img.max()
    rng = img_max - img_min if (img_max>img_min) else 1e-6

    lines = []
    for r in range(28):
        row_str = ""
        for c in range(28):
            diff = abs(adv_img[r,c] - orig_img[r,c])
            if diff > diff_thresh:
                row_str += 'X'
            else:
                val = adv_img[r,c]
                norm = (val - img_min)/rng
                idx = int(norm*(len(chars)-1))
                row_str += chars[idx]
        lines.append(row_str)
    return lines


def main():
    # 1) parse cmd-line
    if len(sys.argv)>1:
        data_t = sys.argv[1]
    else:
        data_t = "float32"
    print("data_t =", data_t)

    CKPT_NAME = data_t + "_ultratinymlp.ckpt"  # e.g. "posit16_ultratinymlp.ckpt"
    CKPT_PATH = os.path.join(".", CKPT_NAME)
    print("Checkpoint path:", CKPT_PATH)

    # 2) single default graph
    tf.reset_default_graph()

    x_ph = tf.placeholder(tf.posit16, shape=(None,28,28,1), name="x_ph")

    def flatten_28x28(x):
        return tf.reshape(x, [-1,784])

    def ultra_tiny_net(x):
        # same var names
        W1 = tf.get_variable("Variable",   shape=[784,8], dtype=tf.posit16)
        b1 = tf.get_variable("Variable_1", shape=[8],     dtype=tf.posit16)
        lin1 = tf.matmul(flatten_28x28(x), W1) + b1
        fc1  = tf.nn.relu(lin1)

        W2 = tf.get_variable("Variable_2", shape=[8,10],  dtype=tf.posit16)
        b2 = tf.get_variable("Variable_3", shape=[10],    dtype=tf.posit16)
        logits = tf.matmul(fc1, W2) + b2
        return logits, (W1, b1, W2, b2)

    logits_tf, var_tuple = ultra_tiny_net(x_ph)
    logits_float = tf.cast(logits_tf, tf.float32)
    pred_tf = tf.argmax(logits_float, axis=1, output_type=tf.int32)

    # gather main variables
    main_vars = {}
    for v in tf.global_variables():
        short_name = v.name.split(":")[0]
        if short_name in ["Variable","Variable_1","Variable_2","Variable_3"]:
            main_vars[short_name] = v
    saver = tf.train.Saver(var_list=main_vars)

    # 3) pick digit=9 from MNIST
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    idx_9 = None
    for i,lab in enumerate(y_test):
        if lab==9:
            idx_9=i
            break
    if idx_9 is None:
        print("No digit=9 found in test set!")
        return
    print("Test image => idx=", idx_9, " label=", y_test[idx_9])

    orig_img_28 = X_test[idx_9].astype(np.float32)
    orig_img_28 = (orig_img_28 - 127.5)/127.5
    test_img = orig_img_28[None,:,:,None]

    # 4) restore => forward pass => read vars
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

    # 5) Z3 real-based patch => rows=10..17, cols=10..17, eps=0.1
    s = Solver()
    x_vars = [ Real(f"x_{i}") for i in range(784) ]

    r0, r1 = 10, 17
    c0, c1 = 10, 17
    eps = 0.1

    def flatten_idx(r,c):
        return r*28 + c

    for r in range(28):
        for c in range(28):
            idx = flatten_idx(r,c)
            orig_val = float(orig_img_28[r,c])
            if (r0<=r<=r1) and (c0<=c<=c1):
                s.add(x_vars[idx] >= (orig_val - eps))
                s.add(x_vars[idx] <= (orig_val + eps))
            else:
                s.add(x_vars[idx] == orig_val)

    # hidden(8)
    hidden_vars = []
    for i in range(8):
        lin_expr = float(b1_val[i])
        for j in range(784):
            lin_expr += float(W1_val[j,i])* x_vars[j]
        hv = Real(f"hidden_{i}")
        hidden_vars.append(hv)

        relu_bool = Bool(f"relu_{i}")
        s.add(Implies(relu_bool, hv==lin_expr))
        s.add(Implies(Not(relu_bool), hv==0))
        s.add((lin_expr>=0)==relu_bool)
        s.add(hv>=0, hv>=lin_expr)

    # final => 10 reals
    logits_vars = []
    for d in range(10):
        ld = float(b2_val[d])
        for i in range(8):
            ld += float(W2_val[i,d])* hidden_vars[i]
        lv = Real(f"logit_{d}")
        logits_vars.append(lv)
        s.add(lv==ld)

    # misclass => ∃ k!=9 => logit[k] > logit[9]
    logit9 = logits_vars[9]
    or_list = []
    for k in range(10):
        if k!=9:
            or_list.append( logits_vars[k] > logit9 )
    s.add(Or(or_list))

    print(f"\nSolving in Z3 for digit=9 with patch [10..17],[10..17], eps=0.1 ...")
    res = s.check()
    print("Solver result:", res)

    # 6) If sat => retrieve new pixels, show ASCII differences
    if res == sat:
        print("Found an adversarial => digit=9 misclassified with smaller patch.\n")

        m = s.model()
        adv_img_28 = np.zeros((28,28), dtype=np.float32)
        for idx in range(784):
            val_z3 = m[x_vars[idx]]
            val_f  = z3_value_to_float(val_z3)
            r, c = divmod(idx,28)
            adv_img_28[r,c] = val_f

        print("ASCII view of changed pixels vs. original (X=changed):")
        ascii_lines = ascii_adversarial_with_changes(orig_img_28, adv_img_28)
        for line in ascii_lines:
            print(line)

    elif res=='unsat':
        print("No adv found => 'unsat' => try bigger patch or bigger eps.")
    else:
        print("Solver returned:", res)


if __name__=="__main__":
    main()
