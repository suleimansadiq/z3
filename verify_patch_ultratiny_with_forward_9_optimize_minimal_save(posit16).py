#!/usr/bin/env python3
"""
VERIFY_PATCH_ULTRATINY_WITH_FORWARD_9_BEFORE_AFTER.PY

1) Loads a tf.posit16 ultra-tiny MLP from 'posit16_ultratinymlp.ckpt'.
2) Picks digit=9 from MNIST test set, prints ASCII of original + classification.
3) Defines a bottom patch => rows=20..27, cols=0..27 => ±0.1,
   uses Z3.Optimize to minimize sum(|x_new - x_orig|).
4) If sat => build adv image => print ASCII of adv => re-feed to MLP => new classification.

Hence you get:
 - original ASCII + class
 - post-attack ASCII + new class

Author: Suleiman Sadiq
"""

import sys, os
import numpy as np
import tensorflow as tf
from z3 import Optimize, Real, Bool, Or, And, Implies, Not, sat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def z3_value_to_float(z3_val):
    val_str = str(z3_val)
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num)/float(den)
    else:
        if '?' in val_str:
            val_str = val_str.split('?')[0]
        return float(val_str)

def ascii_digit(img_28):
    """
    ASCII of a 28x28 [-1..1] image with 4 levels
    """
    lines = []
    for r in range(28):
        row_str = ""
        for c in range(28):
            val = img_28[r,c]
            if val < -0.5:
                row_str += '.'
            elif val < 0:
                row_str += ':'
            elif val < 0.5:
                row_str += '+'
            else:
                row_str += '#'
        lines.append(row_str)
    return lines

def ascii_changed_pixels(orig_img, adv_img, diff_thresh=1e-5):
    """
    Mark 'X' if changed, else 3-level brightness of adv pixel
    """
    lines = []
    for r in range(28):
        row_str = ""
        for c in range(28):
            diff = abs(adv_img[r,c] - orig_img[r,c])
            if diff > diff_thresh:
                row_str += 'X'
            else:
                val = adv_img[r,c]
                if val < -0.5:
                    row_str += '.'
                elif val < 0:
                    row_str += ':'
                elif val < 0.5:
                    row_str += '+'
                else:
                    row_str += '#'
        lines.append(row_str)
    return lines


def main():
    if len(sys.argv)>1:
        data_t = sys.argv[1]
    else:
        data_t = "float32"
    print("data_t =", data_t)

    CKPT_NAME = data_t + "_ultratinymlp.ckpt"
    CKPT_PATH = os.path.join(".", CKPT_NAME)
    print("Checkpoint path:", CKPT_PATH)

    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.posit16, shape=(None,28,28,1), name="x_ph")

    def flatten_28x28(x):
        return tf.reshape(x, [-1,784])

    def ultra_tiny_net(x):
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

    main_vars = {}
    for v in tf.global_variables():
        nm = v.name.split(":")[0]
        if nm in ["Variable","Variable_1","Variable_2","Variable_3"]:
            main_vars[nm] = v
    saver = tf.train.Saver(var_list=main_vars)

    # Load MNIST => pick digit=9
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    idx_9 = None
    for i,lab in enumerate(y_test):
        if lab==9:
            idx_9 = i
            break
    if idx_9 is None:
        print("No digit=9 found.")
        return

    print("Test image => idx=", idx_9, " label=", y_test[idx_9])
    orig_img_28 = X_test[idx_9].astype(np.float32)
    orig_img_28 = (orig_img_28 - 127.5)/127.5
    test_img = orig_img_28[None,:,:,None]

    # Session => original classification
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, CKPT_PATH)
        orig_logits, orig_pred = sess.run([logits_float, pred_tf],
                                          feed_dict={x_ph: test_img})

        # also store W1_val, b1_val, W2_val, b2_val
        W1_val = sess.run(var_tuple[0])
        b1_val = sess.run(var_tuple[1])
        W2_val = sess.run(var_tuple[2])
        b2_val = sess.run(var_tuple[3])

    print("Original logits:", orig_logits)
    print("Original predicted label =>", orig_pred[0])

    # Print ASCII "before"
    print("\nASCII of original digit=9 image =>\n")
    before_lines = ascii_digit(orig_img_28)
    for line in before_lines:
        print(line)

    # Now do Z3.Optimize => patch => [20..27, 0..27], eps=0.1 => minimal sum of abs diffs
    opt = Optimize()
    x_vars = [ Real(f"x_{i}") for i in range(784) ]

    r0,r1 = 20,27
    c0,c1 = 0,27
    eps = 0.1

    abs_diffs = []
    def flatten_idx(r,c): return r*28 + c

    # build constraints
    for r in range(28):
        for c in range(28):
            idx = flatten_idx(r,c)
            val_orig = float(orig_img_28[r,c])
            if r0<=r<=r1 and c0<=c<=c1:
                opt.add(x_vars[idx] >= val_orig - eps)
                opt.add(x_vars[idx] <= val_orig + eps)
                dvar = Real(f"absdiff_{idx}")
                opt.add(dvar >= x_vars[idx] - val_orig,
                        dvar >= val_orig - x_vars[idx])
                abs_diffs.append(dvar)
            else:
                opt.add(x_vars[idx] == val_orig)

    # hidden => 8 reals
    hidden_vars = []
    for i in range(8):
        lin_expr = float(b1_val[i])
        for j in range(784):
            lin_expr += float(W1_val[j,i])* x_vars[j]
        hv = Real(f"hidden_{i}")
        # piecewise ReLU
        rb = Bool(f"relu_{i}")
        opt.add((lin_expr >= 0) == rb)
        opt.add(Implies(rb, hv==lin_expr))
        opt.add(Implies(Not(rb), hv==0))
        opt.add(hv>=0, hv>=lin_expr)
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

    # misclass => ∃ k!=9 => logit[k]>logit[9]
    logit9 = logits_vars[9]
    or_list = []
    for k in range(10):
        if k!=9:
            or_list.append(logits_vars[k] > logit9)
    opt.add(Or(or_list))

    cost = Real("cost")
    opt.add(cost == sum(abs_diffs))
    handle = opt.minimize(cost)

    print("\nOptimize => searching minimal shift in [20..27,0..27], eps=0.1")
    res = opt.check()
    print("Solver result:", res)
    if res==sat:
        print("Found minimal adversarial => digit=9 misclassified.")
        min_cost_str = str(opt.lower(handle))
        print("Minimal L1 shift =>", min_cost_str)

        m = opt.model()
        adv_img_28 = np.zeros((28,28), dtype=np.float32)
        for idx in range(784):
            zval = m[x_vars[idx]]
            val_f = z3_value_to_float(zval)
            rr, cc = divmod(idx,28)
            adv_img_28[rr,cc] = val_f

        # "After" ASCII => which pixels changed
        print("\nASCII of adv image => with X where changed from original:\n")
        after_lines = ascii_changed_pixels(orig_img_28, adv_img_28, diff_thresh=1e-5)
        for line in after_lines:
            print(line)

        # Also classify adv_img => new label
        adv_batch = adv_img_28[None,:,:,None]  # shape(1,28,28,1), float32
        # we re-run in a tf session
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            saver.restore(sess2, CKPT_PATH)
            adv_logits, adv_pred = sess2.run([logits_float, pred_tf],
                                             feed_dict={x_ph: adv_batch})
        print("\nAdversarial forward pass logits:", adv_logits)
        print("Adversarial predicted label =>", adv_pred[0])

    else:
        print("No solution =>", res)
        print("Try bigger eps or different patch.")


if __name__=="__main__":
    main()
