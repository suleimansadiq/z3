#!/usr/bin/env python3
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image

# Z3
from z3 import Real, Optimize, If, Or, sat

##############################################################################
# 1) Parse dtype (posit8 or float32)
##############################################################################
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if data_t == 'posit8':
        tf_type = tf.posit8
    elif data_t == 'posit16':
        tf_type = tf.posit16
    elif data_t == 'posit32':
        tf_type = tf.posit32
    else:
        data_t = 'float32'
        tf_type = tf.float32
        print("Unrecognized type, defaulting float32.")
else:
    data_t = 'float32'
    tf_type = tf.float32

print(f"Using TensorFlow dtype => {data_t} ({tf_type})")

##############################################################################
# 2) Flatten using tf.reshape
##############################################################################
def flatten(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])

##############################################################################
# 3) Define LeNet
##############################################################################
def LeNet(x):
    mu=0.0
    sigma=0.1

    conv1_W = tf.Variable(tf.truncated_normal((5,5,1,6), mean=mu, stddev=sigma, dtype=tf_type), name='Variable')
    conv1_b = tf.Variable(tf.zeros([6], dtype=tf_type), name='Variable_1')
    conv1   = tf.nn.conv2d(x, conv1_W, [1,1,1,1], 'VALID') + conv1_b
    conv1   = tf.nn.relu(conv1)
    pool1   = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], 'VALID')

    conv2_W = tf.Variable(tf.truncated_normal((5,5,6,16), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_2')
    conv2_b = tf.Variable(tf.zeros([16], dtype=tf_type), name='Variable_3')
    conv2   = tf.nn.conv2d(pool1, conv2_W, [1,1,1,1], 'VALID') + conv2_b
    conv2   = tf.nn.relu(conv2)
    pool2   = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'VALID')

    fc0     = flatten(pool2)  # => [?,400]

    fc1_W = tf.Variable(tf.truncated_normal((400,120), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_4')
    fc1_b = tf.Variable(tf.zeros([120], dtype=tf_type), name='Variable_5')
    fc1   = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)

    fc2_W = tf.Variable(tf.truncated_normal((120,84), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_6')
    fc2_b = tf.Variable(tf.zeros([84], dtype=tf_type), name='Variable_7')
    fc2   = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

    fc3_W = tf.Variable(tf.truncated_normal((84,10), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_8')
    fc3_b = tf.Variable(tf.zeros([10], dtype=tf_type), name='Variable_9')
    logits= tf.matmul(fc2, fc3_W) + fc3_b
    return logits

##############################################################################
# 4) Restore
##############################################################################
x_ph = tf.placeholder(tf_type, (None,32,32,1))
logits = LeNet(x_ph)
saver = tf.train.Saver()
ckpt_path = './posit8.ckpt'

with tf.Session() as sess:
    saver.restore(sess, ckpt_path)
    weights_and_biases = {}
    for v in tf.trainable_variables():
        arr = sess.run(v)
        weights_and_biases[v.name] = arr
        print(f"Loaded {v.name}, shape={arr.shape}, dtype={arr.dtype}")

print("\nAll weights/biases loaded. We'll free ONLY final-layer weights (NOT biases).\n")

##############################################################################
# 5) Load a digit '1' => pad => 32x32
##############################################################################
(_, _), (X_test, y_test) = mnist.load_data()

def find_label_1(x_d, y_d):
    for i in range(len(x_d)):
        if y_d[i] == 1:
            return i
    return 0

idx_1 = find_label_1(X_test, y_test)
lab = y_test[idx_1]
print(f"Using MNIST test image index={idx_1}, label={lab}")

img28 = X_test[idx_1]
pil_img = Image.fromarray(img28)
img32  = pil_img.resize((32,32), Image.LANCZOS)
arr32  = np.array(img32, dtype=np.float32)
arr32  = (arr32 - 127.5)/127.5
arr32  = arr32.reshape((1,32,32,1))

##############################################################################
# 6) Python forward pass up to fc2 => shape=(84)
##############################################################################
def relu_np(x):
    return np.maximum(x, 0)

def conv2d_valid_str1(image, W, b):
    (fH, fW, inC, outC) = W.shape
    (H, W_, C_) = image.shape
    outH = H - fH + 1
    outW = W_ - fW + 1
    out = np.zeros((outH, outW, outC), dtype=np.float32)
    for oh in range(outH):
        for ow in range(outW):
            for oc in range(outC):
                val = b[oc]
                for rr in range(fH):
                    for cc in range(fW):
                        for ch in range(inC):
                            val += image[oh+rr, ow+cc, ch]*W[rr,cc,ch,oc]
                out[oh,ow,oc] = val
    return out

def maxpool2x2_np(img):
    (H,W,C) = img.shape
    outH = H//2
    outW = W//2
    out = np.zeros((outH, outW, C), dtype=np.float32)
    for i in range(outH):
        for j in range(outW):
            for c in range(C):
                patch = [
                    img[2*i,   2*j,   c],
                    img[2*i+1, 2*j,   c],
                    img[2*i,   2*j+1, c],
                    img[2*i+1, 2*j+1, c]
                ]
                out[i,j,c] = max(patch)
    return out

def forward_up_to_fc2(x32_1x, wb):
    x = x32_1x[0]  # shape(32,32,1)

    c1W = wb["Variable:0"]     # (5,5,1,6)
    c1b = wb["Variable_1:0"]
    outc1 = conv2d_valid_str1(x, c1W, c1b)
    outc1 = relu_np(outc1)                # => (28,28,6)
    pool1 = maxpool2x2_np(outc1)          # => (14,14,6)

    c2W = wb["Variable_2:0"]     # (5,5,6,16)
    c2b = wb["Variable_3:0"]
    outc2 = conv2d_valid_str1(pool1, c2W, c2b) # => (10,10,16)
    outc2 = relu_np(outc2)
    pool2 = maxpool2x2_np(outc2)               # => (5,5,16) => flatten => 400

    flat400 = pool2.reshape((1,400))

    fc1W = wb["Variable_4:0"]    # (400,120)
    fc1b = wb["Variable_5:0"]
    fc1_ = np.dot(flat400, fc1W) + fc1b
    fc1_ = relu_np(fc1_)                 # (1,120)

    fc2W = wb["Variable_6:0"]    # (120,84)
    fc2b = wb["Variable_7:0"]
    fc2_ = np.dot(fc1_, fc2W) + fc2b
    fc2_ = relu_np(fc2_)                # (1,84)

    return fc2_[0]  # shape= (84,)

fc2_out = forward_up_to_fc2(arr32, weights_and_biases)
print("fc2_out shape=84 => example first 5:", fc2_out[:5])

##############################################################################
# 7) Freed final-layer WEIGHTS only => pinned final-layer biases
##############################################################################
opt = Optimize()

def final_out(digit, FreedW, pinnedBias, fc2acts):
    expr = pinnedBias[digit]
    for i in range(len(fc2acts)):
        expr += FreedW[(i,digit)]*fc2acts[i]
    return expr

fc3_w_orig = weights_and_biases["Variable_8:0"]  # shape=(84,10)
fc3_b_orig = weights_and_biases["Variable_9:0"]  # shape=(10,)
pinnedBias = fc3_b_orig  # we do NOT free bias => pinned

FreedW = {}
abs_vars = []

for i in range(84):
    for o in range(10):
        nm_w  = f"fc3W_{i}_{o}"
        nm_dw = f"d_fc3W_{i}_{o}"
        nm_aw = f"abs_fc3W_{i}_{o}"

        wvar = Real(nm_w)
        dvar = Real(nm_dw)
        awar = Real(nm_aw)

        base_val = float(fc3_w_orig[i,o])
        # wvar = base_val + dvar
        opt.add(wvar == base_val + dvar)
        # absolute val
        opt.add(awar >= dvar, awar >= -dvar)

        FreedW[(i,o)] = wvar
        abs_vars.append(awar)

# classification => final_1 not largest => or( final(k)> final(1) for k!=1 )
ors = []
for k in range(10):
    if k!=1:
        ors.append(final_out(k, FreedW, pinnedBias, fc2_out) > final_out(1, FreedW, pinnedBias, fc2_out))
opt.add(Or(*ors))

# cost => sum of all |dvar|
cost = Real("cost")
opt.add(cost == sum(abs_vars))
h = opt.minimize(cost)

print("\nOnly final-layer weights are freed (no bias shift). Minimizing L1 shift so '1' not top...")

res = opt.check()
if res == sat:
    print("SAT: Found a final-layer weight shift (only W) that breaks digit=1.")
    print("Minimal L1 shift =>", opt.lower(h))

    model = opt.model()
    cost_val = model[cost]
    print("Cost in model =>", cost_val)

    ##########################################################################
    # Freed Weights That Actually Changed
    ##########################################################################
    threshold = 1e-9
    print("\n*** Freed Final Weights That Changed ***")
    changed_any = False

    def z3_numeral_to_float(z3val):
        val_str = str(z3val)
        if '/' in val_str:
            # fraction form
            num, den = val_str.split('/')
            return float(num)/float(den)
        else:
            # try as_decimal
            try:
                dec_str = str(z3val.as_decimal(50))
                if '?' in dec_str:
                    dec_str = dec_str[: dec_str.index('?')]
                return float(dec_str)
            except:
                return float(val_str)

    for i in range(84):
        for o in range(10):
            w_new_z3 = model[FreedW[(i,o)]]
            w_orig   = fc3_w_orig[i,o]

            w_new_float = z3_numeral_to_float(w_new_z3)
            delta = w_new_float - w_orig
            if abs(delta) > threshold:
                changed_any = True
                print(f"fc3W_{i}_{o}: orig={w_orig:.6f}, new={w_new_float:.6f}, shift={delta:.6f}")
    if not changed_any:
        print("No final W changed by more than threshold.")

    ##########################################################################
    # 8) Post-solution check => see new classification
    ##########################################################################
    # Rebuild final-layer W with the found solution
    fc3W_modified = np.array(fc3_w_orig, copy=True)
    for i in range(84):
        for o in range(10):
            w_new_z3 = model[ FreedW[(i,o)] ]
            w_new_float = z3_numeral_to_float(w_new_z3)
            fc3W_modified[i,o] = w_new_float

    fc3b_modified = fc3_b_orig  # pinned

    # new final pass => 10 logits
    new_logits = np.dot(fc2_out, fc3W_modified) + fc3b_modified
    print("\nNew final logits after SHIFT in final-layer weights =>", new_logits)
    new_pred = np.argmax(new_logits)
    print(f"Argmax => digit={new_pred}. (Originally was 1 => misclassification if new_pred != 1)")

    # Also see which digits > final_out(1)
    final_1_val = new_logits[1]
    bigger_digits = []
    for d in range(10):
        if d != 1 and new_logits[d] > final_1_val:
            bigger_digits.append(d)
    print(f"final_out(1) = {final_1_val:.6f}, digits with logit> final_out(1): {bigger_digits}")

else:
    print("Result:", res)
