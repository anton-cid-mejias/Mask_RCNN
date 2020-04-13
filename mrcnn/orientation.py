import keras.layers as KL
import keras.backend as K
import math
import tensorflow as tf
import numpy as np
from keras import losses

from mrcnn import model

# Block that creates the graph which results in the
# probability of each bin and its residual angle values
def bin_block(input_tensor, angle_number, bin_number, train_bn):
    name = "angle_%i_bin_%i" % (angle_number, bin_number)
    # Probability
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_class_1')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_class_2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    bin_logits = KL.TimeDistributed(KL.Dense(2), name= 'mrcnn_' + name + '_logits')(x)
    bin_prob = KL.TimeDistributed(KL.Activation("softmax"),
                                    name= "mrcnn_" + name + "_prob")(bin_logits)

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_'+ name + '_res_1')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_res_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_res_2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_res_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    bin_res = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res')(x)

    return bin_logits, bin_prob, bin_res

def angle_block(input_tensor, angle_number, train_bn):
    name = "angle_%i" % angle_number
    # Probability
    x = KL.TimeDistributed(KL.Dense(512), name='mrcnn_' + name + '_class_1')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_class_2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 4 bins for each angle
    # Bin 1: [-210º, -60º] -> middle = -135º
    # Bin 2: [-120º, 30º] -> middle = -45º
    # Bin 2: [-30º, 120º] -> middle = 45º
    # Bin 2: [60º, 210º] -> middle = 135º
    # 2 outputs for each bin softmax(included in this bin?)
    bin_logits_1 = KL.TimeDistributed(KL.Dense(2), name= 'mrcnn_' + name + '_logits_1')(x)
    bin_prob_1 = KL.TimeDistributed(KL.Activation("softmax"),
                                    name= "mrcnn_" + name + "_prob_1")(bin_logits_1)
    bin_logits_2 = KL.TimeDistributed(KL.Dense(2), name='mrcnn_' + name + '_logits_2')(x)
    bin_prob_2 = KL.TimeDistributed(KL.Activation("softmax"),
                                  name="mrcnn_" + name + "_prob_2")(bin_logits_2)
    bin_logits_3 = KL.TimeDistributed(KL.Dense(2), name='mrcnn_' + name + '_logits_3')(x)
    bin_prob_3 = KL.TimeDistributed(KL.Activation("softmax"),
                                  name="mrcnn_" + name + "_prob_3")(bin_logits_3)
    bin_logits_4 = KL.TimeDistributed(KL.Dense(2), name='mrcnn_' + name + '_logits_4')(x)
    bin_prob_4 = KL.TimeDistributed(KL.Activation("softmax"),
                                  name="mrcnn_" + name + "_prob_4")(bin_logits_4)
    bin_logits = KL.Concatenate(axis=2)([bin_logits_1, bin_logits_2, bin_logits_3, bin_logits_4])
    bin_prob = KL.Concatenate(axis=2)([bin_prob_1, bin_prob_2, bin_prob_3, bin_prob_4])

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(512), name='mrcnn_'+ name + '_res')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_res_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_res2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_' + name + '_res_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 4 bins for aech angle
    # 2 outputs for each bin (sin, cos)
    bin_res_1 = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res_1')(x)
    bin_res_1 = KL.Lambda(lambda x: K.l2_normalize(x, axis=2))(bin_res_1)
    bin_res_2 = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res_2')(x)
    bin_res_2 = KL.Lambda(lambda x: K.l2_normalize(x, axis=2))(bin_res_2)
    bin_res_3 = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res_3')(x)
    bin_res_3 = KL.Lambda(lambda x: K.l2_normalize(x, axis=2))(bin_res_3)
    bin_res_4 = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res_4')(x)
    bin_res_4 = KL.Lambda(lambda x: K.l2_normalize(x, axis=2))(bin_res_4)
    bin_res = KL.Concatenate(axis=2)([bin_res_1, bin_res_2, bin_res_3, bin_res_4])

    return bin_logits, bin_prob, bin_res

def full_block(input_tensor, train_bn):
    # Probability
    x = KL.TimeDistributed(KL.Dense(512), name='mrcnn_class_or_1')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_or_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_class_or_2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_or_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(24), name='mrcnn_class_or_3')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_or_bn3')(x, training=train_bn)
    # 4 bins for each angle
    # Bin 1: [-210º, -60º] -> middle = -135º
    # Bin 2: [-120º, 30º] -> middle = -45º
    # Bin 2: [-30º, 120º] -> middle = 45º
    # Bin 2: [60º, 210º] -> middle = 135º
    # 2 outputs for each bin softmax(included in this bin?)
    bin_logits_list = []
    bin_probs_list = []
    for angle in range(0, 3):
        for bin in range(0, 4):
            name = "%i_%i" % (angle, bin)
            position = angle * 8 + bin * 2
            bin_logits = KL.Lambda(lambda x : x[:, :, position:position + 2])(x)
            #bin_logits = KL.TimeDistributed(KL.Dense(2), name="mrcnn_" + name + '_prob')(bin_logits)
            bin_prob = KL.TimeDistributed(KL.Activation("softmax"),
                                            name= "mrcnn_" + name + "_prob")(bin_logits)
            bin_logits_list.append(bin_logits)
            bin_probs_list.append(bin_prob)

    bin_logits = KL.Concatenate(axis=2)(bin_logits_list)
    bin_prob = KL.Concatenate(axis=2)(bin_probs_list)

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(512), name='mrcnn_res1')(input_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_res_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_res2')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_res_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Dense(24), name='mrcnn_res3')(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_res_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 4 bins for aech angle
    # 2 outputs for each bin (sin, cos)
    bin_res_list = []
    for angle in range(0, 3):
        for bin in range(0, 4):
            name = "%i_%i" % (angle, bin)
            position = angle * 8 + bin * 2
            bin_res = KL.Lambda(lambda x : x[:, :, position:position + 2])(x)
            bin_res = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res')(bin_res)
            bin_res = KL.Lambda(lambda x: K.l2_normalize(x, axis=2))(bin_res)
            bin_res_list.append(bin_res)

    bin_res = KL.Concatenate(axis=2)(bin_res_list)

    return bin_logits, bin_prob, bin_res

def fpn_orientation_graph(rois, feature_maps, mrcnn_probs,
                          mrcnn_bbox, image_meta,
                         pool_size, train_bn=True):
    """Builds the computation graph of the feature pyramid network orientation
     heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    mrcnn_probs: classifier probabilities.
    mrcnn_bbox: Deltas to apply to proposal boxes
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = model.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_orientation")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    # First layer
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_orientation_conv1")(x)
    x = KL.TimeDistributed(model.BatchNorm(), name='mrcnn_orientation_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # Second layer
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_orientation_conv2")(x)
    x = KL.TimeDistributed(model.BatchNorm(), name='mrcnn_orientation_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Squeezed feature maps
    # [batch, num_rois, fc_layers_size]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze_orientation")(x)

    # Add class probabilities
    shared = KL.Concatenate(axis=2)([shared, mrcnn_probs])

    # Add detected bounding box
    s = K.int_shape(mrcnn_bbox)
    mrcnn_bbox = KL.Reshape((s[1], s[2] * s[3]))(mrcnn_bbox)
    shared = KL.Concatenate(axis=2)([shared, mrcnn_bbox])

    #logits = []
    #probs = []
    #res = []
    '''
    for angle in range(0,3):
        for bin in range(0,2):
            bin_logits, bin_prob, bin_res = bin_block(shared, angle, bin, train_bn)
            logits.append(bin_logits)
            probs.append(bin_prob)
            res.append(bin_res)
    '''
    '''
    for angle in range(0,3):
        bin_logits, bin_prob, bin_res = angle_block(shared, angle, train_bn)
        logits.append(bin_logits)
        probs.append(bin_prob)
        res.append(bin_res)

    logits = KL.Concatenate(axis=2)(logits)
    probs = KL.Concatenate(axis=2)(probs)
    res = KL.Concatenate(axis=2)(res)
    '''
    logits, probs, res =  full_block(shared, train_bn)

    return logits, probs, res


# 2 Bins
# First [-210, 30], middle point -90
# Second [-30, 210], middle point 90
def get_transformed_orientations_2d(orientations):
    first_bin = tf.constant([-210., 30., -90.])
    second_bin = tf.constant([-30., 210., 90.])
    deg2rad = tf.constant(math.pi / 180)

    r = []
    for i in range(0, K.int_shape(orientations)[1]):
        # Bin 1
        angles = orientations[:, i]
        less_than1 = angles <= first_bin[1]
        less_than2 = angles >= first_bin[0]
        # Probability 1
        prob1 = tf.cast(tf.math.logical_and(less_than1, less_than2), dtype=tf.float32)
        # Residues 1
        res = angles - first_bin[2]
        res = res * deg2rad
        sin1 = tf.math.sin(res)
        cos1 = tf.math.cos(res)
        # Bin 2
        less_than1 = angles <= second_bin[1]
        less_than2 = angles >= second_bin[0]
        # Probability 2
        prob2 = tf.cast(tf.math.logical_and(less_than1, less_than2), dtype=tf.float32)
        # Residues 2
        res = angles - second_bin[2]
        res = res * deg2rad
        sin2 = tf.math.sin(res)
        cos2 = tf.math.cos(res)
        r.append(tf.stack([prob1, sin1, cos1, prob2, sin2, cos2], axis=1))

    return tf.concat(r, axis=1)

def check_bin(bin_angles, angles):
    deg2rad = tf.constant(math.pi / 180)
    # Bin 1
    less_than1 = angles >= bin_angles[0]
    less_than2 = angles <= bin_angles[1]
    # Probability 1
    prob = tf.cast(tf.math.logical_and(less_than1, less_than2), dtype=tf.float32)
    # Residues 1
    # Calculates the angle that is necessary to sum to the middle point of the bin
    # in order to reach the desired angle
    res = angles - bin_angles[2]
    res = res * deg2rad
    sin = tf.math.sin(res)
    cos = tf.math.cos(res)
    return prob, sin, cos

# 4 Bins for each angle
# Bin 1: [-210º, -60º] -> middle = -135º
# Bin 2: [-120º, 30º] -> middle = -45º
# Bin 2: [-30º, 120º] -> middle = 45º
# Bin 2: [60º, 210º] -> middle = 135º
def get_transformed_orientations_4d(orientations):
    bins = tf.constant([[-210., -60., -135.],
                        [-120., 30., -45.],
                        [-30., 120., 45.],
                        [60., 210., 135.]])

    r = []
    for i in range(0, K.int_shape(orientations)[1]):
        angles = orientations[:, i]
        bin_outputs = []
        for bin_number in range(0, 4):
            prob, sin, cos = check_bin(bins[bin_number], angles)
            bin_outputs.extend([prob, sin, cos])
        r.append(tf.stack(bin_outputs, axis=1))

    return tf.concat(r, axis=1)

def orientation_loss_graph(target_orientations, target_class_ids, pred_logits, pred_res):
    """Loss for Mask R-CNN orientation regression.
    """
    # Remove batch dimension for simplicity
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_orientations = K.reshape(target_orientations, (-1, 3))
    pred_logits = K.reshape(pred_logits, (-1, 24))
    pred_res = K.reshape(pred_res, (-1, 24))

    # Only positive ROIs contribute to the loss.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # Gather the rois that contribute to the loss
    target_orientations = tf.gather(target_orientations, positive_roi_ix)
    pred_logits = tf.gather(pred_logits, positive_roi_ix)
    pred_res = tf.gather(pred_res, positive_roi_ix)

    target_orientations = get_transformed_orientations_4d(target_orientations)

    # Iterate over each of the bins
    losses_list = []
    for i in range(0, 12):
        # target_orientation bin: [prob, sin, cos]
        target_bin_prob = target_orientations[:, i * 3]
        # target_orientation bin: [neg_logit, pos_logit, pos, neg, sin, cos]
        logits = pred_logits[:, i*2: i*2 + 2]

        class_loss = K.sparse_categorical_crossentropy(target=target_bin_prob,
                                                       output=logits,
                                                       from_logits=True)

        target_bin_res = target_orientations[:, i * 3 + 1: i * 3 + 3]
        pred_bin_res = pred_res[:, i * 2: i * 2 + 2]
        l2_loss = losses.mean_squared_error(target_bin_res, pred_bin_res)
        #l1_loss = losses.mean_absolute_error(target_bin_res, pred_bin_res)

        losses_list.append(K.mean(class_loss + l2_loss * target_bin_prob))

    loss = tf.math.add_n(losses_list) / 3

    return loss


def calculate_angle(sin, cos, middle):
    angle = math.degrees(np.arctan2(sin, cos)) + middle
    return angle


def unmold_orientations_2d(detections, mrcnn_or_prob, mrcnn_or_res):
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    if N <= 0:
        return []

    middle_angle = (-90, 90)

    or_prob = mrcnn_or_prob[np.arange(N), :]
    or_res = mrcnn_or_res[np.arange(N), :]
    angles = []
    for i in range(N):
        bins = []
        for j in range(0,3):
            included_1 = or_prob[i, j*4 + 1]
            included_2 = or_prob[i, j*4 + 3]
            res = or_res[i, j * 4: j *4 + 4]
            # First bin
            if included_1 > included_2:
                angle = calculate_angle(res[0], res[1], middle_angle[0])
            # Second bin
            else:
                angle = calculate_angle(res[2], res[3], middle_angle[1])
            bins.append(angle)
        bins = np.hstack(bins)
        angles.append(bins)
    angles = np.vstack(angles).tolist()
    return angles

def unmold_orientations_4d(detections, mrcnn_or_prob, mrcnn_or_res):
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    if N <= 0:
        return []

    middle_angle = (-135, -45, 45, 135)

    or_prob = mrcnn_or_prob[np.arange(N), :]
    or_res = mrcnn_or_res[np.arange(N), :]
    angles = []
    for i in range(N):
        bins = []
        for j in range(0,3):
            included_1 = or_prob[i, j*8 + 1]
            included_2 = or_prob[i, j*8 + 3]
            included_3 = or_prob[i, j*8 + 5]
            included_4 = or_prob[i, j*8 + 7]
            included = np.array([included_1, included_2, included_3, included_4])
            max_indice = np.argmax(included)

            res = or_res[i, j * 8: j * 8 + 8]
            # First bin
            angle = 0.
            if max_indice == 0:
                angle = calculate_angle(res[0], res[1], middle_angle[0])
            # Second bin
            elif max_indice == 1:
                angle = calculate_angle(res[2], res[3], middle_angle[1])
            # Third bin
            elif max_indice == 2:
                angle = calculate_angle(res[4], res[5], middle_angle[2])
            # Forth bin
            elif max_indice == 3:
                angle = calculate_angle(res[6], res[7], middle_angle[3])

            bins.append(angle)
        bins = np.hstack(bins)
        angles.append(bins)
    angles = np.vstack(angles).tolist()
    return angles


def unmold_orientation_test_2d(orientation):
    res_angle = (-90, 90)

    bins = []
    for j in range(0,6):
        bin = orientation[j*3:j*3 + 3]
        included = bin[0]
        angle = np.nan
        if included > 0.75:
            angle = calculate_angle(bin[1], bin[2], res_angle[j % 2])
        bins.append(angle)
    bins = np.hstack(bins)
    return bins

def unmold_orientation_test_4d(orientation):
    res_angle = (-135, -45, 45, 135)

    bins = []
    for j in range(0,12):
        bin = orientation[j*3:j*3 + 3]
        included = bin[0]
        angle = np.nan
        if included > 0.75:
            angle = calculate_angle(bin[1], bin[2], res_angle[j % 4])
        bins.append(angle)
    bins = np.hstack(bins)
    return bins

# Test
if __name__=="__main__":
    angles = tf.reshape(tf.constant([-209., 25., -59.]), (1, 3))
    transformed = get_transformed_orientations_4d(angles)
    sess = tf.compat.v1.Session()
    transformed = sess.run(transformed)
    print(transformed)

    unmolded = unmold_orientation_test_4d(transformed[0, :])
    print(unmolded)