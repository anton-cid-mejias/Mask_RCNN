import keras.layers as KL
import keras.backend as K
import math
import tensorflow as tf

from mrcnn import model

# Block that creates the graph which results in the
# probability of each bin and its residual angle values
def bin_block(input_tensor, angle_number, bin_number):
    name = "angle_%i_bin_%i" % (angle_number, bin_number)
    # Probability
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_bin_classification_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_bin_classification_2')(x)
    bin_logits = KL.TimeDistributed(KL.Dense(2), name= 'mrcnn_' + name + '_logits')(x)
    bin_prob = KL.TimeDistributed(KL.Activation("softmax"),
                                    name= "mrcnn_" + name + "_prob")(bin_logits)

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_'+ name + '_bin_res_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + '_bin_res_2')(x)
    bin_res = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res')(x)

    return KL.Concatenate(axis=2)([bin_logits, bin_prob, bin_res])


def fpn_orientation_graph(rois, feature_maps, mrcnn_probs,
                          mrcnn_bbox, masks, image_meta,
                         pool_size, train_bn=True):
    """Builds the computation graph of the feature pyramid network orientation
     heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    mrcnn_probs: classifier probabilities.
    mrcnn_bbox: Deltas to apply to proposal boxes
    masks: masks regressed. [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = model.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_orientation")([rois, image_meta] + feature_maps)

    masks = KL.TimeDistributed(KL.MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid'))(masks)

    # Concatenate the predicted masks to the feature maps
    x = KL.Concatenate(axis=4)([x,masks])

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
    #newdim = [-1, mrcnn_bbox.shape[1], mrcnn_bbox.shape[2] * mrcnn_bbox.shape[3]]
    s = K.int_shape(mrcnn_bbox)
    mrcnn_bbox = KL.Reshape((s[1], s[2] * s[3]))(mrcnn_bbox)
    shared = KL.Concatenate(axis=2)([shared, mrcnn_bbox])

    outputs = []
    for angle in range(0,3):
        for bin in range(0,2):
            #bin_logits, bin_prob, bin_res = bin_block(shared, angle, bin)
            output = bin_block(shared, angle, bin)
            outputs.append(output)

    orientation = KL.Concatenate(axis=2)(outputs)
    return orientation

# 2 Bins
# First [-210, 30], middle point -90
# Second [-30, 210], middle point 90
def get_transformed_orientations(orientations):
    first_bin = tf.constant([-210., 30., -90.])
    second_bin = tf.constant([-30., 210., 90.])
    deg2rad = math.pi / 180

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
        cos1 = tf.math.cos(res)
        sin1 = tf.math.sin(res)
        # Bin 2
        less_than1 = angles <= second_bin[1]
        less_than2 = angles >= second_bin[0]
        # Probability 2
        prob2 = tf.cast(tf.math.logical_and(less_than1, less_than2), dtype=tf.float32)
        # Residues 2
        res = angles - second_bin[2]
        res = res * deg2rad
        cos2 = tf.math.cos(res)
        sin2 = tf.math.sin(res)
        r.append(tf.stack([prob1, sin1, cos1, prob2, sin2, cos2], axis=1))

    return tf.reshape(tf.stack(r, axis=1), (-1, 18))

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def orientation_loss_graph(target_orientations, target_class_ids, pred_orientation):
    """Loss for Mask R-CNN orientation regression.
    """
    # Remove batch dimension for simplicity
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_orientations = K.reshape(target_orientations, (-1, 3))
    pred_orientation = K.reshape(pred_orientation, (-1, 36))

    # Only positive ROIs contribute to the loss.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # Gather the rois that contribute to the loss
    target_orientations = tf.gather(target_orientations, positive_roi_ix)
    pred_orientation = tf.gather(pred_orientation, positive_roi_ix)

    target_orientations = get_transformed_orientations(target_orientations)

    # Iterate over each of the bins
    losses = []
    for i in range(0, 6):
        # target_orientation bin: [pos, neg, sin, cos]
        target_bin_prob = target_orientations[:, i*3]
        # target_orientation bin: [pos_logit, neg_logit, pos, neg, sin, cos]
        pred_bin_logits = pred_orientation[:, i*6: i*6 + 2]

        softmax_loss = K.sparse_categorical_crossentropy(target=target_bin_prob,
                                                     output=pred_bin_logits,
                                                     from_logits=True)
        softmax_loss = K.mean(softmax_loss)

        target_bin_res = target_orientations[:, i*3 + 1: i*3 + 3]
        pred_bin_res = pred_orientation[:, i*6 + 4: i*6 + 6]
        l1_loss = smooth_l1_loss(target_bin_res, pred_bin_res)
        l1_loss = K.mean((l1_loss[:, 0] + l1_loss[:, 1]) * target_bin_prob)

        losses.append(softmax_loss + l1_loss)

    loss = tf.math.add_n(losses) / 6

    return loss