import keras.layers as KL
import keras.backend as K
import math
import tensorflow as tf
import numpy as np
from mrcnn import or_tools
from mrcnn import model

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
        r_matrices: [batch, 3, 3] rotation matrices
        angles: [batch,, 3] rotation angles in Euler ZYX (radians)
    """

    x = model.PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_orientation")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_or_conv1")(x)
    x = KL.TimeDistributed(model.BatchNorm(), name='mrcnn_or_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_or_conv2")(x)
    x = KL.TimeDistributed(model.BatchNorm(), name='mrcnn_or_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze_or")(x)

    # Add class probabilities
    x = KL.Concatenate(axis=2)([x, mrcnn_probs])

    # Add detected bounding box
    s = K.int_shape(mrcnn_bbox)
    mrcnn_bbox = KL.Reshape((s[1], s[2] * s[3]))(mrcnn_bbox)
    x = KL.Concatenate(axis=2)([x, mrcnn_bbox])

    x = KL.TimeDistributed(KL.Dense(1024), name='mrcnn_or_d1')(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_or_bn3')(x, training=train_bn)
    x = KL.TimeDistributed(KL.Dense(1024), name='mrcnn_or_d2')(x)
    x = KL.LeakyReLU(alpha=0.2)(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_or_bn4')(x, training=train_bn)
    x = KL.TimeDistributed(KL.Dense(6), name='mrcnn_or_d5')(x)

    #s = K.int_shape(x)
    #x = KL.Lambda(lambda t: tf.reshape(t, (-1, s[2])))(x)

    r_matrices = KL.TimeDistributed(KL.Lambda(lambda t: or_tools.compute_rotation_matrix_from_ortho6d(t)))(x)
    #r_matrices = KL.TimeDistributed(KL.Reshape((-1, s[1], 3, 3))(r_matrices))
    angles = KL.TimeDistributed(KL.Lambda(lambda  x: or_tools.compute_euler_angles_from_rotation_matrices(x)))(r_matrices)
    #angles = KL.Reshape((-1, s[1], 3))(angles)

    return r_matrices, angles

def deg2rad(angles):
    return angles * np.pi / 180

def orientation_loss_graph(target_orientations, target_class_ids, pred_matrices):
    """Loss for Mask R-CNN orientation regression.
    """
    # Remove batch dimension for simplicity
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_orientations = K.reshape(target_orientations, (-1, 3))
    pred_matrices = K.reshape(pred_matrices, (-1, 3, 3))

    # Only positive ROIs contribute to the loss.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # Gather the rois that contribute to the loss
    target_orientations = tf.gather(target_orientations, positive_roi_ix)
    pred_matrices = tf.gather(pred_matrices, positive_roi_ix)

    target_orientations = deg2rad(target_orientations)
    target_matrices = or_tools.from_euler(target_orientations)

    thetas = or_tools.compute_geodesic_distance_from_two_matrices(target_matrices, pred_matrices)
    loss = K.mean(thetas)

    return loss

