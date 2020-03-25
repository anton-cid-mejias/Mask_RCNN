import keras.layers as KL
import keras.backend as K

from mrcnn import model

# Block that creates the graph which results in the
# probability of each bin and its residual angle values
def bin_block(input_tensor, angle_number, bin_number):
    name = "angle_%i_bin_%i" % (angle_number, bin_number)
    # Probability
    x = KL.TimeDistributed(KL.Dense(256), name='bin_classification_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='bin_classification_2')(x)
    bin_logits = KL.TimeDistributed(KL.Dense(2), name= name + '_logits')(x)
    bin_prob = KL.TimeDistributed(KL.Activation("softmax"),
                                    name= name + "_prob")(bin_logits)

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(256), name='bin_classification_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='bin_classification_2')(x)
    bin_res = KL.TimeDistributed(KL.Dense(2), name=name + '_res')(x)
    return bin_prob, bin_res


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
    masks: masks regressed.
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
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Concatenate the predicted masks to the feature maps
    x  = KL.Concatenate(axis=-1)(x, masks)

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
                       name="pool_squeeze")(x)

    # Add class probabilities
    shared =  KL.Concatenate(axis=-1)(shared, mrcnn_probs)

    # Add detected bounding box
    shared = KL.Concatenate(axis=-1)(shared, mrcnn_bbox)

    orientation = []
    for angle in range(0,3):
        angle_bins = []
        for bin in range(0,2):
            bin_prob, bin_res = bin_block(shared, angle, bin)
            angle_bins.append((bin_prob, bin_res))

        orientation.append(angle_bins)
    return orientation

# TODO: Reimplement to our type of data
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def orientation_loss_graph(target_bins, target_res_values, pred_orientation):
    """Loss for Mask R-CNN orientation regression.
    """
    loss = 0
    for angle_number in range(len(pred_orientation)):
        angle = pred_orientation[angle_number]
        for bin_number in range(0, 2):
            pred_bin = angle[bin_number][0]
            target_bin = target_bins[angle_number:angle_number+1]
            # Change it to logits?
            bin_loss = K.sparse_categorical_crossentropy(target=target_bin,
                                                         output=pred_bin)
            # TODO
            # Add L1 error as the difference between the target and predicted angle residues


    return loss