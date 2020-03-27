import keras.layers as KL
import keras.backend as K
import math

from mrcnn import model

# 2 Bins
# First [-210, 30], middle point -90
# Second [-30, 210], middle point 90
def get_gt_orientations(orientations):
    first_bin = (-210, 30, -90)
    second_bin = (-30, 210, 90)
    gt_orientations = []
    for orientation in orientations:
        bin1 = [0, 0, 0]
        bin2 = [0, 0, 0]
        # Check first bin
        if first_bin[0] <= orientation <= first_bin[1]:
            res_angle = math.radians(orientation - first_bin[2])
            bin1 = [1, math.cos(res_angle), math.sin(res_angle)]
        # Check second bin
        if second_bin[0] <= orientation <= second_bin[1]:
            res_angle = math.radians(orientation - first_bin[2])
            bin2 = [1, math.cos(res_angle), math.sin(res_angle)]

        gt_orientations.append(tuple(bin1.extend(bin2)))

    return gt_orientations


# Block that creates the graph which results in the
# probability of each bin and its residual angle values
def bin_block(input_tensor, angle_number, bin_number):
    name = "angle_%i_bin_%i" % (angle_number, bin_number)
    # Probability
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + 'bin_classification_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + 'bin_classification_2')(x)
    bin_logits = KL.TimeDistributed(KL.Dense(2), name= name + '_logits')(x)
    bin_prob = KL.TimeDistributed(KL.Activation("softmax"),
                                    name= "mrcnn_" + name + "_prob")(bin_logits)

    # Residual angle
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_'+ name + 'bin_res_1')(input_tensor)
    x = KL.TimeDistributed(KL.Dense(256), name='mrcnn_' + name + 'bin_res_2')(x)
    bin_res = KL.TimeDistributed(KL.Dense(2), name= "mrcnn_" + name + '_res')(x)
    return bin_logits, bin_prob, bin_res


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
            bin_logits, bin_prob, bin_res = bin_block(shared, angle, bin)
            angle_bins.append((bin_logits, bin_prob, bin_res))

        orientation.append(angle_bins)
    return orientation

def l1_loss(y_true, y_pred, target_bin):
    loss = 0
    if target_bin == 1:
        loss = K.abs(y_true - y_pred)
    return loss

def orientation_loss_graph(target_bins, target_res_values, pred_orientation):
    """Loss for Mask R-CNN orientation regression.
    """
    loss = 0
    for angle_number in range(len(pred_orientation)):
        angle = pred_orientation[angle_number]
        for bin_number in range(0, 2):
            pred_logits, _, pred_res = angle[bin_number]
            target_bin = target_bins[angle_number + (bin_number * 2): angle_number + (bin_number * 2) + 1]
            bin_loss = K.sparse_categorical_crossentropy(target=target_bin,
                                                         output=pred_logits,
                                                         from_logits=True)
            # Add L1 error as the difference between the target and predicted angle residues
            target_res = target_res_values[angle_number + (bin_number * 2): angle_number + (bin_number * 2) + 1]
            res_loss = l1_loss(target_res, pred_res, target_bin[0])
            loss += bin_loss + res_loss

    loss = loss / len(pred_orientation)

    return loss