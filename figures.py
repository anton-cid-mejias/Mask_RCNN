"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import numpy as np
np.random.bit_generator = np.random._bit_generator
#import imgaug as ia  # https://github.com/aleju/imgaug (pip3 install imgaug)
from imgaug import augmenters as iaa

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import glob
import skimage.io
import skimage.transform
import skimage.color

# Root directory of the project
ROOT_DIR = os.path.abspath(".")#("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import save_image, show_results
from mrcnn.annotator import AnnotationsGenerator

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "../COCO_weights/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "../logs")

############################################################
#  Configurations
############################################################


class FiguresConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "figures"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Choose between resnet50 or resnet101
    BACKBONE = "resnet50"

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 5

    IMAGE_MIN_DIM = 960
    IMAGE_MAX_DIM = 960

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)#(16, 32, 64, 128, 256)#

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.9

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 500

    LEARNING_RATE = 0.0005


    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512 #256

    RPN_TRAIN_ANCHORS_PER_IMAGE = 512 #256
    PRE_NMS_LIMIT = 12000 #6000
    POST_NMS_ROIS_TRAINING = 6000 #2000

    POST_NMS_ROIS_INFERENCE = 3000 #1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    MEAN_PIXEL = np.array([0., 0., 0.])
    ORIENTATION = False


############################################################
#  Dataset
############################################################

class FiguresDataset(utils.Dataset):
    def load_figures(self, dataset_dir, filename, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        filename: The name of the file containing the annotations
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO(os.path.join(dataset_dir, filename))
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("figure", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "figure", image_id=i,
                path=os.path.join(dataset_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_orientation(self, image_id):
        image_info = self.image_info[image_id]

        instance_orientations = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            instance_orientations.append(annotation['orientation'])

        return np.array(instance_orientations)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        '''
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        '''
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FiguresDataset, self).load_mask(image_id)

    def load_mask_and_resize(self, image_id, scale, padding, crop=None):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        '''
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        '''
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)

                # Resize the mask
                m = utils.resize_single_mask(m, scale, padding, crop)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FiguresDataset, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FiguresDataset()
    dataset_train.load_figures(args.train_dataset, "train_annotations.json")
    dataset_train.prepare()

    if args.command == "validation_dataset":
        # Validation dataset
        dataset_val = FiguresDataset()
        dataset_val.load_figures(args.validation_dataset, "val_annotations.json")
        dataset_val.prepare()
    else:
        dataset_val = FiguresDataset()
        dataset_val.load_figures(args.train_dataset, "train_annotations.json")
        dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.Sequential([
        # Blur
        iaa.SomeOf((0, 1), [
            iaa.GaussianBlur(sigma=(0.0, 2.5)),
            iaa.AverageBlur(k=(2, 5))
        ]),
        # Flip
        iaa.SomeOf((0,2), [
            iaa.Fliplr(1),
            iaa.Flipud(1),
        ]),
        # Rotation and padding
        iaa.SomeOf((0,1), [
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
            iaa.Affine(rotate=(-45, 45))
        ]),
        # Light
        iaa.SomeOf((0,2), [
            iaa.GammaContrast((0.5, 2.0)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
            iaa.Multiply((0.5, 1.5)),
            iaa.ContrastNormalization((0.5, 1.5))
        ]),
        # Noise
        iaa.SomeOf((0, 1), [
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
            iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255)),
            iaa.SaltAndPepper(0.02),
        ])
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation,
                layers='heads')

def evaluate(model):

    # Load validation dataset
    dataset = FiguresDataset()
    dataset.load_figures(args.validation_dataset, "val_annotations.json")
    dataset.prepare()

    gen = AnnotationsGenerator("Dataset validation results")
    gen.add_categories(dataset.class_names)

    for image_id in dataset.image_ids:
        if config.ORIENTATION:
            image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_orientations = modellib.load_image_gt_or(dataset, config,
                                                                                      image_id, use_mini_mask=False,
                                                                                      orientation=True)
        else:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
                                                                                      use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        # Run object detection
        results = model.detect([image], config, verbose=1)
        r = results[0]
        # Save image as pred_x.png and save annotations in COCO format
        gen.add_image(info['path'], image.shape[0], image.shape[1])
        image_id = gen.get_image_id(info['path'])
        filename = info["source"] + "_" + str(image_id)

        if config.ORIENTATION:
            show_results(image, filename, r['rois'], r['masks'], r['orientations'], r['class_ids'], r['scores'],
                      dataset.class_names, gen, image_id, gt_bbox, gt_orientations, save_dir="../results/val", mode=3)
        else:
            show_results(image, filename, r['rois'], r['masks'], None, r['class_ids'], r['scores'],
                       dataset.class_names, gen, image_id, gt_bbox, None, save_dir="../results/val", mode=0)

    # Save the results in an annotation file following the COCO dataset structure
    gen.save_json("../results/val" + "/evaluation_annotations.json", pretty=True)


def detect(model, image_dir):

    # Run model detection
    print("Running predictions on {}".format(image_dir))
    files = glob.glob(image_dir + "/*")

    dataset = FiguresDataset()
    dataset.load_figures(args.validation_dataset, "val_annotations.json")
    dataset.prepare()

    gen = AnnotationsGenerator("Dataset detection results")
    gen.add_categories(dataset.class_names)

    i = 0
    for file in files:
        if file.endswith((".png", ".tiff", ".jpg", ".jpeg", ".tif", ".TIF")):
            print("Predicting \"%s\"..." % (file))
            # Load image
            image = skimage.io.imread(file)
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            # Detect objects
            results = model.detect([image], config, verbose=1)
            r = results[0]
            # Save image as pred_x.png and save annotations in COCO format
            image_name = file.replace(image_dir, "").replace("\\", "").replace("/", "")
            gen.add_image(image_name, image.shape[0], image.shape[1])
            image_id = gen.get_image_id(image_name)
            filename = "pred_%i" % i
            # Save image as pred_x.png
            if config.ORIENTATION:
                save_image(image, filename, r['rois'], r['masks'], r['orientations'], r['class_ids'], r['scores'],
                           dataset.class_names, gen, image_id, save_dir="../results/predictions", mode=4)
            else:
                save_image(image, filename, r['rois'], r['masks'], None, r['class_ids'], r['scores'],
                           dataset.class_names, gen, image_id, save_dir="../results/predictions", mode=4)
            # Save the results in an annotation file following the COCO dataset structure
            i += 1
    # Save the results in an annotation file following the COCO dataset structure
    gen.save_json("../results/predictions" + "/prediction_annotations.json", pretty=True)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect figures')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'evaluate' or 'predict'")
    parser.add_argument('--train_dataset', required=False,
                        metavar="/path/to/train_dataset/",
                        help='Directory of the Figures train dataset')
    parser.add_argument('--validation_dataset', required=False,
                        metavar="/path/to/validation_dataset/",
                        help='Directory of the Figures validation dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image_dir', required=False,
                        metavar="path to image directory",
                        help='Image directory to apply the inference on')
    args = parser.parse_args()

    print("Command: ", args.command)
    print("Weights: ", args.weights)

    # Validate arguments
    if args.command == "train":
        assert args.train_dataset, "Argument --train_dataset is required for training"
        print("Training Dataset: ", args.train_dataset)
    elif args.command == "evaluate":
        assert args.validation_dataset, "Argument --validation_dataset is required for evaluation"
        print("Validation Dataset: ", args.validation_dataset)
    elif args.command == "predict":
        assert args.image_dir, "Provide --image_dir to apply the inference"
        print("Images dir: ", args.image_dir)
        assert args.validation_dataset, "Argument --validation_dataset is required for prediction"
        print("Validation Dataset: ", args.validation_dataset)

    print("Logs: ", args.logs)

    # Configurations
    config = FiguresConfig()
    if args.command == 'evaluate' or args.command == 'predict':
        # Override the training configurations with a few
        # changes for inference.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            # Non-max suppression threshold to filter RPN proposals.
            # You can increase this during training to generate more proposals.
            RPN_NMS_THRESHOLD = 0.7
            DETECTION_MIN_CONFIDENCE = 0.8
            NUM_CLASSES = 1 + 1
            POST_NMS_ROIS_INFERENCE = 3000

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        evaluate(model)
    elif args.command == "predict":
        detect(model, args.image_dir)
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'evaluate' or 'predict'".format(args.command))
